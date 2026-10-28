"""A fair fold-0 probe of fine-tuning AST, built to be compared with the frozen aH member.

Every candidate that can be built from the existing members lands within +-0.001 of v14
(K1 1.4474, K2 1.4456, K3 1.4448 on validation with v14's decode), while rank 1 is 0.0143
ahead on the leaderboard. That gap needs a stronger model, and the one large lever never
tested fairly is updating the encoder: the earlier probe (afTq) trained only 4 blocks for 2
epochs with a plain linear head, and scored 0.8372 against the frozen aH's 1.0837.

This probe changes only the thing under test. It keeps aH's recipe everywhere else:
  - the same features: patch tokens of layers 4/8/12 averaged over frequency, per-frame
    layer norm, mean over layers (sed.compact), aligned onto the 20 ms grid exactly as
    features.ASTFrames does (token centres, not a uniform stretch);
  - the same BiGRU head (sed.Head), channel weights and masked BCE as sed.train_head;
  - the same training data: fold-0 validation-train clips plus all Gold clips;
and differs only in that all 12 AST blocks train (encoder LR 2e-5, head LR 1e-3, fp16,
gradient checkpointing, SpecAugment on the filterbank). Measured earlier on this GPU: 4.86
GiB and 18 clips/s at batch 8.

Output: fold-0 out-of-fold probabilities (cache/oof_<tag>_fold0.pkl) and, on the held-out
natural clips, a comparison with aH at v14's decode and at each model's own best decode,
plus the effect of adding the new stream to the K1 mixture (v14's weights without aA).

    python ast_ft_gru.py --tag aFG --epochs 3
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

from calib import fallback
from postproc import decode_clip
from scorer import score
from sed import CATEGORIES, FPS, N_OUT, Head, frame_targets, hf_items, val_items

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
SR = 16000
HOP_SAMPLES = SR // FPS            # 320 samples per 20 ms output frame


class ASTGru(nn.Module):
    def __init__(self, name, layers=(4, 8, 12), hidden=192, dropout=0.3):
        super().__init__()
        from transformers import ASTModel
        self.ast = ASTModel.from_pretrained(name)
        c = self.ast.config
        self.f_dim = (c.num_mel_bins - c.patch_size) // c.frequency_stride + 1
        self.t_dim = (c.max_length - c.patch_size) // c.time_stride + 1
        self.win = c.max_length * 160                     # samples per 10.24 s window
        self.n_frames = self.win // HOP_SAMPLES           # 512 output frames per window
        self.layers = tuple(layers)
        for p in self.ast.embeddings.parameters():         # patch/position embeddings stay fixed
            p.requires_grad = False
        self.ast.gradient_checkpointing_enable()
        self.head = Head(c.hidden_size, hidden=hidden, dropout=dropout)
        # same token-to-frame alignment as features.ASTFrames
        t_c = (np.arange(self.t_dim) * c.time_stride + c.patch_size / 2) * 0.01
        centres = (np.arange(self.n_frames) + 0.5) / FPS
        idx = np.clip(np.searchsorted(t_c, centres) - 1, 0, self.t_dim - 2)
        w = np.clip((centres - t_c[idx]) / (t_c[idx + 1] - t_c[idx]), 0, 1)
        self.register_buffer("idx", torch.from_numpy(idx).long(), persistent=False)
        self.register_buffer("w", torch.from_numpy(w).float(), persistent=False)

    def forward(self, x):                                  # x: (B, 1024, 128) filterbank
        hs = self.ast(x, output_hidden_states=True).hidden_states
        feats = []
        for l in self.layers:
            h = hs[l][:, -self.f_dim * self.t_dim:]
            b, _, d = h.shape
            h = h.view(b, self.f_dim, self.t_dim, d).mean(1).float()
            feats.append((h - h.mean(-1, keepdim=True)) / (h.std(-1, keepdim=True) + 1e-5))
        h = torch.stack(feats).mean(0)                     # (B, t_dim, D)
        w = self.w[None, :, None]
        h = h[:, self.idx] * (1 - w) + h[:, self.idx + 1] * w   # (B, n_frames, D)
        return self.head(h)


def spec_augment(x, rng, n_time=2, max_t=40, n_freq=2, max_f=16):
    b, t, f = x.shape
    for j in range(b):
        for _ in range(n_time):
            wdt = int(rng.integers(0, max_t + 1)); a = int(rng.integers(0, max(1, t - wdt)))
            x[j, a:a + wdt] = 0
        for _ in range(n_freq):
            wdt = int(rng.integers(0, max_f + 1)); a = int(rng.integers(0, max(1, f - wdt)))
            x[j, :, a:a + wdt] = 0
    return x


def make_batch(fe, model, batch_items, rng, train=True):
    xs, ys, ms = [], [], []
    for it in batch_items:
        wave, _ = sf.read(it["path"], dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(1)
        full = int(np.ceil(len(wave) / HOP_SAMPLES))
        a = 0
        if train and len(wave) > model.win:
            a = int(rng.integers(0, len(wave) - model.win + 1)) // HOP_SAMPLES * HOP_SAMPLES
        chunk = wave[a:a + model.win]
        xs.append(fe(chunk, sampling_rate=SR, return_tensors="np")["input_values"][0])
        y = np.zeros((model.n_frames, N_OUT), np.float32)
        m = np.zeros(model.n_frames, np.float32)
        tgt = frame_targets(it["events"], full)[a // HOP_SAMPLES:a // HOP_SAMPLES + model.n_frames]
        y[:len(tgt)] = tgt
        m[:len(tgt)] = 1
        ys.append(y); ms.append(m)
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(ms)))


@torch.no_grad()
def predict_clip(fe, model, path, device):
    """(T, 8) probabilities on the 20 ms grid; long clips use 10.24 s windows, the last one
    ending at the clip end, with each frame read from exactly one window (as ASTFrames)."""
    wave, _ = sf.read(path, dtype="float32")
    if wave.ndim > 1:
        wave = wave.mean(1)
    n = int(np.ceil(len(wave) / HOP_SAMPLES))
    centres = (np.arange(n) + 0.5) / FPS
    n_chunks = max(1, int(np.ceil(len(wave) / model.win)))
    owner = np.minimum((centres * SR // model.win).astype(int), n_chunks - 1)
    out = np.zeros((n, N_OUT), np.float32)
    for ci in range(n_chunks):
        a = ci * model.win
        if ci == n_chunks - 1 and ci > 0:
            a = max(0, len(wave) - model.win)
        x = torch.from_numpy(fe(wave[a:a + model.win], sampling_rate=SR, return_tensors="np")["input_values"])
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(x.to(device)).float())[0].cpu().numpy()
        sel = np.where(owner == ci)[0]
        local = np.clip(np.round((centres[sel] - a / SR) * FPS - 0.5).astype(int), 0, model.n_frames - 1)
        out[sel] = p[local]
    return out


def best_decode(src, keys, ref, durs, base):
    grids = [("thr_det", [0.2, 0.3, 0.4, 0.5]), ("alpha", [0.4, 0.5, 0.6, 0.7]), ("floor", [0.1, 0.15, 0.25]),
             ("o_thr_det", [0.2, 0.3, 0.4, 0.5]), ("o_alpha", [0.4, 0.5, 0.6]), ("o_floor", [0.1, 0.2, 0.3]),
             ("o_gap", [0.6, 1.5, 2.5]), ("med", [3, 5, 7]), ("o_med", [3, 5, 7])]

    def run(cfg):
        pred = {k: decode_clip(src[k], cfg, durs[k]) or fallback(src[k], cfg, durs[k], durs[k]) for k in keys}
        return score({k: ref[k] for k in keys}, pred)

    at_base = run(base)
    cur, best = dict(base), at_base["combined_score"]
    for _ in range(2):
        for g, vals in grids:
            old, keep = cur[g], None
            for v in vals:
                if v == old:
                    continue
                cur[g] = v
                s = run(cur)["combined_score"]
                if s > best + 1e-9:
                    best, keep = s, v
            cur[g] = keep if keep is not None else old
    return at_base, run(cur)


def fmt(s):
    return f"combined {s['combined_score']:.4f} f1 {s['event_f1']:.4f} dice {s['segment_dice']:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="aFG")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--backbone", default=os.path.join(ROOT, "models", "ast-audioset"))
    ap.add_argument("--no-eval", action="store_true",
                    help="stop after writing the out-of-fold pickle (Kaggle runs lack the other members' pickles)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch on 64 training clips and 60 held-out clips, to exercise training, "
                         "prediction and every evaluation step before committing hours of GPU")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"

    from transformers import ASTFeatureExtractor
    fe = ASTFeatureExtractor.from_pretrained(args.backbone)
    items = val_items(os.path.join(ROOT, "data", "validation"))
    groups = np.array([it["group"] for it in items])
    tr, te = list(GroupKFold(n_splits=5).split(items, groups=groups))[args.fold]
    gold = hf_items(os.path.join(ROOT, "data", "hf_gold"))
    train_items = [items[i] for i in tr] + gold
    held = [items[i] for i in te]
    if args.smoke:
        train_items = [train_items[i] for i in rng.choice(len(train_items), 64, replace=False)]
        held = held[:60]
        args.epochs, args.log_every, args.tag = 1, 4, args.tag + "_smoke"
    print(f"fold {args.fold}: train {len(train_items)} clips ({len(tr)} validation + {len(gold)} Gold), "
          f"held out {len(held)}", flush=True)

    model = ASTGru(args.backbone).to(device)
    enc = [p for p in model.ast.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": model.head.parameters(), "lr": args.head_lr},
                             {"params": enc, "lr": args.encoder_lr}], weight_decay=1e-2)
    steps = args.epochs * int(np.ceil(len(train_items) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.head_lr, args.encoder_lr],
                                                total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    ch_w = torch.tensor([1.0] + [0.5] * len(CATEGORIES), device=device)
    # per-epoch resume point: other sessions' jobs grab the GPU/RAM now and then and kill the run
    resume = os.path.join(ROOT, "models", f"ast_ftgru_{args.tag}_f{args.fold}_resume.pt")
    step, start_ep = 0, 0
    if os.path.exists(resume):
        r = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(r["model"]); opt.load_state_dict(r["opt"])
        sched.load_state_dict(r["sched"]); scaler.load_state_dict(r["scaler"])
        rng.bit_generator.state = r["rng"]; torch.set_rng_state(r["torch_rng"])
        step, start_ep = r["step"], r["epoch"]
        print(f"resumed from epoch {start_ep} (step {step})", flush=True)
    step0, t0 = step, time.time()
    for ep in range(start_ep, args.epochs):
        model.train()
        order = rng.permutation(len(train_items))
        tot, seen = 0.0, 0
        for b in range(0, len(order), args.bs):
            x, y, m = make_batch(fe, model, [train_items[i] for i in order[b:b + args.bs]], rng, train=True)
            x = spec_augment(x, rng)
            x, y, m = x.to(device), y.to(device), m.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
            loss = ((loss * ch_w).sum(-1) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(x); seen += len(x); step += 1
            if step % args.log_every == 0:
                el = (time.time() - t0) / (step - step0)
                print(f"  epoch {ep + 1} step {step}/{steps} loss {tot / seen:.4f} "
                      f"{el:.2f} s/step, eta {el * (steps - step) / 60:.0f} min", flush=True)
        print(f"epoch {ep + 1}/{args.epochs} loss {tot / seen:.4f} [{(time.time() - t0) / 60:.1f} min]", flush=True)
        if ep + 1 < args.epochs:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "step": step, "epoch": ep + 1}, resume + ".tmp")
            os.replace(resume + ".tmp", resume)
    torch.save({"state": model.state_dict(), "tag": args.tag, "fold": args.fold, "epochs": args.epochs,
                "encoder_lr": args.encoder_lr, "head_lr": args.head_lr, "layers": model.layers},
               os.path.join(ROOT, "models", f"ast_ftgru_{args.tag}_f{args.fold}.pt"))
    if os.path.exists(resume):
        os.remove(resume)

    model.eval()
    oof = {it["key"]: predict_clip(fe, model, it["path"], device) for it in held}
    pickle.dump(oof, open(os.path.join(CACHE, f"oof_{args.tag}_fold{args.fold}.pkl"), "wb"))
    print(f"held-out probabilities written for {len(oof)} clips", flush=True)
    if args.no_eval:
        return

    base = json.load(open(os.path.join(ROOT, "models", "fusion_A1.json")))["sed_nat"]
    nat = [it for it in held if not it["synthetic"]]
    keys = [it["key"] for it in nat]
    ref = {it["key"]: [(s, e) for s, e, _ in it["events"]] for it in nat}
    durs = {it["key"]: it["dur"] for it in nat}
    aH = pickle.load(open(os.path.join(CACHE, "oof_aH.pkl"), "rb"))
    print(f"fold {args.fold} natural held-out clips: {len(keys)}")
    for name, src in ((f"{args.tag} fine-tuned AST+GRU", oof), ("aH frozen AST+Gold", aH)):
        at_v14, own = best_decode(src, keys, ref, durs, base)
        print(f"  {name:24s} @ v14 decode {fmt(at_v14)} | @ own best decode {fmt(own)}", flush=True)

    k1 = {"wsA": 0.5, "wlP": 3, "wlH": 2, "wlH2": 6, "wlH3": 1, "wlHS": 6, "aH": 8}
    members = {t: pickle.load(open(os.path.join(CACHE, f"oof_{t}.pkl"), "rb")) for t in k1}
    wsum = sum(k1.values())
    blend = {k: sum(w * np.asarray(members[t][k], np.float32) for t, w in k1.items()) / wsum for k in keys}

    def run_blend(extra_w):
        pred = {}
        for k in keys:
            p = blend[k] if not extra_w else (blend[k] * wsum + extra_w * oof[k]) / (wsum + extra_w)
            pred[k] = decode_clip(p, base, durs[k]) or fallback(p, base, durs[k], durs[k])
        return score(ref, pred)

    s0 = run_blend(0)
    print(f"K1 mixture (v14 weights without aA) @ v14 decode: {fmt(s0)}")
    for wa in (0.5, 1, 2, 4, 8):
        s = run_blend(wa)
        print(f"  + {args.tag} weight {wa:<3}: {fmt(s)} ({s['combined_score'] - s0['combined_score']:+.4f})", flush=True)


if __name__ == "__main__":
    main()
