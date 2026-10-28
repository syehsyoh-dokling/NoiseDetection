"""Fine-tune Dasheng-base with the BiGRU head (dFG) - the aFG/bFG recipe applied to Dasheng.

Frozen Dasheng (dH) on top of E_add8 moved the leaderboard +0.0016 (F4, 1.59382), with Event F1 up for once
(40 ms tokens). Fine-tuning BEATs added +0.0032 over its frozen version, so this is the same step for Dasheng.
All 12 blocks train (patch embedding, positional tables and the input BatchNorm stay frozen and in eval mode),
encoder LR 2e-5, head LR 1e-3, 3 epochs, fold-train validation + Gold, SpecAugment on the log-mel, masked BCE
with the usual channel weights, per-epoch resume. Training crops are one Dasheng chunk (1008 frames = 10.08 s);
prediction walks the clip in chunks exactly as features.DashengFrames does.

    python dasheng_ft_gru.py --tag dFG --fold 0 --no-eval
    python dasheng_ft_gru.py --smoke
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
from torch.utils.checkpoint import checkpoint

from features import dasheng_logmel
from sed import CATEGORIES, FPS, N_OUT, Head, frame_targets, hf_items, val_items

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
SR = 16000
HOP_SAMPLES = SR // FPS


class DashengGru(nn.Module):
    def __init__(self, name, layers=(4, 8, 12), hidden=192, dropout=0.3):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name, trust_remote_code=True).encoder
        for mod in (self.enc.patch_embed, self.enc.init_bn):
            for p in mod.parameters():
                p.requires_grad = False
        self.enc.time_pos_embed.requires_grad = False
        self.enc.freq_pos_embed.requires_grad = False
        self.layers = tuple(layers)
        self.chunk = self.enc.target_length                     # 1008 log-mel frames
        self.stride = self.enc.patch_stride[-1]                 # 4 frames = 40 ms per token
        self.head = Head(self.enc.embed_dim, hidden=hidden, dropout=dropout)

    def train(self, mode=True):
        super().train(mode)
        self.enc.init_bn.eval()                                 # keep the pretrained input statistics
        return self

    def tokens(self, mel):
        """mel: (B, 1, 64, <=1008) after init_bn -> list of (B, 252, D) taps, normalised per frame."""
        e = self.enc
        if mel.shape[-1] < self.chunk:
            mel = nn.functional.pad(mel, (0, self.chunk - mel.shape[-1]))
        h = e.patch_embed(mel)
        h = h + e.time_pos_embed[:, :, :, :h.shape[-1]] + e.freq_pos_embed
        h = torch.permute(torch.flatten(h, 2, 3), (0, 2, 1))
        taps = []
        for i, blk in enumerate(e.blocks):
            h = checkpoint(blk, h, use_reentrant=False) if self.training else blk(h)
            if i + 1 in self.layers:
                o = e.norm(h) if i + 1 == len(e.blocks) else h
                o = o.float()
                taps.append((o - o.mean(-1, keepdim=True)) / (o.std(-1, keepdim=True) + 1e-5))
        return torch.stack(taps).mean(0)

    def to_grid(self, tok, n_frames):
        """tok: (B, T', D) tokens of consecutive chunks -> (B, n_frames, D) on the 20 ms grid."""
        t_p = tok.shape[1]
        t_c = (torch.arange(t_p, device=tok.device) * self.stride + 1.5) * 0.01
        centres = (torch.arange(n_frames, device=tok.device) + 0.5) / FPS
        idx = (torch.searchsorted(t_c, centres) - 1).clamp(0, max(t_p - 2, 0))
        nxt = (idx + 1).clamp(max=t_p - 1)
        den = torch.where(nxt > idx, t_c[nxt] - t_c[idx], torch.ones_like(t_c[idx]))
        w = ((centres - t_c[idx]) / den).clamp(0, 1)[None, :, None]
        return tok[:, idx] * (1 - w) + tok[:, nxt] * w

    def forward(self, mel, n_frames):
        return self.head(self.to_grid(self.tokens(mel), n_frames))


def load_wave(path):
    wave, _ = sf.read(path, dtype="float32")
    return wave.mean(1) if wave.ndim > 1 else wave


def spec_augment(x, rng, n_time=2, max_t=40, n_freq=2, max_f=8):
    b, _, f, t = x.shape
    for j in range(b):
        for _ in range(n_time):
            w = int(rng.integers(0, max_t + 1)); a = int(rng.integers(0, max(1, t - w)))
            x[j, :, :, a:a + w] = 0
        for _ in range(n_freq):
            w = int(rng.integers(0, max_f + 1)); a = int(rng.integers(0, max(1, f - w)))
            x[j, :, a:a + w, :] = 0
    return x


def make_batch(model, batch_items, rng, device):
    crop = (model.chunk - 1) * 160                              # samples whose centred STFT gives <= 1008 frames
    mels, ys, lens = [], [], []
    for it in batch_items:
        wave = load_wave(it["path"])
        full = int(np.ceil(len(wave) / HOP_SAMPLES))
        a = 0
        if len(wave) > crop:
            a = int(rng.integers(0, len(wave) - crop + 1)) // HOP_SAMPLES * HOP_SAMPLES
        w = wave[a:a + crop]
        with torch.no_grad():
            m = model.enc.init_bn(dasheng_logmel(w, device)[:, None])   # (1, 1, 64, T)
        mels.append(nn.functional.pad(m, (0, model.chunk - m.shape[-1])))
        ys.append(frame_targets(it["events"], full)[a // HOP_SAMPLES:a // HOP_SAMPLES + crop // HOP_SAMPLES + 1])
        lens.append(int(np.ceil(len(w) / HOP_SAMPLES)))
    n = int(np.ceil(crop / HOP_SAMPLES))
    y = np.zeros((len(ys), n, N_OUT), np.float32)
    msk = np.zeros((len(ys), n), np.float32)
    for j, (tg, L) in enumerate(zip(ys, lens)):
        k = min(len(tg), L, n)
        y[j, :k] = tg[:k]
        msk[j, :k] = 1
    return torch.cat(mels), torch.from_numpy(y), torch.from_numpy(msk), n


@torch.no_grad()
def predict_clip(model, path, device):
    wave = load_wave(path)
    n = int(np.ceil(len(wave) / HOP_SAMPLES))
    mel = model.enc.init_bn(dasheng_logmel(wave, device)[:, None])
    toks = []
    with torch.autocast("cuda", dtype=torch.float16):
        for a in range(0, mel.shape[-1], model.chunk):
            c = mel[..., a:a + model.chunk]
            n_tok = int(np.ceil(c.shape[-1] / model.stride))
            toks.append(model.tokens(c)[:, :n_tok])
        p = torch.sigmoid(model.head(model.to_grid(torch.cat(toks, 1), n)).float())[0]
    return p.cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dFG")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--backbone", default=os.path.join(ROOT, "models", "dasheng-base"))
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpu-frac", type=float, default=1.0,
                    help="share of the GPU to use: caps this process's GPU memory at that fraction and, after each "
                         "training step, idles for (1/frac - 1) x the step time so the GPU is busy ~frac of the time")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"
    if args.gpu_frac < 1.0:
        torch.cuda.set_per_process_memory_fraction(args.gpu_frac)
        print(f"GPU share capped at {args.gpu_frac:.0%} (memory limit and duty cycle)", flush=True)

    items = val_items(os.path.join(ROOT, "data", "validation"))
    groups = np.array([it["group"] for it in items])
    tr, te = list(GroupKFold(n_splits=5).split(items, groups=groups))[args.fold]
    train_items = [items[i] for i in tr] + hf_items(os.path.join(ROOT, "data", "hf_gold"))
    held = [items[i] for i in te]
    if args.smoke:
        train_items = [train_items[i] for i in rng.choice(len(train_items), 64, replace=False)]
        held = held[:60]
        args.epochs, args.log_every, args.tag = 1, 4, args.tag + "_smoke"
    print(f"fold {args.fold}: train {len(train_items)} clips, held out {len(held)}", flush=True)

    model = DashengGru(args.backbone).to(device)
    enc = [p for p in model.enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": model.head.parameters(), "lr": args.head_lr},
                             {"params": enc, "lr": args.encoder_lr}], weight_decay=1e-2)
    steps = args.epochs * int(np.ceil(len(train_items) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.head_lr, args.encoder_lr],
                                                total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    ch_w = torch.tensor([1.0] + [0.5] * len(CATEGORIES), device=device)

    resume = os.path.join(ROOT, "models", f"dasheng_ftgru_{args.tag}_f{args.fold}_resume.pt")
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
            t_step = time.time()
            mel, y, m, n = make_batch(model, [train_items[i] for i in order[b:b + args.bs]], rng, device)
            mel = spec_augment(mel, rng)
            y, m = y.to(device), m.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(mel, n)
            loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
            loss = ((loss * ch_w).sum(-1) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(mel); seen += len(mel); step += 1
            if args.gpu_frac < 1.0:
                torch.cuda.synchronize()
                time.sleep((time.time() - t_step) * (1.0 / args.gpu_frac - 1.0))
            if step % args.log_every == 0:
                el = (time.time() - t0) / (step - step0)
                print(f"  epoch {ep + 1} step {step}/{steps} loss {tot / seen:.4f} {el:.2f} s/step, "
                      f"eta {el * (steps - step) / 60:.0f} min, gpu peak "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        print(f"epoch {ep + 1}/{args.epochs} loss {tot / seen:.4f} [{(time.time() - t0) / 60:.1f} min]", flush=True)
        if ep + 1 < args.epochs:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "step": step, "epoch": ep + 1}, resume + ".tmp")
            os.replace(resume + ".tmp", resume)
    torch.save({"state": model.state_dict(), "tag": args.tag, "fold": args.fold, "epochs": args.epochs,
                "layers": model.layers}, os.path.join(ROOT, "models", f"dasheng_ftgru_{args.tag}_f{args.fold}.pt"))
    if os.path.exists(resume):
        os.remove(resume)

    model.eval()
    oof = {it["key"]: predict_clip(model, it["path"], device) for it in held}
    pickle.dump(oof, open(os.path.join(CACHE, f"oof_{args.tag}_fold{args.fold}.pkl"), "wb"))
    print(f"held-out probabilities written for {len(oof)} clips", flush=True)
    if args.no_eval:
        return

    from ast_ft_gru import fmt
    from calib import fallback
    from postproc import decode_clip
    from scorer import score
    base = json.load(open(os.path.join(ROOT, "models", "fusion_A1.json")))["sed_nat"]
    nat = [it for it in held if not it["synthetic"]]
    keys = [it["key"] for it in nat]
    ref = {it["key"]: [(s, e) for s, e, _ in it["events"]] for it in nat}
    durs = {it["key"]: it["dur"] for it in nat}
    f4 = {"wsA": 0.5, "wlP": 3, "wlH": 2, "wlH2": 6, "wlH3": 1, "wlHS": 6, "aH": 8, "aFG": 8, "bH": 8, "bFG": 8, "dH": 4}
    members = {t: pickle.load(open(os.path.join(CACHE, f"oof_{t}.pkl"), "rb")) for t in f4}
    print(f"frame-count mismatches vs dH: {sum(len(oof[k]) != len(members['dH'][k]) for k in keys)}", flush=True)
    wsum = sum(f4.values())
    blend = {k: sum(w * np.asarray(members[t][k], np.float32) for t, w in f4.items()) / wsum for k in keys}

    def run(src):
        pred = {k: decode_clip(src[k], base, durs[k]) or fallback(src[k], base, durs[k], durs[k]) for k in keys}
        return score(ref, pred)

    print(f"  {args.tag} alone @ v14 decode {fmt(run(oof))}")
    s0 = run(blend)
    print(f"F4 mixture @ v14 decode: {fmt(s0)}")
    for wa in (4, 8):
        s = run({k: (blend[k] * wsum + wa * oof[k]) / (wsum + wa) for k in keys})
        print(f"  + {args.tag} weight {wa}: {fmt(s)} ({s['combined_score'] - s0['combined_score']:+.4f})", flush=True)


if __name__ == "__main__":
    main()
