"""Fine-tune BEATs with the BiGRU head (bFG) - the aFG recipe applied to the BEATs backbone.

Frozen BEATs (bH) added to M8 moved the leaderboard +0.0021 (B8, 1.58896); fine-tuning AST (aH -> aFG) was
worth +0.006, so the same step on BEATs is the next lever. Everything but the backbone follows aFG: all 12
blocks train (patch embedding frozen), encoder LR 2e-5, head LR 1e-3, 3 epochs, fold-train validation + Gold,
SpecAugment on the filterbank, masked BCE with the same channel weights, 10.24 s training crops, per-epoch resume.
Features: layers 4/8/12, frequency-averaged, per-frame layer norm, mean over layers, then token centres
interpolated onto the 20 ms grid exactly as features.BEATsFrames does. Clips (<= 24 s) run in one pass at
prediction time, as in BEATsFrames.

    python beats_ft_gru.py --tag bFG --fold 0 --no-eval
    python beats_ft_gru.py --smoke
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.utils.checkpoint import checkpoint

from sed import CATEGORIES, FPS, N_OUT, Head, frame_targets, hf_items, val_items

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
SR = 16000
HOP_SAMPLES = SR // FPS
CROP = int(10.24 * SR)
sys.path.insert(0, os.path.join(ROOT, "beats"))


class BEATsGru(nn.Module):
    def __init__(self, ckpt, layers=(4, 8, 12), hidden=192, dropout=0.3):
        super().__init__()
        from BEATs import BEATs, BEATsConfig
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        cfg = BEATsConfig(ck["cfg"])
        cfg.encoder_layerdrop = 0.0
        self.beats = BEATs(cfg)
        self.beats.load_state_dict(ck["model"])
        for p in self.beats.patch_embedding.parameters():
            p.requires_grad = False
        self.layers = tuple(layers)
        self.patch = cfg.input_patch_size
        self.head = Head(cfg.encoder_embed_dim, hidden=hidden, dropout=dropout)

    def fbank(self, waves):                                      # (B, S) float -> (B, frames, 128)
        return self.beats.preprocess(waves)

    def forward(self, fb, n_frames):
        """fb: (B, frames, 128) filterbank; returns (B, n_frames, 8) logits on the 20 ms grid."""
        m = self.beats
        x = m.patch_embedding(fb.unsqueeze(1))                   # (B, C, T', F')
        b, c, t_p, f_p = x.shape
        x = m.layer_norm(x.reshape(b, c, -1).transpose(1, 2))
        if m.post_extract_proj is not None:
            x = m.post_extract_proj(x)
        enc = m.encoder
        x = x + enc.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        if not enc.layer_norm_first:
            x = enc.layer_norm(x)
        x = nn.functional.dropout(x, p=enc.dropout, training=self.training).transpose(0, 1)
        taps, pos_bias = [], None
        for i, layer in enumerate(enc.layers):
            if self.training:
                x, _, pos_bias = checkpoint(lambda h, pb, l=layer: l(h, self_attn_padding_mask=None, need_weights=False,
                                                                     pos_bias=pb), x, pos_bias, use_reentrant=False)
            else:
                x, _, pos_bias = layer(x, self_attn_padding_mask=None, need_weights=False, pos_bias=pos_bias)
            if i + 1 in self.layers:
                taps.append(x)
        feats = []
        for h in taps:
            h = h.transpose(0, 1).float().reshape(b, t_p, f_p, -1).mean(2)
            feats.append((h - h.mean(-1, keepdim=True)) / (h.std(-1, keepdim=True) + 1e-5))
        h = torch.stack(feats).mean(0)                           # (B, T', D)
        t_c = (torch.arange(t_p, device=h.device) * self.patch + self.patch / 2) * 0.01 + 0.0025
        centres = (torch.arange(n_frames, device=h.device) + 0.5) / FPS
        idx = (torch.searchsorted(t_c, centres) - 1).clamp(0, max(t_p - 2, 0))
        nxt = (idx + 1).clamp(max=t_p - 1)
        den = torch.where(nxt > idx, t_c[nxt] - t_c[idx], torch.ones_like(t_c[idx]))
        w = ((centres - t_c[idx]) / den).clamp(0, 1)[None, :, None]
        h = h[:, idx] * (1 - w) + h[:, nxt] * w
        return self.head(h)


def load_wave(path):
    wave, _ = sf.read(path, dtype="float32")
    return wave.mean(1) if wave.ndim > 1 else wave


def spec_augment(x, rng, n_time=2, max_t=40, n_freq=2, max_f=16):
    b, t, f = x.shape
    for j in range(b):
        for _ in range(n_time):
            w = int(rng.integers(0, max_t + 1)); a = int(rng.integers(0, max(1, t - w)))
            x[j, a:a + w] = 0
        for _ in range(n_freq):
            w = int(rng.integers(0, max_f + 1)); a = int(rng.integers(0, max(1, f - w)))
            x[j, :, a:a + w] = 0
    return x


def make_batch(batch_items, rng):
    waves, ys = [], []
    for it in batch_items:
        wave = load_wave(it["path"])
        full = int(np.ceil(len(wave) / HOP_SAMPLES))
        a = 0
        if len(wave) > CROP:
            a = int(rng.integers(0, len(wave) - CROP + 1)) // HOP_SAMPLES * HOP_SAMPLES
        waves.append(wave[a:a + CROP])
        ys.append(frame_targets(it["events"], full)[a // HOP_SAMPLES:a // HOP_SAMPLES + CROP // HOP_SAMPLES])
    n = max(len(w) for w in waves)
    x = np.zeros((len(waves), n), np.float32)
    t = int(np.ceil(n / HOP_SAMPLES))
    y = np.zeros((len(waves), t, N_OUT), np.float32)
    m = np.zeros((len(waves), t), np.float32)
    for j, (w, tg) in enumerate(zip(waves, ys)):
        x[j, :len(w)] = w
        k = min(len(tg), t, int(np.ceil(len(w) / HOP_SAMPLES)))
        y[j, :k] = tg[:k]
        m[j, :k] = 1
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m), t


@torch.no_grad()
def predict_clip(model, path, device):
    wave = load_wave(path)
    n = int(np.ceil(len(wave) / HOP_SAMPLES))
    fb = model.fbank(torch.from_numpy(wave)[None].to(device))
    with torch.autocast("cuda", dtype=torch.float16):
        p = torch.sigmoid(model(fb, n).float())[0].cpu().numpy()
    return p.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bFG")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models", "BEATs_iter3_plus_AS2M.pt"))
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"

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

    model = BEATsGru(args.ckpt).to(device)
    enc = [p for p in model.beats.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": model.head.parameters(), "lr": args.head_lr},
                             {"params": enc, "lr": args.encoder_lr}], weight_decay=1e-2)
    steps = args.epochs * int(np.ceil(len(train_items) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.head_lr, args.encoder_lr],
                                                total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    ch_w = torch.tensor([1.0] + [0.5] * len(CATEGORIES), device=device)

    resume = os.path.join(ROOT, "models", f"beats_ftgru_{args.tag}_f{args.fold}_resume.pt")
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
            x, y, m, n = make_batch([train_items[i] for i in order[b:b + args.bs]], rng)
            x, y, m = x.to(device), y.to(device), m.to(device)
            with torch.no_grad():
                fb = spec_augment(model.fbank(x), rng)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(fb, n)
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
                "layers": model.layers}, os.path.join(ROOT, "models", f"beats_ftgru_{args.tag}_f{args.fold}.pt"))
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
    b8 = {"wsA": 0.5, "wlP": 3, "wlH": 2, "wlH2": 6, "wlH3": 1, "wlHS": 6, "aH": 8, "aFG": 8, "bH": 8}
    members = {t: pickle.load(open(os.path.join(CACHE, f"oof_{t}.pkl"), "rb")) for t in b8}
    bad = [k for k in keys if len(oof[k]) != len(members["bH"][k])]
    print(f"frame-count mismatches vs bH: {len(bad)}", flush=True)
    wsum = sum(b8.values())
    blend = {k: sum(w * np.asarray(members[t][k], np.float32) for t, w in b8.items()) / wsum for k in keys}

    def run(src):
        pred = {k: decode_clip(src[k], base, durs[k]) or fallback(src[k], base, durs[k], durs[k]) for k in keys}
        return score(ref, pred)

    print(f"  {args.tag} alone  @ v14 decode {fmt(run(oof))}")
    print(f"  bH frozen alone @ v14 decode {fmt(run({k: members['bH'][k] for k in keys}))}")
    s0 = run(blend)
    print(f"B8 mixture @ v14 decode: {fmt(s0)}")
    for wa in (4, 8):
        s = run({k: (blend[k] * wsum + wa * oof[k]) / (wsum + wa) for k in keys})
        print(f"  + {args.tag} weight {wa}: {fmt(s)} ({s['combined_score'] - s0['combined_score']:+.4f})", flush=True)


if __name__ == "__main__":
    main()
