"""Fine-tune the top of Whisper large-v3 with the BiGRU head (wFG), the aFG recipe applied to Whisper.

aFG (all 12 AST blocks trained) added to K1 at weight 8 moved the leaderboard +0.0060 (M8, 1.5868),
the first gain that transferred since v14. Whisper large-v3 carries most of the ensemble weight
(wlP 3, wlH 2, wlH2 6, wlH3 1, wlHS 6 - all frozen), so the same move on Whisper is the next lever.

An 8 GB GPU cannot train all 32 layers (635M parameters), so layers 1-24 stay frozen in fp16 and run
without gradients, and layers 25-32 plus the final layer norm train in fp32 under autocast with
gradient checkpointing. Everything else follows the frozen Whisper members and aFG:
  - features: layers 8/16/24/32 (the last after the encoder layer norm), per-frame layer norm,
    mean over layers (sed.compact), on Whisper's native 20 ms grid;
  - head, channel weights, masked BCE, data (fold-train validation + Gold), 3 epochs,
    encoder LR 2e-5, head LR 1e-3, OneCycle, SpecAugment on the log-mel;
  - training crops of at most 10 s; batches are zero-padded in audio, as Whisper was trained;
  - per-epoch resume point, as ast_ft_gru.py.

Output: models/whisper_ftgru_<tag>_f<fold>.pt (trained layers and head only), out-of-fold
probabilities cache/oof_<tag>_fold<fold>.pkl, and on the held-out natural clips a comparison
with the frozen wlH2 member and the effect of adding wFG to M8 (the current best, 1.5868).

    python whisper_ft_gru.py --tag wFG --fold 0
    python whisper_ft_gru.py --smoke
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

from ast_ft_gru import best_decode, fmt
from calib import fallback
from features import load_encoder
from postproc import decode_clip
from scorer import score
from sed import CATEGORIES, FPS, N_OUT, Head, frame_targets, hf_items, val_items

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "cache")
SR = 16000
HOP_SAMPLES = SR // FPS            # 320 samples per 20 ms encoder frame
CROP = 10 * SR                     # training crop


class WhisperGru(nn.Module):
    def __init__(self, name, layers=(8, 16, 24, 32), n_train=8, hidden=192, dropout=0.3):
        super().__init__()
        self.enc = load_encoder(name)                      # fp16
        self.layers = tuple(layers)
        self.n_frozen = len(self.enc.layers) - n_train
        for p in self.enc.parameters():
            p.requires_grad = False
        for mod in list(self.enc.layers[self.n_frozen:]) + [self.enc.layer_norm]:
            mod.float()
            for p in mod.parameters():
                p.requires_grad = True
        self.max_frames = self.enc.embed_positions.num_embeddings   # 1500 = 30 s
        self.head = Head(self.enc.config.d_model, hidden=hidden, dropout=dropout)

    def trainable_state(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in keep or k.startswith("head.")}

    @staticmethod
    def _layer(layer, h):
        out = layer(h, None)
        return out[0] if isinstance(out, tuple) else out

    def forward(self, mel):                                 # mel: (B, n_mels, 2T)
        e, taps = self.enc, []
        with torch.no_grad():
            h = nn.functional.gelu(e.conv1(mel.half()))
            h = nn.functional.gelu(e.conv2(h)).permute(0, 2, 1)
            h = h + e.embed_positions.weight[: h.shape[1]].to(h.dtype)
            for i in range(self.n_frozen):
                h = self._layer(e.layers[i], h)
                if i + 1 in self.layers:
                    taps.append(h.float())
        h = h.float()
        with torch.autocast("cuda", dtype=torch.float16):
            for i in range(self.n_frozen, len(e.layers)):
                if self.training:
                    h = checkpoint(self._layer, e.layers[i], h, use_reentrant=False)
                else:
                    h = self._layer(e.layers[i], h)
                if i + 1 in self.layers and i + 1 != len(e.layers):
                    taps.append(h.float())
            h = e.layer_norm(h.float())
        taps.append(h.float())
        feats = [(t - t.mean(-1, keepdim=True)) / (t.std(-1, keepdim=True) + 1e-5) for t in taps]
        with torch.autocast("cuda", dtype=torch.float16):
            return self.head(torch.stack(feats).mean(0)).float()


def mel_batch(fe, waves):
    """Zero-pad waves to a common even number of 20 ms frames, then log-mel (B, n_mels, 2T)."""
    n = max(len(w) for w in waves)
    n = int(np.ceil(n / HOP_SAMPLES)) * HOP_SAMPLES
    padded = [np.pad(w, (0, n - len(w))) for w in waves]
    mel = fe(padded, sampling_rate=SR, return_tensors="np", padding="do_not_pad")["input_features"]
    t = n // HOP_SAMPLES * 2
    if mel.shape[-1] < t:
        mel = np.pad(mel, ((0, 0), (0, 0), (0, t - mel.shape[-1])), mode="edge")
    return torch.from_numpy(mel[..., :t].astype(np.float32))


def load_wave(path):
    wave, _ = sf.read(path, dtype="float32")
    return wave.mean(1) if wave.ndim > 1 else wave


def make_batch(fe, batch_items, rng):
    waves, ys, ms = [], [], []
    for it in batch_items:
        wave = load_wave(it["path"])
        full = int(np.ceil(len(wave) / HOP_SAMPLES))
        a = 0
        if len(wave) > CROP:
            a = int(rng.integers(0, len(wave) - CROP + 1)) // HOP_SAMPLES * HOP_SAMPLES
        waves.append(wave[a:a + CROP])
        ys.append(frame_targets(it["events"], full)[a // HOP_SAMPLES:a // HOP_SAMPLES + CROP // HOP_SAMPLES])
    x = mel_batch(fe, waves)
    t = x.shape[-1] // 2
    y = np.zeros((len(waves), t, N_OUT), np.float32)
    m = np.zeros((len(waves), t), np.float32)
    for j, tg in enumerate(ys):
        n = min(len(tg), t)
        y[j, :n] = tg[:n]
        m[j, :n] = 1
    return x, torch.from_numpy(y), torch.from_numpy(m)


def spec_augment(x, rng, n_time=2, max_t=40, n_freq=2, max_f=16):
    b, f, t = x.shape
    for j in range(b):
        for _ in range(n_time):
            w = int(rng.integers(0, max_t + 1)); a = int(rng.integers(0, max(1, t - w)))
            x[j, :, a:a + w] = x[j].min()
        for _ in range(n_freq):
            w = int(rng.integers(0, max_f + 1)); a = int(rng.integers(0, max(1, f - w)))
            x[j, a:a + w, :] = x[j].min()
    return x


@torch.no_grad()
def predict_clip(fe, model, path, device):
    """(T, 8) probabilities on the 20 ms grid; clips longer than 30 s go in 30 s pieces."""
    wave = load_wave(path)
    n = int(np.ceil(len(wave) / HOP_SAMPLES))
    step = model.max_frames * HOP_SAMPLES
    outs = []
    for a in range(0, len(wave), step):
        x = mel_batch(fe, [wave[a:a + step]]).to(device)
        outs.append(torch.sigmoid(model(x))[0].cpu().numpy())
    p = np.concatenate(outs)
    if len(p) < n:
        p = np.pad(p, ((0, n - len(p)), (0, 0)), mode="edge")
    return p[:n].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="wFG")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--backbone", default=os.path.join(ROOT, "models", "whisper-large-v3"))
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch on 64 training clips and 60 held-out clips, end to end")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"

    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained(args.backbone)
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
    print(f"fold {args.fold}: train {len(train_items)} clips, held out {len(held)}", flush=True)

    model = WhisperGru(args.backbone, n_train=args.n_train).to(device)
    enc = [p for p in model.enc.parameters() if p.requires_grad]
    print(f"trainable encoder params {sum(p.numel() for p in enc) / 1e6:.0f}M "
          f"(layers {model.n_frozen + 1}-{len(model.enc.layers)})", flush=True)
    opt = torch.optim.AdamW([{"params": model.head.parameters(), "lr": args.head_lr},
                             {"params": enc, "lr": args.encoder_lr}], weight_decay=1e-2)
    steps = args.epochs * int(np.ceil(len(train_items) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.head_lr, args.encoder_lr],
                                                total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    ch_w = torch.tensor([1.0] + [0.5] * len(CATEGORIES), device=device)
    params = [p for p in model.parameters() if p.requires_grad]

    resume = os.path.join(ROOT, "models", f"whisper_ftgru_{args.tag}_f{args.fold}_resume.pt")
    step, start_ep = 0, 0
    if os.path.exists(resume):
        r = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(r["model"], strict=False); opt.load_state_dict(r["opt"])
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
            x, y, m = make_batch(fe, [train_items[i] for i in order[b:b + args.bs]], rng)
            x = spec_augment(x, rng)
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            loss = ((loss * ch_w).sum(-1) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(x); seen += len(x); step += 1
            if step % args.log_every == 0:
                el = (time.time() - t0) / (step - step0)
                print(f"  epoch {ep + 1} step {step}/{steps} loss {tot / seen:.4f} {el:.2f} s/step, "
                      f"eta {el * (steps - step) / 60:.0f} min, gpu peak "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        print(f"epoch {ep + 1}/{args.epochs} loss {tot / seen:.4f} [{(time.time() - t0) / 60:.1f} min]", flush=True)
        if ep + 1 < args.epochs:
            torch.save({"model": model.trainable_state(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "step": step, "epoch": ep + 1}, resume + ".tmp")
            os.replace(resume + ".tmp", resume)
    torch.save({"state": model.trainable_state(), "tag": args.tag, "fold": args.fold, "epochs": args.epochs,
                "n_train": args.n_train, "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
                "layers": model.layers}, os.path.join(ROOT, "models", f"whisper_ftgru_{args.tag}_f{args.fold}.pt"))
    if os.path.exists(resume):
        os.remove(resume)

    model.eval()
    oof = {it["key"]: predict_clip(fe, model, it["path"], device) for it in held}
    pickle.dump(oof, open(os.path.join(CACHE, f"oof_{args.tag}_fold{args.fold}.pkl"), "wb"))
    print(f"held-out probabilities written for {len(oof)} clips", flush=True)

    base = json.load(open(os.path.join(ROOT, "models", "fusion_A1.json")))["sed_nat"]
    nat = [it for it in held if not it["synthetic"]]
    keys = [it["key"] for it in nat]
    ref = {it["key"]: [(s, e) for s, e, _ in it["events"]] for it in nat}
    durs = {it["key"]: it["dur"] for it in nat}
    bad = [k for k in keys if len(oof[k]) != len(pickle.load(open(os.path.join(CACHE, "oof_wlH2.pkl"), "rb"))[k])] \
        if args.smoke else []
    if bad:
        print(f"WARNING: {len(bad)} clips differ in frame count from wlH2, e.g. {bad[0]}", flush=True)
    wl = pickle.load(open(os.path.join(CACHE, "oof_wlH2.pkl"), "rb"))
    print(f"fold {args.fold} natural held-out clips: {len(keys)}")
    for name, src in ((f"{args.tag} fine-tuned Whisper+GRU", oof), ("wlH2 frozen Whisper", wl)):
        at_v14, own = best_decode(src, keys, ref, durs, base)
        print(f"  {name:28s} @ v14 decode {fmt(at_v14)} | @ own best decode {fmt(own)}", flush=True)

    m8 = {"wsA": 0.5, "wlP": 3, "wlH": 2, "wlH2": 6, "wlH3": 1, "wlHS": 6, "aH": 8, "aFG": 8}
    members = {t: pickle.load(open(os.path.join(CACHE, f"oof_{t}.pkl"), "rb")) for t in m8}
    wsum = sum(m8.values())
    blend = {k: sum(w * np.asarray(members[t][k], np.float32) for t, w in m8.items()) / wsum for k in keys}

    def run_blend(extra_w):
        pred = {}
        for k in keys:
            p = blend[k] if not extra_w else (blend[k] * wsum + extra_w * oof[k]) / (wsum + extra_w)
            pred[k] = decode_clip(p, base, durs[k]) or fallback(p, base, durs[k], durs[k])
        return score(ref, pred)

    s0 = run_blend(0)
    print(f"M8 mixture (current best) @ v14 decode: {fmt(s0)}")
    for wa in (2, 4, 8, 12):
        s = run_blend(wa)
        print(f"  + {args.tag} weight {wa:<3}: {fmt(s)} ({s['combined_score'] - s0['combined_score']:+.4f})", flush=True)


if __name__ == "__main__":
    main()
