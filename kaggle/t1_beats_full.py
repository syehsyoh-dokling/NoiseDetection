"""Full-data BEATs fine-tune (bFS) for Track 1: all validation + Gold + Silver (loss weight SILVER_W), no CV hold-out.

Same model, crops, augmentation, optimiser and channel weights as beats_ft_gru.py (the bFG member, LB-proven); the
only changes are the training set (all 2501 validation clips, Gold, and Silver from data/hf_silver at a lower loss
weight because its timestamps are unverified) and a wall-clock guard. There is no out-of-fold estimate: the test
probabilities are judged on the leaderboard by swapping/adding this member in the fusion.

Env: TAG (bFS), EPOCHS (2), SILVER_W (0.4), MAX_HOURS (3.0). Writes cache/test_probs_<TAG>_full.pkl.
"""
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beats_ft_gru as B  # noqa: E402
from run import test_clips  # noqa: E402
from sed import CATEGORIES, hf_items, val_items  # noqa: E402

ROOT = B.ROOT


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    tag = os.environ.get("TAG", "bFS")
    epochs = int(os.environ.get("EPOCHS", "2"))
    sw = float(os.environ.get("SILVER_W", "0.4"))
    max_h = float(os.environ.get("MAX_HOURS", "3.0"))
    bs, head_lr, enc_lr = 8, 1e-3, 2e-5
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    val = val_items(os.path.join(ROOT, "data", "validation"))
    gold = hf_items(os.path.join(ROOT, "data", "hf_gold"))
    silver = hf_items(os.path.join(ROOT, "data", "hf_silver"))
    for it in val + gold:
        it["w"] = 1.0
    for it in silver:
        it["w"] = sw
    train = val + gold + silver
    log(f"train clips: {len(val)} validation + {len(gold)} Gold + {len(silver)} Silver (w {sw}) = {len(train)}")

    model = B.BEATsGru(os.path.join(ROOT, "models", "BEATs_iter3_plus_AS2M.pt")).to("cuda")
    enc = [p for p in model.beats.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": model.head.parameters(), "lr": head_lr},
                             {"params": enc, "lr": enc_lr}], weight_decay=1e-2)
    steps = epochs * int(np.ceil(len(train) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[head_lr, enc_lr], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    ch_w = torch.tensor([1.0] + [0.5] * len(CATEGORIES), device="cuda")
    t0, step, stop = time.time(), 0, False
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        tot, seen = 0.0, 0
        for b in range(0, len(order), bs):
            batch = [train[i] for i in order[b:b + bs]]
            x, y, m, n = B.make_batch(batch, rng)
            x, y, m = x.cuda(), y.cuda(), m.cuda()
            wb = torch.tensor([it["w"] for it in batch], device="cuda")[:, None]
            with torch.no_grad():
                fb = B.spec_augment(model.fbank(x), rng)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(fb, n)
            el = nn.functional.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
            per = (el * ch_w).sum(-1) * m * wb
            loss = per.sum() / (m * wb).sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(x); seen += len(x); step += 1
            if step % 200 == 0:
                sp = (time.time() - t0) / step
                log(f"epoch {ep + 1} step {step}/{steps} loss {tot / seen:.4f} {sp:.2f} s/step "
                    f"eta {sp * (steps - step) / 60:.0f} min")
            if time.time() - t0 > max_h * 3600:
                log(f"wall-clock guard hit at step {step}/{steps}"); stop = True; break
        log(f"epoch {ep + 1} done, loss {tot / max(seen, 1):.4f}, {(time.time() - t0) / 60:.1f} min")
        if stop:
            break
    torch.save({"state": model.state_dict(), "tag": tag, "layers": model.layers, "steps": step},
               os.path.join(ROOT, "models", f"beats_ftgru_{tag}_full.pt"))
    model.eval()
    clips = test_clips()
    probs, t1 = {}, time.time()
    for i, (k, p) in enumerate(clips):
        probs[k] = B.predict_clip(model, p, "cuda")
        if i % 1000 == 0:
            log(f"test {i}/{len(clips)} {time.time() - t1:.0f}s")
    pickle.dump(probs, open(os.path.join(ROOT, "cache", f"test_probs_{tag}_full.pkl"), "wb"))
    log(f"DONE {tag}: {step} steps, test probs for {len(probs)} clips")


if __name__ == "__main__":
    main()
