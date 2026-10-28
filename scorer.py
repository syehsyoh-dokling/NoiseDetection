"""Local copy of the Track 1 metric, transcribed from the competition's Evaluation page.

match_events, event_based_f1, events_to_frames and segment_dice are kept line-for-line
identical to the published reference code so local numbers match the leaderboard's.
Everything else here (loading, validation, CLI) is ours.
"""
import json
import sys


def match_events(ref_events, pred_events, tolerance_frac=0.20):
    matched_ref, matched_pred = set(), set()
    candidates = []
    for ri, (r_on, r_off) in enumerate(ref_events):
        tol = max(tolerance_frac * (r_off - r_on), 0.05)   # at least 50 ms
        for pi, (p_on, p_off) in enumerate(pred_events):
            if abs(p_on - r_on) <= tol and abs(p_off - r_off) <= tol:
                candidates.append((abs(p_on - r_on) + abs(p_off - r_off), ri, pi))
    for _, ri, pi in sorted(candidates):        # closest first
        if ri not in matched_ref and pi not in matched_pred:
            matched_ref.add(ri); matched_pred.add(pi)
    tp = len(matched_ref)
    return tp, len(pred_events) - tp, len(ref_events) - tp   # tp, fp, fn


def event_based_f1(ref_data, pred_data):
    TP = FP = FN = 0
    for clip_id, ref_events in ref_data.items():
        tp, fp, fn = match_events(ref_events, pred_data.get(clip_id, []))
        TP += tp; FP += fp; FN += fn
    for clip_id in pred_data:                   # extra clips -> false positives
        if clip_id not in ref_data:
            FP += len(pred_data[clip_id])
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    return (2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0)


def events_to_frames(events, max_time, frame_len=0.01):
    n_frames = int(max_time / frame_len) + 1
    mask = [0] * n_frames
    for on, off in events:
        for i in range(int(on / frame_len), min(int(off / frame_len) + 1, n_frames)):
            mask[i] = 1
    return mask


def segment_dice(ref_data, pred_data):
    scores = []
    for clip_id, ref_events in ref_data.items():
        pred_events = pred_data.get(clip_id, [])
        all_events = ref_events + pred_events
        if not all_events:
            scores.append(1.0); continue
        max_time = max(off for _, off in all_events) + 0.5   # pad 0.5 s
        ref_mask  = events_to_frames(ref_events,  max_time)
        pred_mask = events_to_frames(pred_events, max_time)
        inter = sum(r & p for r, p in zip(ref_mask, pred_mask))
        total = sum(ref_mask) + sum(pred_mask)
        scores.append(1.0 if total == 0 else 2.0 * inter / total)
    return sum(scores) / len(scores) if scores else 0.0


def score(ref_data, pred_data):
    f1 = event_based_f1(ref_data, pred_data)
    dice = segment_dice(ref_data, pred_data)
    return {"event_f1": f1, "segment_dice": dice, "combined_score": f1 + dice}


def load_jsonl(path):
    """clip_id -> list of (onset, offset); raises on anything the leaderboard might reject."""
    data = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec["clip_id"]
            if cid in data:
                raise ValueError(f"line {n}: duplicate clip_id {cid!r}")
            evs = []
            for ev in rec["events"]:
                on, off = float(ev["onset"]), float(ev["offset"])
                if not (0.0 <= on < off):
                    raise ValueError(f"line {n}: bad event {ev!r} in {cid!r}")
                evs.append((on, off))
            data[cid] = evs
    return data


if __name__ == "__main__":
    ref, pred = load_jsonl(sys.argv[1]), load_jsonl(sys.argv[2])
    missing = set(ref) - set(pred)
    if missing:
        print(f"WARNING: {len(missing)} reference clips missing from predictions")
    print(json.dumps(score(ref, pred), indent=2))
