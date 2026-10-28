"""Overlap check between the 5,517 test clips and every external training clip (Gold, Silver), by Vaani file stem
and by (state, district) group. Vaani segment names are unique recording ids, so a stem match is a clip match."""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(R, "data", "test", "audio", "*.wav"))}
val = {m["segmentFileName"].rsplit(".", 1)[0]
       for m in json.load(open(os.path.join(R, "data", "validation", "validationMetadata.json"), encoding="utf-8"))}
print(f"test {len(test)} | validation {len(val)} | test&val {len(test & val)}")
for name in ("hf_gold", "hf_silver"):
    p = os.path.join(R, "data", name, "metadata.json")
    if not os.path.exists(p):
        print(name, "metadata not on disk")
        continue
    meta = json.load(open(p, encoding="utf-8"))
    stems = {m["segmentFileName"].rsplit(".", 1)[0] for m in meta}
    print(f"{name}: {len(stems)} clips | & test {len(stems & test)} | & validation {len(stems & val)}")
    # recording-session prefix (everything before the utterance index): same speaker/session as a test clip?
    pre = lambda s: s.rsplit("_", 4)[0]
    tp = {pre(s) for s in test}
    print(f"  same recording prefix as a test clip: {sum(pre(s) in tp for s in stems)} clips")
