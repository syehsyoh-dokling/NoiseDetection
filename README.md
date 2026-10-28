# Datathon@IndoML 2026, Track 1 (Noise Event Detection): matched-filter library + 12-member SED ensemble

Team **syehsyoh**. Final leaderboard: **rank 5**, Codabench submission **994586**, combined score
**1.600895** (event F1 0.780264, segment Dice 0.820631; the metric's maximum is 2.0).

The test set has two kinds of clips. Synthetic clips are clean speech with noise samples pasted in from a finite
library, and the validation release ships both the noisy mix and the clean reference, so `noisy − clean` recovers
every pasted waveform exactly: a normalised cross-correlation matched filter over that library finds the pasted
sample and its exact position. Natural field recordings get a frame-level sound-event detector: a weighted mean of
12 members (9 frozen-encoder BiGRU heads on Whisper / AST / BEATs / Dasheng features and 3 fine-tuned encoders),
decoded with peak-relative boundaries. A clip with any confident library hit is treated as synthetic.

```
clip ──► matched filter over 1,198 library templates (NCC − per-template null level)
   │             │ hit  ──► template events + SED events that do not overlap them
   │             │ none ──► SED events (peak-relative decode, two category branches)
   └──► 12-member SED ensemble ──► 20 ms frame probabilities (8 channels)
```

## Repository layout

```
README.md
requirements.txt
run.py                     predict: library matching + SED ensemble + fusion -> predictions.jsonl in a zip
sed.py features.py         frozen-encoder features (20 ms grid), BiGRU head, training loop
template_match.py          library construction, FFT matched filter, null levels, event selection
eval_templates.py          leave-one-clip-out matcher evaluation on validation (writes val_template_cands.pkl)
postproc.py reldecode.py calib.py fusion.py   decoding of frame probabilities and fusion with template events
train_cv.py                5-fold district-grouped training of a frozen-encoder head (members wsA, wl*, aH, bH, dH)
ast_ft_gru.py beats_ft_gru.py                 fine-tuned members aFG and bFG (per fold)
dasheng_ft_gru.py whisper_ft_gru.py           fine-tuned variants tried and not used in the final system
hf_data.py hf_extract.py   Gold / Silver tiers of the Vaani noise-event dataset (Hugging Face, gated)
pseudo.py                  leak-free pseudo-labels on the natural-like test clips (member wlP)
build_ft_candidate.py build_nodata.py         build a submission from a fusion config (+ cached probabilities)
scorer.py                  local copy of the Track 1 metric
verify_retrain_oof.py      out-of-fold check of the retrained aFG/bFG against the originals
beats/                     BEATs model code (microsoft/unilm, MIT)
models/fusion_Q1_bFSadd4.json                 the submitted configuration (members, weights, decode parameters)
models/fusion_Q1_P2_og20.json                 the same without bFS (the last configuration with an OOF estimate)
models/sed_head_<tag>_f<k>.pt                 the 45 frozen-encoder heads of the submission (9 tags x 5 folds)
kaggle/                    the Kaggle scripts used for the retrain and for the reproduction (see below)
submissions/submission_994586_original.zip    the file that was scored as 994586
submissions/submission_reproduced.zip         rebuilt from this repository (see "Reproduction")
docs/technical_report.pdf  system description
```

## Environment and hardware

| | Training | Inference |
| --- | --- | --- |
| Python | 3.13.6 (laptop), 3.13 (Kaggle image) | 3.13.6 |
| PyTorch / CUDA | 2.14.0+cu126 / 12.6 (laptop); Kaggle default GPU image of October 2026 | 2.14.0+cu126 / 12.6 |
| GPU | NVIDIA RTX 4060 Laptop 8 GB (frozen heads, aFG, bFG folds 0-2); Kaggle 2x Tesla T4 16 GB (bFG folds 3-4, bFS; the retrain) | any CUDA GPU with >= 8 GB (Whisper large-v3 features) |
| CPU / RAM | Windows 11 laptop, 12 GB RAM | matched filter: several GB RAM for the longest clips (FFT size 2^19); 85 min for the 5,517 test clips |

Pretrained models (downloaded from Hugging Face / GitHub into `models/<name>`; none are redistributed here):

| Folder | Source | Used by |
| --- | --- | --- |
| `models/whisper-small` | openai/whisper-small | wsA |
| `models/whisper-large-v3` | openai/whisper-large-v3 | wlP, wlH, wlH2, wlH3, wlHS |
| `models/ast-audioset` | MIT/ast-finetuned-audioset-10-10-0.4593 | aH, aFG |
| `models/BEATs_iter3_plus_AS2M.pt` | microsoft/unilm BEATs (iter3+ AS2M) | bH, bFG, bFS |
| `models/dasheng-base` | mispeech/dasheng-base | dH |

## Inference

```bash
pip install -r requirements.txt
# data/validation/ = official validation release (validationMetadata.json, naturalNoisyAudio/, syntheticNoiseAudio/,
#                    syntheticCleanRefAudio/); data/test/audio/*.wav = the 5,517 test clips
python eval_templates.py                       # library (noisy - clean) + validation candidates for the null levels
python build_ft_candidate.py Q1_bFSadd4        # -> submission/submission_v20_Q1_bFSadd4.zip
```

`build_ft_candidate.py` runs the matched filter over the test clips, computes the frozen members' probabilities from
`models/sed_head_*.pt` (one encoder pass per backbone), averages the five fold models of each fine-tuned member
(`cache/test_probs_<tag>_f<k>.pkl`, written by the fine-tuning scripts), and fuses everything with the parameters in
the config. The fine-tuned checkpoints (aFG, bFG: 5 folds each, ~350 MB per fold; bFS: one full-data model) are
released as separate archives, see "Checkpoints" below.

## Training

Data: the released validation set (2,501 clips: 1,806 natural, 695 synthetic with clean references; 5,157 events,
7 categories) and the Gold tier of ARTPARK-IISc/Vaani-Noise-Event-Dataset (10,772 clips with verified timestamps).
Two members also use weaker labels: wlHS and bFS add the Silver tier (unverified timestamps; 14,529 clips for wlHS,
the first 12,000 of the Kaggle copy `ayanokoji2028/vaani-noise-hf` for bFS) at loss weight 0.4, and wlP is trained
on pseudo-labels (`pseudo.py`, set p1) that three validation-only Whisper heads (small, medium, large at weights
1/1/2) produced on the 3,353 natural-like **test** clips (transductive, leak-free per fold: the student of fold f
only sees pseudo-labels from teachers that never saw fold f). The p1 file and the medium/large teacher heads are
not part of this release; wlP is released as its five trained heads. No test labels were released or used. Folds
are GroupKFold by state/district because the test set comes from unseen districts and speakers.

| Member | Encoder (frozen unless noted) | Layers | Training data | Recipe |
| --- | --- | --- | --- | --- |
| wsA | whisper-small | 3,6,9,12 | validation | `train_cv.py --tag wsA --aug` |
| wlP | whisper-large-v3 | 8,16,24,32 | validation + pseudo-labels p1 | `--aug --pseudo p1 --pseudo-weight 0.5` |
| wlH, wlH2, wlH3 | whisper-large-v3 | 8,16,24,32 | validation + Gold | `--hf --aug --epochs 8 --bs 8 --max-frames 400`, seeds 0/1/2 |
| wlHS | whisper-large-v3 | 8,16,24,32 | validation + Gold + Silver (w 0.4) | as wlH + `--silver hf_silver --silver-weight 0.4` |
| aH / bH / dH | AST / BEATs / Dasheng | 4,8,12 | validation + Gold | as wlH, `--backbone models/<name>` |
| aFG | AST, all blocks fine-tuned | 4,8,12 | validation + Gold | `ast_ft_gru.py --fold k` (3 epochs, encoder lr 2e-5, head lr 1e-3) |
| bFG | BEATs, blocks fine-tuned | 4,8,12 | validation + Gold | `beats_ft_gru.py --fold k` (same recipe) |
| bFS | BEATs fine-tuned, no hold-out | 4,8,12 | all validation + Gold + Silver (w 0.4) | `kaggle/t1_beats_full.py` (2 epochs) |

Every head is a BiGRU (hidden 192) over per-frame layer-normed, frequency-averaged encoder tokens interpolated onto
a 20 ms grid and averaged over the selected layers; 8 output channels (any-noise + 7 categories), masked BCE with
channel weights, SpecAugment-style feature masking. Ensemble weights (`wsA:0.5, wlP:3, wlH:2, wlH2:6, wlH3:1,
wlHS:6, aH:8, aFG:8, bH:8, bFG:8, dH:4, bFS:4`) and decode parameters were tuned on out-of-fold predictions; the
configuration without bFS (`fusion_Q1_P2_og20.json`) scores 1.4787 out-of-fold on validation and 1.6002 on the
leaderboard (submission 991838); adding bFS at weight 4 was a leaderboard probe (+0.0007, no OOF estimate because
bFS trains on all of validation).

## Validation (out-of-fold, 5 district-grouped folds, library in leave-one-clip-out mode)

`kaggle/component_oof.py` reproduces these from `cache/oof_<tag>.pkl` and `cache/val_template_cands.pkl`
(combined score, with event F1 / segment Dice in brackets):

| System | All 2,501 clips | 1,806 natural | 695 synthetic |
| --- | --- | --- | --- |
| Q1 (final members without bFS, templates + SED) | 1.4787 (.6894/.7894) | 1.1570 (.4465/.7105) | 1.9868 (.9925/.9943) |
| Q1 without wlP | 1.4759 (.6872/.7887) | 1.1527 (.4431/.7096) | 1.9866 (.9923/.9943) |
| Q1 with retrained aFG/bFG | 1.4771 (.6886/.7885) | 1.1542 (.4450/.7092) | 1.9869 (.9925/.9944) |
| SED only | 1.2005 (.4500/.7505) | 1.1409 (.4331/.7077) | 1.3378 (.4761/.8618) |
| Templates only | 0.8871 (.5976/.2895) | 0.0691 (.0424/.0268) | 1.9542 (.9819/.9723) |

Leaderboard lineage (submission id, combined): 921278 1.4591 (first upload) -> 924220 1.5808 (v14) -> 929944
1.5868 (+aFG) -> 932046 1.5890 (+bH) -> 932698 1.5938 (+bFG, dH) -> 933670 1.5972 (boundary calibration) ->
936587 1.5999 -> 991824 1.6000 -> 991838 1.6002 (Q1) -> 994586 1.6009 (+bFS:4, final). Codabench lists 47 uploads
by the team (46 scored, 1 failed); the first one is no longer listed. Selection was leaderboard-guided.

Overlap of external training data with the competition data (`kaggle/overlap_check.py`, by Vaani file stem):
Gold vs test 0 clips (and 0 shared recording sessions), Gold vs validation 0 clips (13 Gold clips share a
recording session with a validation clip); both Silver extractions exclude validation and test file names.

## Reproduction

After the final upload the workspace lost the backbone weights, the fine-tuned checkpoints and all cached test
probabilities; only the 45 frozen heads, the configurations, the template library and the original zip survived.
For this release we re-ran the lost parts with the identical recipes (`kaggle/retrain_ast_beats.py`,
`kaggle/t1_beats_full.py` on Kaggle 2x T4; `kaggle/frozen_test_probs.py` for the frozen members;
`kaggle/test_matcher.py` for the matched filter) and rebuilt the submission with `kaggle/install_retrain.py`.

- Retrained aFG/bFG vs the originals, out-of-fold on validation (`verify_retrain_oof.py`): Q1 configuration 1.47873
  (original) vs 1.47710 (retrained), i.e. -0.0016.
- Rebuilt submission (`submissions/submission_reproduced.zip`) vs the scored file 994586: the same 2,164 clips
  flagged as synthetic-like, 13,442 vs 13,451 events, 2,306 of 5,517 clips (41.8 %) identical, and an agreement of
  **1.9858** under the competition metric (event F1 0.9943 / segment Dice 0.9915) with the scored file as reference
  (`kaggle/install_retrain.py`).

The fine-tuned members are therefore not bit-identical to the scored ones (GPU nondeterminism and a different GPU);
the frozen members and the matched filter are deterministic given the released heads and library.

## Checkpoints

`models/sed_head_*.pt` (45 files, 314 MB) are in this repository. The retrained fine-tuned checkpoints
(`ast_ftgru_aFG_f0-4.pt`, `beats_ftgru_bFG_f0-4.pt`, `beats_ftgru_bFS_full.pt`; 11 files, 3.96 GB; SHA-256 in
`models/CHECKSUMS_finetuned.txt`) are published as assets of the GitHub release `v1.0` of this repository; place
them in `models/` before running `build_ft_candidate.py` (the fold scripts cache their test probabilities in
`cache/test_probs_<tag>_f<k>.pkl`; `kaggle/install_retrain.py` shows the exact assembly used for the release).

## License and data

Our code is released under the MIT License. `beats/` is vendored from microsoft/unilm (MIT). Pretrained encoders
keep their own licences (Whisper: Apache-2.0; AST: BSD-3-Clause; BEATs: MIT; Dasheng: Apache-2.0). Vaani audio and the
competition data are not redistributed.
