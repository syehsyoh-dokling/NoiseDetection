# Datathon@IndoML 2026, Track 1 (Noise Event Detection): matched-filter library + 12-member SED ensemble

Team **syehsyoh**. Final leaderboard: **rank 5**, Codabench submission **994586**, combined score
**1.600895** (event F1 0.780264, segment Dice 0.820631; the metric's maximum is 2.0).
System description: [`docs/technical_report.pdf`](docs/technical_report.pdf) (3 pages, IEEE two-column).

The synthetic test clips appear to be clean speech with noise samples pasted from a finite library. The validation
release ships the clean references, so `noisy − clean` recovers the pasted waveforms (up to a residual of at most
4.4e-3 at full scale), and a normalised cross-correlation matched filter over that library finds library waveforms
in the test clips. Clips without a library hit get a frame-level sound-event detector: a weighted mean of 12 members
(9 frozen-encoder BiGRU heads on Whisper / AST / BEATs / Dasheng features and 3 fine-tuned encoders), decoded with
peak-relative boundaries.

```
clip ──► matched filter over 1,198 library templates (NCC − per-template null level)
   │             │ hit  ──► template events + SED events that do not overlap them
   │             │ none ──► SED events (peak-relative decode, two category branches)
   └──► 12-member SED ensemble ──► 20 ms frame probabilities (8 channels)
```

## Repository layout

```
README.md  LICENSE  requirements_train.txt  requirements_inference.txt
inference/infer.py         end-to-end inference: test WAVs -> predictions.jsonl in a zip (Phase 2 format)
scripts/run_inference.sh   one-command inference (builds the template library first if needed)
run.py                     library matching + SED ensemble + fusion -> predictions.jsonl in a zip
sed.py features.py         frozen-encoder features (20 ms grid), BiGRU head, training loop, augmentation
template_match.py          library construction, FFT matched filter, null levels, event selection
eval_templates.py          builds the library and leave-one-clip-out validation candidates (val_template_cands.pkl)
postproc.py reldecode.py fusion.py            decoding of frame probabilities and fusion with template events
train_cv.py                5-fold district-grouped training of a frozen-encoder head (wsA, wl*, aH, bH, dH)
ast_ft_gru.py beats_ft_gru.py                 fine-tuned members aFG and bFG (per fold)
kaggle/t1_beats_full.py    fine-tuned member bFS (full data + Silver)
pseudo.py                  leak-free pseudo-labels on the natural-like test clips (member wlP)
hf_data.py hf_extract.py   Gold / Silver tiers of the Vaani noise-event dataset (Hugging Face, gated)
tune_rel.py tune_weights.py                   decode / weight tuning on out-of-fold predictions
build_ft_candidate.py build_nodata.py         build a submission from a fusion config (+ cached probabilities)
scorer.py                  local copy of the Track 1 metric
calib.py dasheng_ft_gru.py whisper_ft_gru.py  variants tried and not used in the final system
beats/                     BEATs model code (microsoft/unilm, MIT)
models/fusion_Q1_bFSadd4.json                 the submitted configuration (members, weights, decode parameters)
models/fusion_*.json                          every configuration built during development (incl. rejected ones)
models/sed_head_<tag>_f<k>.pt                 the 45 frozen-encoder heads of the submission (9 tags x 5 folds)
models/CHECKSUMS_finetuned.txt                SHA-256 of the 11 fine-tuned checkpoints (release v1.0 assets)
kaggle/                    Kaggle scripts of the retrain and the reproduction, and the analysis scripts of the report
submissions/submission_994586_original.zip    the file that was scored as 994586
submissions/submission_reproduced.zip         rebuilt from the released artefacts (see "Reproduction")
docs/technical_report.pdf  system description
```

## Environment and hardware

| | Training | Inference |
| --- | --- | --- |
| Python | 3.13.6 (laptop), 3.13 (Kaggle default GPU image, Oct 2026) | 3.13.6 / 3.13 |
| PyTorch / CUDA | 2.14.0+cu126 / CUDA 12.6 at the October rebuild (versions at the original September training were not recorded) | same |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB (frozen heads, all aFG folds, bFG folds 0-2); Kaggle GPUs (bFG folds 3-4, bFS; type not recorded); retrain: Kaggle Tesla T4, 16 GB, one GPU per job | CUDA GPU with >= 8 GB (Whisper large-v3 features); reproduction ran on Tesla T4 16 GB and RTX 4060 Laptop 8 GB |
| CPU / RAM | Windows 11 laptop, 12 GB RAM; Kaggle standard GPU session | matched filter: several GB RAM for the longest clips (FFT size up to 2^19) |
| System packages | none beyond a CUDA driver (`soundfile` bundles libsndfile); `bash` for `scripts/run_inference.sh` | same |

Measured compute (Kaggle T4 retrain, log timestamps): one aFG fold 61-63 min, one bFG fold 55-60 min, bFS 90 min
including its test pass (the original bFS run took 75 min); about 11 T4-hours for the fine-tuned members. Inference:
one fine-tuned encoder pass over the 5,517 test clips (10.81 h of audio) 4.4-5.5 min, the five frozen-backbone
passes under 1.4 h, the matched filter 84.6 min on the laptop GPU. The system is offline and non-causal.

Pretrained models (all public; none redistributed here). Revisions are the Hugging Face commits used for the
released reproduction on 28 Oct 2026; the revisions downloaded for the original September training were not
recorded (`mispeech/dasheng-base` was modified on 25 Oct 2026, after our training).

| Folder in `models/` | Source | Revision | Licence | Role |
| --- | --- | --- | --- | --- |
| `whisper-small` | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | 973afd24965f | Apache-2.0 | encoder of wsA; pseudo-label teacher |
| `whisper-large-v3` | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 06f233fe06e7 | Apache-2.0 | encoder of wlP, wlH, wlH2, wlH3, wlHS; teacher |
| `ast-audioset` | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | f826b80d2822 | BSD-3-Clause | encoder of aH; initialisation of aFG |
| `BEATs_iter3_plus_AS2M.pt` | [microsoft/unilm, BEATs](https://github.com/microsoft/unilm/tree/master/beats) (checkpoint "BEATs_iter3+ (AS2M)") | file from the BEATs page | MIT | encoder of bH; initialisation of bFG, bFS |
| `dasheng-base` | [mispeech/dasheng-base](https://huggingface.co/mispeech/dasheng-base) | e0b0a27818689 | Apache-2.0 | encoder of dH |

`inference/infer.py` downloads the Hugging Face models at these revisions on first use. The BEATs checkpoint must be
downloaded from the BEATs page and placed in `models/`. The wlP pseudo-label teachers also used a whisper-medium
encoder (`models/whisper-medium`, head wmA; validation-only, not released, revision not recorded).

## Inference

Inputs: the official validation release in `data/validation/` (`validationMetadata.json`, `naturalNoisyAudio/`,
`syntheticNoiseAudio/`, `syntheticCleanRefAudio/`; needed once to build the template library and the null levels)
and a folder of test WAVs (16 kHz mono, searched recursively). Checkpoints: the 45 heads in `models/` (in this
repository) and the 11 fine-tuned checkpoints from release `v1.0` placed in `models/` (verify with
`models/CHECKSUMS_finetuned.txt`).

```bash
pip install -r requirements_inference.txt
bash scripts/run_inference.sh /path/to/test_wavs work/submission_track1.zip
```

Output: a zip with a single `predictions.jsonl` at its root, one line per clip,
`{"clip_id": "<wav name without .wav>", "events": [{"onset": s, "offset": s}, ...]}` (the Phase 2 format).

`inference/infer.py` follows the code path of the released reproduction: the frozen-member step is the Kaggle script
`kaggle/frozen_test_probs.py` with local paths (backbones are resolved by folder name, so the heads load on any
machine), bFS is run from its checkpoint, aFG/bFG average their five fold models (`build_ft_candidate.fold_probs`),
and `run.predict` applies the matched filter and the fusion of `models/fusion_Q1_bFSadd4.json`. In this release the
script was not executed end to end on one machine; the same steps were run separately (Kaggle T4 for the encoders,
laptop for the matcher and fusion) to produce `submissions/submission_reproduced.zip`.

## Training

Data: the validation release (2,501 clips: 1,806 natural, 695 synthetic with clean references; 5,157 events,
7 categories) and the Gold tier of ARTPARK-IISc/Vaani-Noise-Event-Dataset (10,772 clips with verified timestamps;
gated, CC BY 4.0). wlHS and bFS add the Silver tier (unverified timestamps, sample weight 0.4; 14,529 clips for wlHS,
the first 12,000 of the Kaggle copy `ayanokoji2028/vaani-noise-hf` for bFS). wlP is trained on pseudo-labels
(`pseudo.py`, set p1, sample weight 0.5) that three validation-only Whisper heads (small, medium, large at weights
1/1/2) produced on the 3,353 test clips without a library hit (transductive; leak-free per fold). The p1 file and the
medium/large teacher heads are not part of this release, so wlP can be run but not retrained. No test labels were
released or used. Folds are GroupKFold by state/district.

Augmentation pipeline (no waveform augmentation or noise mixing): frozen members use cached float16 encoder
features, a random crop of 400 or 600 frames, then 2 time masks of up to 25 frames set to zero and 10 % of feature
channels dropped (`sed.augment`); fine-tuned members use a random 10.24 s crop, then 2 time masks (<= 40 frames) and
2 frequency masks (<= 16 bins) on the encoder's log-mel filterbank (`spec_augment` in `ast_ft_gru.py` /
`beats_ft_gru.py`). Loss: frame-masked binary cross-entropy over 8 channels with channel weights 1.0 (any noise) and
0.5 (each category), weighted per clip by the sample weight. Head: LayerNorm -> Linear(dim, 384) -> GELU -> 2-layer
BiGRU (hidden 192, dropout 0.3) -> Linear(8). Optimiser AdamW (weight decay 1e-2), one-cycle schedule (peak 1e-3;
encoder 2e-5 for fine-tuned members), last epoch kept.

| Member | Encoder (frozen unless noted) | Layers | Training data | Recipe |
| --- | --- | --- | --- | --- |
| wsA | whisper-small | 3,6,9,12 | validation | `train_cv.py --tag wsA --aug` (25 epochs, batch 16, 600-frame crops) |
| wlP | whisper-large-v3 | 8,16,24,32 | validation + pseudo-labels p1 | `--aug --pseudo p1 --pseudo-weight 0.5 --compact` |
| wlH, wlH2, wlH3 | whisper-large-v3 | 8,16,24,32 | validation + Gold | `--hf --aug --epochs 8 --bs 8 --max-frames 400`, seeds 0/1/2 |
| wlHS | whisper-large-v3 | 8,16,24,32 | validation + Gold + Silver | `--hf --silver hf_silver --silver-weight 0.4 --aug --epochs 6 --bs 8 --max-frames 400` |
| aH / bH / dH | AST / BEATs / Dasheng | 4,8,12 | validation + Gold | `--hf --aug --epochs 8 --bs 8 --max-frames 400 --backbone models/<name>` |
| aFG | AST, all 12 blocks fine-tuned | 4,8,12 | validation + Gold | `ast_ft_gru.py --fold k --epochs 3 --seed 0` (encoder lr 2e-5, head lr 1e-3, batch 8, fp16) |
| bFG | BEATs fine-tuned, patch embedding frozen | 4,8,12 | validation + Gold | `beats_ft_gru.py --fold k --epochs 3 --seed 0` |
| bFS | BEATs fine-tuned, no hold-out | 4,8,12 | all validation + Gold + Silver | `kaggle/t1_beats_full.py` (TAG=bFS, EPOCHS=2, SILVER_W=0.4; 6,320 steps, seed 7) |

Ensemble weights (`wsA:0.5, wlP:3, wlH:2, wlH2:6, wlH3:1, wlHS:6, aH:8, aFG:8, bH:8, bFG:8, dH:4, bFS:4`): the eight
v14 weights are a recipe recovered from a log (the search that produced them was not preserved); each later member
entered at one value from a ladder (aFG 2-12 `fusion_M*`, bH 2-16 `fusion_B*`, bFG add4/add8/swap8 `fusion_E_*`
with the `auto_bfg` gate of a pooled OOF gain and >= 4/5 folds up, dH 4-12 `fusion_F*`, bFS 4-16
`fusion_Q1_bFS*`), chosen by OOF checks and/or leaderboard uploads. Not all choices are recorded and the weights were
never re-tuned jointly, so the final mix is not an out-of-fold optimum (three members have negative leave-one-out
support, below). The decode parameters were tuned on out-of-fold predictions (`tune_rel.py`) and confirmed on the
leaderboard.

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

Fold-to-fold spread (`kaggle/fold_category_oof.py`): Q1 1.4912 / 1.4412 / 1.5895 / 1.3787 / 1.4806 (mean 1.476,
sd 0.077); without wlP mean 1.474, sd 0.075 (paired difference positive on 4 of 5 folds, mean +0.0026, t = 1.7,
not significant); SED only mean 1.201, sd 0.031. Each member was trained with one seed; seed variance was not
measured.

Leave-one-member-out (`kaggle/member_ablation_oof.py`, decode fixed; change of pooled OOF when the member is
removed, folds where the member helps): bFG -0.0097 (5/5), dH -0.0069 (5/5), bH -0.0061 (5/5), aFG -0.0035 (3/5),
wlH -0.0033 (5/5), wlP -0.0028 (4/5), wlH2 -0.0027 (4/5), wlHS -0.0027 (3/5), wlH3 +0.0001, wsA +0.0010,
aH +0.0014 (the last three carry no OOF support). The 73 natural validation clips with a library hit: 64 of 77
template events (83 %) match a reference event, so library waveforms also occur in natural recordings (test hits are
therefore not all synthetic clips, and such matches raise the null levels); on these clips the fused output scores
1.3047 vs 0.9519 for the SED decode alone.

Recall of the 2,944 natural reference events per category (Q1 OOF, metric tolerance; predictions carry no
category): human non-speech 0.622 (n=1,254, median duration 0.32 s), vehicle/traffic 0.282 (655, 0.70 s), animal
0.269 (594, 0.79 s), baby/child 0.215 (265, 0.70 s), phone/signal/alarm 0.300 (110, 0.60 s), singing/music 0.225
(40, 2.78 s), appliance/machine 0.231 (26, 6.23 s); all 0.418.

Leaderboard lineage (submission id, combined): 921278 1.4591 (first upload) -> 924220 1.5808 (v14) -> 929944
1.5868 (+aFG) -> 932046 1.5890 (+bH) -> 932698 1.5938 (+bFG, dH) -> 933670 1.5972 (decode re-tune) -> 936587
1.5999 -> 991824 1.6000 -> 991838 1.6002 (Q1) -> 994586 1.6009 (+bFS:4, final). Codabench lists 47 uploads by the
team (46 scored, 1 failed); the first one is no longer listed. Selection was leaderboard-guided over at least 48
uploads; leaderboard noise was not quantified.

Overlap of external training data with the competition data (`kaggle/overlap_check.py`, by Vaani file stem):
Gold vs test 0 clips and 0 shared recording sessions; Gold vs validation 0 clips (13 Gold clips share a recording
session with a validation clip; effect not measured); both Silver extractions exclude validation and test file
names (not checked by session).

## Reproduction

After the final upload the workspace lost the backbone weights, the fine-tuned checkpoints, all cached test
probabilities and the test template candidates; the 45 frozen heads, the configurations, the template library and
the original zip survived. For this release we re-ran the lost parts with the same scripts and seeds
(`kaggle/retrain_ast_beats.py`, `kaggle/t1_beats_full.py` on Kaggle T4; `kaggle/frozen_test_probs.py` for the frozen
members; `kaggle/test_matcher.py` for the matched filter) and rebuilt the submission with
`kaggle/install_retrain.py`.

- Retrained aFG/bFG vs the originals, out-of-fold on validation (`verify_retrain_oof.py`): 1.47873 (original) vs
  1.47710 (retrained), i.e. -0.0016.
- Rebuilt submission (`submissions/submission_reproduced.zip`) vs the scored file 994586, with the scored file as
  reference: the identical set of 2,164 clips flagged as synthetic-like; agreement 1.9994 on those clips
  (F1 0.9999 / Dice 0.9996) and 1.9727 on the other 3,353 (F1 0.9865 / Dice 0.9862), 1.9858 overall; 13,442 vs
  13,451 events; byte-identical on 2,008 synthetic-like and 298 natural-like clips.

The rebuilt probabilities are not bit-identical (three members retrained; different GPUs, software and Dasheng
revision; the causes were not separated).

## Checkpoints

`models/sed_head_*.pt` (45 files, 314 MB) are in this repository. The retrained fine-tuned checkpoints
(`ast_ftgru_aFG_f0-4.pt`, `beats_ftgru_bFG_f0-4.pt`, `beats_ftgru_bFS_full.pt`; 11 files, 3.96 GB; SHA-256 in
`models/CHECKSUMS_finetuned.txt`) are the assets of the GitHub release `v1.0` of this repository (each is larger
than GitHub's 100 MB file limit); place them in `models/`.

Cached model outputs (no audio) are also release assets; unzip them at the repository root (they extract to
`cache/`):

| Asset | Contents | SHA-256 |
| --- | --- | --- |
| `cache_oof_validation.zip` (269 MB) | `oof_<tag>.pkl` out-of-fold validation probabilities of every member (incl. the retrained `aFGRT`/`bFGRT` and the rejected cH/eH/dFG) and `val_template_cands.pkl`; enough to rerun `kaggle/component_oof.py`, `fold_category_oof.py`, `member_ablation_oof.py` | `8d834e617aec4557637b88bdf017c157de95b0836709953ccd58dd3e2b9be98e` |
| `cache_test_outputs.zip` (896 MB) | the rebuild's test probabilities of all 12 members (`test_sed_probs_<tag>.pkl`, `test_probs_<tag>_f<k>.pkl`), `test_template_cands.pkl`, `t1_syn_keys.json`, `test_durs.json`; with these, `val_template_cands.pkl` from the other asset and the template library rebuilt from the validation release, `python build_nodata.py Q1_bFSadd4` rebuilds `submissions/submission_reproduced.zip` without a GPU | `67d7f4978f182413611f3b09191de655a37e9dafc7d564f56e4bf747339665cc` |

`cache/template_library.pkl` is not released: it holds waveforms cut from the validation audio, which may not be
redistributed; `python eval_templates.py` rebuilds it from the official validation release.

## License and data

Our code is released under the MIT License (`LICENSE`). `beats/` is vendored from microsoft/unilm (MIT). The
fine-tuned checkpoints are derivatives of AST (BSD-3-Clause) and BEATs (MIT) and keep those notices. Pretrained
encoders are not redistributed. Vaani audio and the competition data are not redistributed.
