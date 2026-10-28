"""Frozen Whisper encoder as a frame-level (20 ms) feature extractor.

Whisper representations are known to keep background-sound identity (Whisper-AT,
Gong et al. 2023) while being robust to it, which is what a noise-event detector
needs; speech SSL models such as WavLM are trained to be noise-invariant instead.

HF's WhisperEncoder.forward insists on 30 s inputs. Clips here are ~6.5 s, so we run
the same layers on the true length with the positional table truncated.
"""
import os

import numpy as np
import torch
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder

SR = 16000
FPS = 50  # encoder frame rate: 10 ms mel hop, conv stride 2


def load_encoder(model_dir):
    """HF WhisperEncoder from model_dir: model.safetensors if present, otherwise the
    official OpenAI checkpoint (<size>.pt, SHA-256 checked by the download step) with its
    encoder keys renamed to the HF layout."""
    if os.path.exists(os.path.join(model_dir, "model.safetensors")):
        return WhisperModel.from_pretrained(model_dir, torch_dtype=torch.float16).encoder
    # models/whisper-large-v3 -> models/large-v3.pt
    pt = os.path.join(os.path.dirname(model_dir), os.path.basename(os.path.normpath(model_dir)).split("-", 1)[-1] + ".pt")
    # mmap: only the encoder tensors are ever touched, so the decoder half of the
    # checkpoint (and most of its RAM) is never paged in
    sd = torch.load(pt, map_location="cpu", weights_only=True, mmap=True)["model_state_dict"]
    ren = {"attn.query": "self_attn.q_proj", "attn.key": "self_attn.k_proj", "attn.value": "self_attn.v_proj",
           "attn.out": "self_attn.out_proj", "attn_ln": "self_attn_layer_norm", "mlp.0": "fc1", "mlp.2": "fc2",
           "mlp_ln": "final_layer_norm"}
    enc_sd = {}
    for k, v in sd.items():
        if not k.startswith("encoder."):
            continue
        k = k[len("encoder."):]
        if k == "positional_embedding":
            k = "embed_positions.weight"
        elif k.startswith("ln_post."):
            k = "layer_norm." + k.split(".", 1)[1]
        elif k.startswith("blocks."):
            k = "layers." + k[len("blocks."):]
            for a, b in ren.items():
                k = k.replace("." + a + ".", "." + b + ".")
        enc_sd[k] = v
    enc = WhisperEncoder(WhisperConfig.from_pretrained(model_dir))
    missing, unexpected = enc.load_state_dict(enc_sd, strict=False)
    # k_proj has no bias in Whisper; nothing else may be missing
    assert all(m.endswith("k_proj.bias") for m in missing) and not unexpected, (missing, unexpected)
    return enc.half()


def make_extractor(name, layers):
    """Backbone by directory name: models/ast-* -> ASTFrames, models/clap-* -> CLAPFrames,
    anything else -> WhisperFrames."""
    base = os.path.basename(os.path.normpath(name)).lower()
    if base.startswith("ast"):
        return ASTFrames(name, layers)
    if base.startswith("beats"):
        return BEATsFrames(name, layers)
    if base.startswith("dasheng"):
        return DashengFrames(name, layers, device=os.environ.get("INDOML_DEVICE", "cuda"))
    if base.startswith("ced"):
        return CEDFrames(name, layers, device=os.environ.get("INDOML_DEVICE", "cuda"))
    if base.startswith("clap"):
        return CLAPFrames(name, layers)
    return WhisperFrames(name, layers)


class CLAPFrames:
    """LAION CLAP's HTSAT audio tower (Swin transformer; AudioSet and general audio) as a
    third backbone.

    AST took the largest ensemble weight by making different mistakes from Whisper, not
    better ones (diag_weakness.py: AST alone is weaker per category, the blend beats
    both), so a second sound-event model with a different architecture is the direct way
    to add more of that kind of diversity.

    The input layout is not a plain spectrogram image. HTSAT resizes a 10 s log-mel to
    1024 x 64, cuts the time axis into freq_ratio (=4) consecutive blocks of 256 frames,
    and STACKS the blocks vertically: rows are [block, mel bin], columns are time within
    a block (2.5 s). Patching by 4 gives 64 columns per block, so the time tokens per 10 s
    window are 4 x 64 = 256 at stage 0 (~39 ms), 128 at stage 1 (~78 ms), 64 at stage 2
    (~156 ms) and 32 at stage 3 (~312 ms). Reading columns as absolute time would put
    every event in the wrong place, which clap_smoke.py checks with a synthetic burst.

    `layers` selects pre-downsampling stage outputs: 1 = stage 0 ... 4 = stage 3. Each is
    averaged over mel bins, unstacked back into time order, layer-normalised per frame
    (stage scales differ), interpolated onto the 20 ms grid, and concatenated. Stage widths
    differ, so the result is one pseudo-layer and sed.compact()'s layer mean is a no-op.
    """

    WIN_S = 10.0

    def __init__(self, name, layers=(2, 3, 4), device="cuda"):
        from transformers import ClapFeatureExtractor, ClapModel
        self.fe = ClapFeatureExtractor.from_pretrained(name)
        model = ClapModel.from_pretrained(name, dtype=torch.float32)
        # only the audio tower is used; the text tower never goes to the GPU
        self.encoder = model.audio_model.audio_encoder.to(device).eval()
        del model
        self.sr = self.fe.sampling_rate                       # 48 kHz
        self.win = int(self.WIN_S * SR)                       # window length on the 16 kHz clip
        self.blocks = self.encoder.freq_ratio                 # 4
        self.layers = tuple(layers)
        self.device = device
        base = self.encoder.config.patch_embeds_hidden_size
        self.dims = [int(base * 2 ** (l - 1)) for l in self.layers]
        self.dim = int(sum(self.dims))

    def _tokens(self, h):
        """(C, rows, cols) stage output -> (time, C): mean over mel bins, blocks unstacked."""
        c, rows, cols = h.shape
        h = h.reshape(c, self.blocks, rows // self.blocks, cols).mean(2)   # (C, blocks, cols)
        return h.permute(1, 2, 0).reshape(self.blocks * cols, c)          # block 0 cols..., block 1 ...

    @torch.no_grad()
    def __call__(self, wave):
        from scipy.signal import resample_poly
        n_frames = int(np.ceil(len(wave) / (SR / FPS)))
        centres = (np.arange(n_frames) + 0.5) / FPS
        out = np.zeros((1, n_frames, self.dim), np.float32)
        n_chunks = max(1, int(np.ceil(len(wave) / self.win)))
        # same ownership rule as ASTFrames: each output frame is read from one window
        owner = np.minimum((centres * SR // self.win).astype(int), n_chunks - 1)
        for ci in range(n_chunks):
            a = ci * self.win
            if ci == n_chunks - 1 and ci > 0:
                a = max(0, len(wave) - self.win)
            chunk = resample_poly(wave[a:a + self.win], self.sr // SR, 1).astype(np.float32)
            # zero-pad to 10 s ("repeatpad" would repeat the audio and invent events)
            x = self.fe(chunk, sampling_rate=self.sr, return_tensors="pt", padding="pad",
                        truncation="rand_trunc").input_features.to(self.device)
            hs = self.encoder(x, output_hidden_states=True,
                              output_hidden_states_before_downsampling=True).hidden_states
            sel = owner == ci
            q = centres[sel] - a / SR
            col = 0
            for l, d in zip(self.layers, self.dims):
                tok = self._tokens(hs[l][0].float())
                tok = (tok - tok.mean(-1, keepdim=True)) / (tok.std(-1, keepdim=True) + 1e-5)
                tok = tok.cpu().numpy()
                n_tok = tok.shape[0]
                t_centres = (np.arange(n_tok) + 0.5) * self.WIN_S / n_tok
                idx = np.clip(np.searchsorted(t_centres, q) - 1, 0, n_tok - 2)
                wgt = np.clip((q - t_centres[idx]) / (t_centres[idx + 1] - t_centres[idx]), 0, 1)[:, None]
                out[0, sel, col:col + d] = (1 - wgt) * tok[idx] + wgt * tok[idx + 1]
                col += d
        return torch.from_numpy(out).half()


class ASTFrames:
    """Audio Spectrogram Transformer fine-tuned on AudioSet, as a second opinion.

    Whisper is a speech model; AST was trained to name sound events (vehicle horn,
    bird, dog, baby cry...), exactly the categories whose recall is lowest here. Its
    patch grid is 12 frequency x 101 time positions per 10.24 s window (100 ms time
    stride), so tokens are averaged over frequency and linearly interpolated onto the
    same 20 ms grid the Whisper features use; only the ensemble sees both.
    """

    def __init__(self, name, layers=(4, 8, 12), device="cuda"):
        from transformers import ASTFeatureExtractor, ASTModel
        self.fe = ASTFeatureExtractor.from_pretrained(name)
        self.model = ASTModel.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
        c = self.model.config
        self.f_dim = (c.num_mel_bins - c.patch_size) // c.frequency_stride + 1
        self.t_dim = (c.max_length - c.patch_size) // c.time_stride + 1
        self.win = c.max_length * 160                      # samples per 10.24 s window
        self.t_centres = (np.arange(self.t_dim) * c.time_stride + c.patch_size / 2) * 0.01
        self.layers = tuple(layers)
        self.device = device
        self.dim = c.hidden_size

    @torch.no_grad()
    def __call__(self, wave):
        n_frames = int(np.ceil(len(wave) / (SR / FPS)))
        centres = (np.arange(n_frames) + 0.5) / FPS
        out = np.zeros((len(self.layers), n_frames, self.dim), np.float32)
        n_chunks = max(1, int(np.ceil(len(wave) / self.win)))
        # every output frame belongs to exactly one window (the last frame's centre can
        # sit just past the final sample, so clip into the last window)
        owner = np.minimum((centres * SR // self.win).astype(int), n_chunks - 1)
        for ci in range(n_chunks):
            a = ci * self.win
            if ci == n_chunks - 1 and ci > 0:
                # a tail of a few samples breaks the filterbank; end the last window at
                # the clip end instead (it overlaps the previous one, same frames owned)
                a = max(0, len(wave) - self.win)
            chunk = wave[a:a + self.win]
            x = self.fe(chunk, sampling_rate=SR, return_tensors="pt").input_values.to(self.device, torch.float16)
            hs = self.model(x, output_hidden_states=True).hidden_states
            t0 = a / SR
            sel = owner == ci
            for li, l in enumerate(self.layers):
                tok = hs[l][0, -self.f_dim * self.t_dim:].float().view(self.f_dim, self.t_dim, self.dim).mean(0)
                tok = tok.cpu().numpy()
                q = centres[sel] - t0
                idx = np.clip(np.searchsorted(self.t_centres, q) - 1, 0, self.t_dim - 2)
                w = np.clip((q - self.t_centres[idx]) / (self.t_centres[idx + 1] - self.t_centres[idx]), 0, 1)[:, None]
                out[li, sel] = (1 - w) * tok[idx] + w * tok[idx + 1]
        return torch.from_numpy(out).half()


class BEATsFrames:
    """BEATs (iter3+, AudioSet-2M self-supervised; microsoft/unilm, MIT) as a third architecture.

    aFG showed that a differently built model moves the leaderboard where more Whisper views do not.
    BEATs learns from acoustic-token prediction rather than labels (AST) or transcripts (Whisper). Its
    16x16 patches on a 128-bin, 10 ms Kaldi filterbank give 8 frequency x T/16 time tokens; tokens are
    averaged over frequency and interpolated from their centres (160 ms apart) onto the 20 ms grid, as
    ASTFrames does. The encoder has no absolute positions (convolutional + relative), so a whole clip
    (<= 24 s here) goes through in one pass. `name` is the checkpoint path, e.g.
    models/BEATs_iter3_plus_AS2M.pt (SHA-256 d43cbfad...).
    """

    def __init__(self, name, layers=(4, 8, 12), device="cuda"):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "beats"))
        from BEATs import BEATs, BEATsConfig
        ck = torch.load(name, map_location="cpu", weights_only=True)
        self.model = BEATs(BEATsConfig(ck["cfg"]))
        self.model.load_state_dict(ck["model"])
        self.model = self.model.to(device).eval()
        self.layers = tuple(layers)
        self.device = device
        self.dim = self.model.cfg.encoder_embed_dim
        self.patch = self.model.cfg.input_patch_size

    @torch.no_grad()
    def __call__(self, wave):
        n_frames = int(np.ceil(len(wave) / (SR / FPS)))
        centres = (np.arange(n_frames) + 0.5) / FPS
        m = self.model
        x = torch.from_numpy(np.asarray(wave, np.float32))[None].to(self.device)
        fbank = m.preprocess(x).unsqueeze(1)                        # (1, 1, frames, 128)
        feats = m.patch_embedding(fbank)                            # (1, C, T', F')
        t_p, f_p = feats.shape[2], feats.shape[3]
        feats = m.layer_norm(feats.reshape(1, feats.shape[1], -1).transpose(1, 2))
        if m.post_extract_proj is not None:
            feats = m.post_extract_proj(feats)
        _, layer_results = m.encoder.extract_features(feats, None, tgt_layer=len(m.encoder.layers))
        # token t covers filterbank frames [16t, 16t+16); frame i is centred at 10i + 12.5 ms
        t_c = (np.arange(t_p) * self.patch + self.patch / 2) * 0.01 + 0.0025
        idx = np.clip(np.searchsorted(t_c, centres) - 1, 0, max(t_p - 2, 0))
        nxt = np.minimum(idx + 1, t_p - 1)
        den = np.where(nxt > idx, t_c[nxt] - t_c[idx], 1.0)
        w = np.clip((centres - t_c[idx]) / den, 0, 1)[:, None]
        out = np.zeros((len(self.layers), n_frames, self.dim), np.float32)
        for li, l in enumerate(self.layers):
            tok = layer_results[l][0][:, 0].float().view(t_p, f_p, -1).mean(1).cpu().numpy()
            out[li] = (1 - w) * tok[idx] + w * tok[nxt]
        return torch.from_numpy(out).half()


def dasheng_logmel(wave, device):
    """Dasheng's front end without torchaudio: torchaudio.transforms.MelSpectrogram(16 kHz, n_fft 512,
    win 512 periodic Hann, hop 160, center/reflect, power 2, 64 HTK mel bands 0-8000 Hz, no norm) then
    AmplitudeToDB(power, top_db=120) with the top-dB floor taken per clip. Returns (1, 64, frames)."""
    x = torch.as_tensor(np.asarray(wave, np.float32), device=device)[None]
    spec = torch.stft(x, 512, hop_length=160, win_length=512, window=torch.hann_window(512, device=device),
                      center=True, pad_mode="reflect", normalized=False, onesided=True,
                      return_complex=True).abs().pow(2)                              # (1, 257, T)
    all_f = torch.linspace(0, 8000, 257, device=device)
    m = torch.linspace(0.0, 2595.0 * np.log10(1.0 + 8000 / 700.0), 66, device=device)
    f_pts = 700.0 * (10 ** (m / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_f[:, None]                                          # (257, 66)
    fb = torch.clamp(torch.minimum(-slopes[:, :-2] / f_diff[:-1], slopes[:, 2:] / f_diff[1:]), min=0)
    mel = torch.matmul(spec.transpose(1, 2), fb).transpose(1, 2)                     # (1, 64, T)
    db = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
    return torch.maximum(db, db.amax() - 120.0)


class DashengFrames:
    """Dasheng-base (Xiaomi, Apache-2.0; mispeech/dasheng-base, code reviewed) as a fourth architecture.

    Built for frame-level audio embeddings: 64x4 patches on a 64-band log-mel with 10 ms hop give one token per
    40 ms (the whole frequency axis in one patch), 4x finer in time than BEATs. Clips go through in 1008-frame
    (10.08 s) chunks as the model was trained, the last chunk zero-padded; tokens past the clip end are dropped.
    Layers are tapped with forward hooks on the transformer blocks (the last one after the final norm), token
    centres ((4j + 1.5) * 10 ms) are interpolated onto the 20 ms grid.
    """

    def __init__(self, name, layers=(4, 8, 12), device="cuda"):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
        self.enc = self.model.encoder
        self.layers = tuple(layers)
        self.device = device
        self.dim = self.enc.embed_dim
        self.chunk = self.enc.target_length
        self.stride = self.enc.patch_stride[-1]

    @torch.no_grad()
    def __call__(self, wave):
        n_frames = int(np.ceil(len(wave) / (SR / FPS)))
        centres = (np.arange(n_frames) + 0.5) / FPS
        x = dasheng_logmel(wave, self.device)[:, None]                               # (1, 1, 64, T)
        x = self.enc.init_bn(x)
        t_total = x.shape[-1]
        taps = {l: [] for l in self.layers}
        for a in range(0, t_total, self.chunk):
            c = x[..., a:a + self.chunk]
            n_tok = int(np.ceil(c.shape[-1] / self.stride))
            if c.shape[-1] < self.chunk:
                c = torch.nn.functional.pad(c, (0, self.chunk - c.shape[-1]))
            h = self.enc.patch_embed(c)
            h = h + self.enc.time_pos_embed[:, :, :, :h.shape[-1]] + self.enc.freq_pos_embed
            h = torch.permute(torch.flatten(h, 2, 3), (0, 2, 1))
            for i, blk in enumerate(self.enc.blocks):
                h = blk(h)
                if i + 1 in self.layers:
                    o = self.enc.norm(h) if i + 1 == len(self.enc.blocks) else h
                    taps[i + 1].append(o[0, :n_tok])
        t_c = (np.arange(sum(t.shape[0] for t in taps[self.layers[0]])) * self.stride + 1.5) * 0.01
        idx = np.clip(np.searchsorted(t_c, centres) - 1, 0, max(len(t_c) - 2, 0))
        nxt = np.minimum(idx + 1, len(t_c) - 1)
        den = np.where(nxt > idx, t_c[nxt] - t_c[idx], 1.0)
        w = np.clip((centres - t_c[idx]) / den, 0, 1)[:, None]
        out = np.zeros((len(self.layers), n_frames, self.dim), np.float32)
        for li, l in enumerate(self.layers):
            tok = torch.cat(taps[l]).float().cpu().numpy()
            out[li] = (1 - w) * tok[idx] + w * tok[nxt]
        return torch.from_numpy(out).half()


class CEDFrames:
    """CED-base (Xiaomi, Apache-2.0; mispeech/ced-base, code reviewed): a ViT distilled from an AudioSet teacher
    ensemble, on the same 64-band log-mel front end as Dasheng (dasheng_logmel). 16x16 patches give 4 frequency x
    63 time tokens per 1012-frame (10.12 s) chunk, one token per 160 ms; tokens are averaged over frequency and
    interpolated from their centres ((16j + 7.5) * 10 ms) onto the 20 ms grid.

    The model is built from the reviewed local files without transformers' remote-code path: CedConfig with
    outputdim=527 would fetch a label CSV from another repository (topel/ConvNeXt-Tiny-AT) at construction, so the
    encoder is built with outputdim=0 and the checkpoint's "encoder." weights are loaded directly, all keys checked.
    """

    def __init__(self, name, layers=(4, 8, 12), device="cuda"):
        import importlib
        import json
        import sys
        import types
        from safetensors.torch import load_file
        pkg = "ced_local"
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [os.path.abspath(name)]
            sys.modules[pkg] = mod
        cfg_mod = importlib.import_module(f"{pkg}.configuration_ced")
        mdl_mod = importlib.import_module(f"{pkg}.modeling_ced")
        raw = json.load(open(os.path.join(name, "config.json")))
        keep = ("name", "depth", "embed_dim", "num_heads", "mlp_ratio", "patch_size", "patch_stride", "pooling",
                "qkv_bias", "target_length", "n_mels", "pad_last", "eval_avg")
        cfg = cfg_mod.CedConfig(**{k: raw[k] for k in keep if k in raw}, outputdim=0)
        self.model = mdl_mod.CedModel(cfg)
        sd = {k[len("encoder."):]: v for k, v in load_file(os.path.join(name, "model.safetensors")).items()
              if k.startswith("encoder.")}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing[:3], unexpected[:3])
        self.model = self.model.to(device).eval()
        self.layers = tuple(layers)
        self.device = device
        self.dim = cfg.embed_dim
        self.chunk = cfg.target_length
        self.stride = cfg.patch_stride if isinstance(cfg.patch_stride, int) else cfg.patch_stride[-1]
        self.f_tokens = cfg.n_mels // (cfg.patch_stride if isinstance(cfg.patch_stride, int) else cfg.patch_stride[0])

    @torch.no_grad()
    def __call__(self, wave):
        n_frames = int(np.ceil(len(wave) / (SR / FPS)))
        centres = (np.arange(n_frames) + 0.5) / FPS
        m = self.model
        x = dasheng_logmel(wave, self.device)[:, None]                               # (1, 1, 64, T)
        x = torch.permute(m.init_bn(torch.permute(x, (0, 2, 1, 3))), (0, 2, 1, 3))
        taps = {l: [] for l in self.layers}
        for a in range(0, x.shape[-1], self.chunk):
            c = x[..., a:a + self.chunk]
            n_tok = max(1, c.shape[-1] // self.stride)
            if c.shape[-1] < self.chunk:
                c = torch.nn.functional.pad(c, (0, self.chunk - c.shape[-1]))
            h = m.patch_embed(c)
            t = h.shape[-1]
            h = h + m.time_pos_embed[:, :, :, :t] + m.freq_pos_embed
            h = torch.permute(torch.flatten(h, 2, 3), (0, 2, 1))                   # (1, f*t, D), f-major
            for i, blk in enumerate(m.blocks):
                h = blk(h)
                if i + 1 in self.layers:
                    o = m.norm(h) if i + 1 == len(m.blocks) else h
                    taps[i + 1].append(o[0].view(self.f_tokens, t, -1).mean(0)[:n_tok])
        n_all = sum(tk.shape[0] for tk in taps[self.layers[0]])
        t_c = (np.arange(n_all) * self.stride + self.stride / 2 - 0.5) * 0.01
        idx = np.clip(np.searchsorted(t_c, centres) - 1, 0, max(n_all - 2, 0))
        nxt = np.minimum(idx + 1, n_all - 1)
        den = np.where(nxt > idx, t_c[nxt] - t_c[idx], 1.0)
        w = np.clip((centres - t_c[idx]) / den, 0, 1)[:, None]
        out = np.zeros((len(self.layers), n_frames, self.dim), np.float32)
        for li, l in enumerate(self.layers):
            tok = torch.cat(taps[l]).float().cpu().numpy()
            out[li] = (1 - w) * tok[idx] + w * tok[nxt]
        return torch.from_numpy(out).half()


class WhisperFrames:
    def __init__(self, name="models/whisper-small", layers=(3, 6, 9, 12), device="cuda"):
        self.fe = WhisperFeatureExtractor.from_pretrained(name)
        enc = load_encoder(name)
        self.enc = enc.to(device).eval()
        self.layers = tuple(layers)
        self.device = device
        self.dim = enc.config.d_model

    @torch.no_grad()
    def __call__(self, wave):
        """wave: float32 mono 16 kHz -> (len(layers), T, d) float16 tensor, T = ceil(n/320)."""
        n_frames = int(np.ceil(len(wave) / (SR / FPS)))
        max_s = self.enc.embed_positions.num_embeddings / FPS
        chunks = []
        # Whisper's positional table covers 30 s; longer clips go in 30 s pieces.
        step = int(max_s * SR)
        for a in range(0, len(wave), step):
            chunks.append(self._run(wave[a:a + step]))
        out = torch.cat(chunks, dim=1)
        if out.shape[1] < n_frames:
            out = torch.nn.functional.pad(out, (0, 0, 0, n_frames - out.shape[1]))
        return out[:, :n_frames]

    def _run(self, wave):
        mel = self.fe(wave, sampling_rate=SR, return_tensors="pt", padding="do_not_pad",
                      device=self.device).input_features  # (1, 80, frames)
        if mel.shape[-1] % 2:
            mel = torch.nn.functional.pad(mel, (0, 1), value=float(mel.min()))
        x = mel.to(self.device, torch.float16)
        e = self.enc
        h = torch.nn.functional.gelu(e.conv1(x))
        h = torch.nn.functional.gelu(e.conv2(h)).permute(0, 2, 1)
        h = h + e.embed_positions.weight[: h.shape[1]].to(h.dtype)
        hs = [h]
        for layer in e.layers:
            h = layer(h, None)
            hs.append(h)
        hs[-1] = e.layer_norm(hs[-1])
        return torch.stack([hs[i][0] for i in self.layers])
