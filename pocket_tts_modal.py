"""
Pocket-TTS Fine-tuning & Inference on Modal
=============================================

Fine-tunes Kyutai's Pocket-TTS (100M param) text-to-speech model on the
MrDragonFox/Elise dataset using LoRA + flow matching loss, then serves
a web endpoint that generates spoken audio from text.

Pocket-TTS is fundamentally different from CSM:
  - Uses continuous latents (32-dim at 12.5 Hz) instead of discrete tokens
  - Trained with flow matching loss (predict velocity field), not cross-entropy
  - Has a Mimi audio codec (frozen during fine-tuning) + FlowLM backbone
  - Only 100M params total — fits comfortably on an L4 GPU

Architecture:
  FlowLM (trainable via LoRA):
    - Causal Transformer backbone (6 layers, d=1024, 16 heads)
    - Flow MLP head (SimpleMLPAdaLN, 6 ResBlocks, d=512)
    - SentencePiece text conditioner (4000 vocab)
    - Speaker projection for voice conditioning
  Mimi (frozen):
    - SEANet encoder/decoder + transformer
    - Produces 32-dim continuous latents at 12.5 Hz
    - 24 kHz sample rate

Usage:
  # 1. Fine-tune (default 60 steps)
  modal run pocket_tts_modal.py
  modal run pocket_tts_modal.py --max-steps 120

  # 2. Smoke test (quick validation)
  modal run pocket_tts_modal.py --smoke-test

  # 3. Deploy inference endpoint
  modal deploy pocket_tts_modal.py

  # 4. Test inference
  modal run pocket_tts_modal.py --action test --text "Hello world"
"""

import math
import modal

# ── Modal app ────────────────────────────────────────────────────────────────

app = modal.App("pocket-tts-finetune")

# ── Persistent volume ────────────────────────────────────────────────────────
# Sub-directories:
#   /vol/hf_cache/          – HuggingFace download cache
#   /vol/dataset_pocket/    – Preprocessed (Mimi-encoded) dataset
#   /vol/pocket_lora/       – Saved LoRA + fine-tuned weights
#   /vol/pocket_merged/     – Merged full model weights

volume = modal.Volume.from_name("pocket-tts-vol", create_if_missing=True)
VOL = "/vol"

DATASET_PATH = f"{VOL}/dataset_pocket"
LORA_PATH = f"{VOL}/pocket_lora"
MERGED_PATH = f"{VOL}/pocket_merged"

# ── Container image ─────────────────────────────────────────────────────────
# pocket-tts requires torch>=2.5.  On Modal with GPU, the default PyPI torch
# includes CUDA support.  We install torch first, then pocket-tts so it
# doesn't pull in a conflicting version.

hf_secret = modal.Secret.from_name("my-huggingface-secret")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg", "git")
    .pip_install(
        "torch>=2.5.0",
    )
    .pip_install(
        "pocket-tts>=1.0.0",
        "datasets>=3.0.0",
        "soundfile",
        "librosa",
        "safetensors",
        "torchcodec",
        "hf_transfer>=0.1.9",
        "huggingface_hub>=0.13.0",
        "fastapi[standard]",
        "scipy",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": f"{VOL}/hf_cache",
    })
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LoRA IMPLEMENTATION                                                     ║
# ╠═══════════════════════════════════════════════════════════════════════════╣
# ║  We implement LoRA manually because pocket-tts uses custom PyTorch       ║
# ║  modules (not HuggingFace), and PEFT's model wrapping would break        ║
# ║  the attribute access patterns used throughout the codebase.             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class LoRALinear:
    """Drop-in replacement for nn.Linear with low-rank adaptation.

    Creates two small matrices A (r × in_features) and B (out_features × r)
    that add a low-rank perturbation to the original linear layer's output:
        output = original(x) + (x @ A^T @ B^T) * scale
    B is initialized to zero, so the LoRA starts as an identity (no change).
    """

    @staticmethod
    def create(original, r=16, alpha=16):
        """Create a LoRALinear module wrapping an existing nn.Linear."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _LoRALinear(nn.Module):
            def __init__(self, orig, r, alpha):
                super().__init__()
                self.original = orig
                self.r = r
                self.scale = alpha / r

                # Freeze original weights
                self.original.weight.requires_grad = False
                if self.original.bias is not None:
                    self.original.bias.requires_grad = False

                # LoRA low-rank matrices
                self.lora_A = nn.Parameter(
                    torch.zeros(r, orig.in_features,
                                device=orig.weight.device,
                                dtype=orig.weight.dtype)
                )
                self.lora_B = nn.Parameter(
                    torch.zeros(orig.out_features, r,
                                device=orig.weight.device,
                                dtype=orig.weight.dtype)
                )
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                # lora_B stays zero → LoRA starts as identity

            @property
            def weight(self):
                """Proxy .weight to the original linear layer for compatibility."""
                return self.original.weight

            @property
            def bias(self):
                """Proxy .bias to the original linear layer for compatibility."""
                return self.original.bias

            @property
            def in_features(self):
                return self.original.in_features

            @property
            def out_features(self):
                return self.original.out_features

            def forward(self, x):
                result = self.original(x)
                lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
                return result + lora_out

            def merge_weights(self):
                """Merge LoRA into the original linear and return it."""
                with torch.no_grad():
                    self.original.weight.add_(
                        (self.lora_B @ self.lora_A) * self.scale
                    )
                return self.original

        return _LoRALinear(original, r, alpha)


def apply_lora(model, target_names, r=16, alpha=16):
    """Replace target nn.Linear modules in `model` with LoRA-wrapped versions.

    Args:
        model: nn.Module to modify in-place
        target_names: list of leaf module names to wrap (e.g. ["in_proj", "out_proj"])
        r: LoRA rank
        alpha: LoRA alpha (scaling = alpha / r)

    Returns:
        list of replaced module paths
    """
    import torch.nn as nn

    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        parts = name.split(".")
        leaf_name = parts[-1]
        if leaf_name not in target_names:
            continue

        # Navigate to parent module
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        # Replace with LoRA wrapper
        lora_layer = LoRALinear.create(module, r=r, alpha=alpha)
        setattr(parent, leaf_name, lora_layer)
        replaced.append(name)

    return replaced


def merge_lora(model):
    """Merge all LoRA layers back into their base nn.Linear layers."""
    for name, module in list(model.named_modules()):
        if not hasattr(module, "merge_weights"):
            continue
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        merged = module.merge_weights()
        setattr(parent, parts[-1], merged)


def save_finetuned_weights(flow_lm, path):
    """Save all trainable parameters (LoRA + other fine-tuned layers)."""
    import os
    import safetensors.torch

    os.makedirs(path, exist_ok=True)

    state = {}
    for name, param in flow_lm.named_parameters():
        if param.requires_grad:
            state[name] = param.data

    safetensors.torch.save_file(state, f"{path}/finetuned_weights.safetensors")

    # Also save LoRA config for reconstruction
    import json
    config = {
        "target_names": ["in_proj", "out_proj", "linear1", "linear2"],
        "r": 32,
        "alpha": 32,
        "finetuned_modules": ["flow_net", "out_eos", "speaker_proj_weight"],
    }
    with open(f"{path}/lora_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved {len(state)} tensors ({sum(v.numel() for v in state.values()):,} params)")


def load_finetuned_weights(flow_lm, path):
    """Load fine-tuned weights (LoRA + other) into the model."""
    import safetensors.torch

    state = safetensors.torch.load_file(f"{path}/finetuned_weights.safetensors")
    loaded = 0
    for name, param in flow_lm.named_parameters():
        if name in state:
            param.data = state[name].to(param.device, dtype=param.dtype)
            loaded += 1

    print(f"  Loaded {loaded} tensors from {path}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SAMPLE GENERATION HELPERS                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

TEST_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello! How are you doing today?",
    "Welcome to the world of artificial intelligence.",
    "She sells sea shells by the sea shore.",
    "I can't believe how good this sounds!",
    "The weather is absolutely beautiful today.",
    "Technology is advancing at an incredible pace.",
    "Would you like to hear a story?",
    "Every moment is a fresh beginning.",
    "Music is the universal language of mankind.",
]


def _build_voice_state(model, audio_path):
    """Build a voice state from an audio file.

    When has_voice_cloning is True, delegates to the model's built-in method.
    When False, builds the state manually (bypasses the gate check).
    """
    import torch
    from pathlib import Path

    if model.has_voice_cloning:
        return model.get_state_for_audio_prompt(Path(audio_path))
    else:
        # Manual approach for non-voice-cloning model
        import soundfile as sf_read
        from pocket_tts.modules.stateful_module import init_states, increment_steps
        from pocket_tts.data.audio_utils import convert_audio

        audio, sr = sf_read.read(audio_path)
        audio_t = torch.tensor(audio, dtype=torch.float32)
        if audio_t.dim() == 1:
            audio_t = audio_t.unsqueeze(0)
        if sr != model.sample_rate:
            audio_t = convert_audio(audio_t, sr, model.sample_rate, 1)

        with torch.no_grad():
            conditioning = model._encode_audio(audio_t.unsqueeze(0))

        model_state = init_states(
            model.flow_lm, batch_size=1,
            sequence_length=conditioning.shape[1],
        )
        model._run_flow_lm_and_increment_step(
            model_state=model_state, audio_conditioning=conditioning,
        )

        for mn in model_state:
            for k in model_state[mn]:
                if isinstance(model_state[mn][k], torch.Tensor):
                    model_state[mn][k] = model_state[mn][k].detach().clone()

        return model_state


def _generate_samples(model, prompts, voice="alba"):
    """Generate audio for each prompt.  Returns [(prompt, wav_bytes), …].

    Args:
        model: TTSModel instance
        prompts: list of text strings
        voice: voice identifier — can be:
            - "alba" (built-in default voice)
            - a file path to a .wav file (e.g. /vol/dataset_pocket/reference.wav)
            - a torch.Tensor of audio

    Pocket-TTS is designed for CPU inference, so we move the model to CPU
    for generation, then move it back to its original device.
    """
    import torch
    import soundfile as sf
    import io

    was_training = model.training
    original_device = model.device
    model.eval()
    model.to("cpu")

    import os
    if isinstance(voice, str) and os.path.exists(voice):
        print(f"      Using voice file: {os.path.basename(voice)}")
        voice_state = _build_voice_state(model, voice)
    else:
        voice_state = model.get_state_for_audio_prompt(voice)

    samples = []
    for i, prompt in enumerate(prompts):
        try:
            with torch.no_grad():
                audio = model.generate_audio(voice_state, prompt, copy_state=True)
            audio_np = audio.numpy()
            buf = io.BytesIO()
            sf.write(buf, audio_np, model.sample_rate, format="WAV")
            samples.append((prompt, buf.getvalue()))
        except Exception as e:
            print(f"      ⚠  Failed on prompt {i}: {e}")
            # Create a tiny silent WAV as placeholder
            import numpy as np
            buf = io.BytesIO()
            sf.write(buf, np.zeros(4800, dtype=np.float32), 24000, format="WAV")
            samples.append((prompt, buf.getvalue()))
        print(f"      [{i+1}/{len(prompts)}] {prompt[:40]}…")

    # Move model back to training device
    model.to(original_device)
    if was_training:
        model.train()
    return samples


def _build_html(all_samples):
    """Build self-contained HTML with embedded <audio> players."""
    import base64
    from datetime import datetime

    rows = ""
    for label, samples in all_samples.items():
        cards = ""
        for prompt, wav_bytes in samples:
            b64 = base64.b64encode(wav_bytes).decode()
            cards += (
                '<div class="sample">'
                f'<p class="prompt">"{prompt}"</p>'
                f'<audio controls preload="none" '
                f'src="data:audio/wav;base64,{b64}"></audio>'
                "</div>\n"
            )
        rows += f'<div class="checkpoint"><h2>{label}</h2>\n{cards}</div>\n'

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Pocket-TTS Fine-tuning Samples</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px;
         margin: 0 auto; padding: 24px; background: #fafafa; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 24px; }}
  .checkpoint {{ background: #fff; border: 1px solid #ddd;
                 border-radius: 10px; padding: 20px; margin: 20px 0;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .checkpoint h2 {{ margin-top: 0; color: #c42; }}
  .sample {{ margin: 14px 0; }}
  .prompt {{ margin: 4px 0; color: #333; font-style: italic; }}
  audio {{ width: 100%; height: 36px; }}
</style></head><body>
<h1>🗣️ Pocket-TTS Fine-tuning Samples</h1>
<p class="meta">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
 &mdash; model: Pocket-TTS (100M)
 &mdash; dataset: MrDragonFox/Elise
 &mdash; {len(all_samples)} checkpoints
 &times; {len(next(iter(all_samples.values())))} prompts</p>
{rows}
</body></html>"""


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TRAINING                                                                ║
# ╠═══════════════════════════════════════════════════════════════════════════╣
# ║  Flow Matching Loss (LSD 1-step):                                        ║
# ║    1. Encode target audio with Mimi → continuous latents (frozen)        ║
# ║    2. Normalize latents: (x - mean) / std                               ║
# ║    3. Teacher-forced backbone: [voice_cond | text_emb | shifted_latents] ║
# ║    4. At each position, sample noise x_0 ~ N(0,I)                       ║
# ║    5. Flow net predicts velocity: v(c, s=0, t=1, x_0) ≈ x_1 - x_0      ║
# ║    6. Loss = MSE(predicted_velocity, true_velocity)                      ║
# ║  Matches the inference regime (LSD decode with 1 step: s=0, t=1).       ║
# ║  Plus EOS loss (binary cross-entropy) for end-of-sequence detection.     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@app.function(
    image=image,
    gpu="L4",          # 100M model fits easily on L4 (24 GB)
    volumes={VOL: volume},
    secrets=[hf_secret],
    timeout=7200,      # 2 hours
)
def train(max_steps: int = 1200, smoke_test: bool = False):
    """Fine-tune Pocket-TTS on MrDragonFox/Elise with LoRA + flow matching."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pocket_tts import TTSModel
    from pocket_tts.conditioners.base import TokenizedText
    from collections import OrderedDict
    from datasets import load_dataset, Audio
    import os
    import time as time_module

    if smoke_test:
        max_steps = 2

    tag = "SMOKE TEST" if smoke_test else "Full"
    print("=" * 60)
    print(f"  Pocket-TTS  ·  LoRA Fine-tuning  ·  {tag}")
    print("=" * 60)

    # ── HuggingFace auth (enables voice cloning model) ────────────────
    import huggingface_hub
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)
        print("  ✅  Logged into HuggingFace (voice cloning enabled)")
    else:
        print("  ⚠  No HF_TOKEN — voice cloning may not be available")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Load base model ────────────────────────────────────────────────
    print("\n[1/5] Loading Pocket-TTS model …")
    model = TTSModel.load_model()
    print(f"  Voice cloning: {'✅ enabled' if model.has_voice_cloning else '❌ disabled'}")
    model = model.to(device)
    volume.commit()

    flow_params = sum(p.numel() for p in model.flow_lm.parameters())
    mimi_params = sum(p.numel() for p in model.mimi.parameters())
    print(f"  FlowLM: {flow_params:,} params")
    print(f"  Mimi:   {mimi_params:,} params (frozen)")
    print(f"  Total:  {flow_params + mimi_params:,} params")

    # ── 2. Apply LoRA + selective fine-tuning ─────────────────────────────
    print("\n[2/5] Applying LoRA (r=16) to transformer …")

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Apply LoRA to transformer attention projections and FFN
    replaced = apply_lora(
        model.flow_lm,
        target_names=["in_proj", "out_proj", "linear1", "linear2"],
        r=32,
        alpha=32,
    )
    print(f"  LoRA applied to {len(replaced)} layers")

    # flow_net stays FROZEN — it's too sensitive and collapses even with
    # the full dataset at lr>=1e-4. The backbone LoRA adapts instead.

    # Unfreeze EOS head (adapts to voice's speaking rhythm)
    for p in model.flow_lm.out_eos.parameters():
        p.requires_grad = True

    # Unfreeze speaker projection (improves voice conditioning)
    if hasattr(model.flow_lm, "speaker_proj_weight"):
        model.flow_lm.speaker_proj_weight.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ── 3. Prepare dataset ────────────────────────────────────────────────
    # Key insight: Mimi encoder outputs 512-dim latents, but the FlowLM
    # backbone operates on 32-dim latents (the VAE bottleneck).
    # The DummyQuantizer has a learned projection 32→512 (for decoding).
    # To get 32-dim training targets from audio, we:
    #   1. Encode audio with Mimi → 512-dim
    #   2. Project 512→32 using pseudo-inverse of quantizer weights
    # This gives us the optimal 32-dim representation.
    print("\n[3/5] Preparing dataset …")

    SMOKE_N = 8
    ds_tag = "_smoke" if smoke_test else ""
    ds_path = f"{DATASET_PATH}{ds_tag}"
    encoded_file = f"{ds_path}/encoded.pt"

    if os.path.exists(encoded_file):
        cache = torch.load(encoded_file, weights_only=False)
        # Verify cache has enough examples (stale smoke-test caches may exist)
        cached_n = len(cache["latents"])
        if not smoke_test and cached_n < 20:
            print(f"  ⚠  Stale cache has only {cached_n} examples — re-encoding full dataset")
            os.remove(encoded_file)
            cache = None
        else:
            print(f"  ✅  Cached encoded dataset found ({cached_n} examples)")
    else:
        cache = None

    if cache is not None:
        latent_cache = cache["latents"]          # list of [T, 32] tensors
        enc512_cache = cache["enc512"]            # list of [T, 512] tensors
        text_cache = cache["texts"]

        # Ensure we have a reference voice clip (may be missing from old caches)
        ref_path = f"{ds_path}/elise_reference.wav"
        if not os.path.exists(ref_path):
            print("  ⬇  Downloading reference Elise clip …")
            raw_ds = load_dataset("MrDragonFox/Elise", split="train")
            raw_ds = raw_ds.cast_column("audio", Audio(sampling_rate=24000))
            # Find a 5-10s clip
            import soundfile as sf_save
            for ex in raw_ds:
                arr = torch.tensor(ex["audio"]["array"], dtype=torch.float32)
                dur = arr.shape[0] / 24000
                if 5.0 <= dur <= 10.0:
                    sf_save.write(ref_path, arr.numpy(), 24000)
                    print(f"  Saved reference voice clip: {ref_path} ({dur:.1f}s)")
                    break
            else:
                # Fallback: use first clip
                arr = torch.tensor(raw_ds[0]["audio"]["array"], dtype=torch.float32)
                sf_save.write(ref_path, arr.numpy(), 24000)
                print(f"  Saved reference voice clip (fallback): {ref_path}")
            volume.commit()
    else:
        print("  ⬇  Downloading & encoding MrDragonFox/Elise …")
        raw_ds = load_dataset("MrDragonFox/Elise", split="train")

        if smoke_test:
            raw_ds = raw_ds.select(range(min(SMOKE_N, len(raw_ds))))
            print(f"  🔬  Smoke test: {len(raw_ds)} examples")

        raw_ds = raw_ds.cast_column("audio", Audio(sampling_rate=24000))

        # Compute pseudo-inverse of quantizer weight for 512→32 projection
        # quantizer.output_proj.weight has shape [512, 32, 1]
        W = model.mimi.quantizer.output_proj.weight.squeeze(-1)  # [512, 32]
        W_pinv = torch.linalg.pinv(W)                             # [32, 512]
        print(f"  Quantizer weight: {W.shape} → pseudo-inverse: {W_pinv.shape}")

        latent_cache = []
        enc512_cache = []
        text_cache = []
        skipped = 0

        # Pick a good reference clip for voice conditioning during evaluation
        # (prefer a clip around 5-10 seconds for clear voice identity)
        best_ref_idx = None
        best_ref_audio = None
        best_ref_len = 0

        model.mimi.eval()
        for i, example in enumerate(raw_ds):
            audio_array = torch.tensor(
                example["audio"]["array"], dtype=torch.float32
            )

            # Skip very short audio (< 1.5 seconds)
            if audio_array.shape[0] < 36000:
                skipped += 1
                continue

            # Track best reference clip (5-10 seconds ideal)
            dur_s = audio_array.shape[0] / 24000
            if 5.0 <= dur_s <= 10.0 and dur_s > best_ref_len:
                best_ref_idx = i
                best_ref_audio = audio_array.clone()
                best_ref_len = dur_s

            # Truncate very long audio (> 30 seconds)
            max_samples = 30 * 24000
            if audio_array.shape[0] > max_samples:
                audio_array = audio_array[:max_samples]

            # Encode with Mimi → [1, 512, T_frames]
            audio_tensor = audio_array.unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                enc_512 = model.mimi.encode_to_latent(audio_tensor)

            # Project 512→32 using pseudo-inverse: [1, 32, T] = W_pinv @ [1, 512, T]
            with torch.no_grad():
                latents_32 = F.conv1d(
                    enc_512, W_pinv.unsqueeze(-1).to(enc_512.device)
                )

            # Store both representations on CPU
            #   enc512: [T, 512] (for voice conditioning during training)
            #   latent_32: [T, 32] (flow matching target)
            enc512_cache.append(enc_512.squeeze(0).transpose(0, 1).cpu())
            latent_cache.append(latents_32.squeeze(0).transpose(0, 1).cpu())
            text_cache.append(example["text"])

            if (i + 1) % 100 == 0:
                print(f"    Encoded {i+1}/{len(raw_ds)} …")

        # Fallback: use the longest clip if no 5-10s clip found
        if best_ref_audio is None:
            best_ref_audio = torch.tensor(
                raw_ds[0]["audio"]["array"], dtype=torch.float32
            )
            best_ref_len = best_ref_audio.shape[0] / 24000

        print(f"  Encoded {len(latent_cache)} examples (skipped {skipped} short)")

        os.makedirs(ds_path, exist_ok=True)

        # Save reference Elise audio clip for voice conditioning at eval time
        import soundfile as sf_save
        ref_path = f"{ds_path}/elise_reference.wav"
        sf_save.write(ref_path, best_ref_audio.numpy(), 24000)
        print(f"  Saved reference voice clip: {ref_path} ({best_ref_len:.1f}s)")
        torch.save(
            {"latents": latent_cache, "enc512": enc512_cache, "texts": text_cache},
            encoded_file,
        )
        volume.commit()

    print(f"  Dataset: {len(latent_cache)} examples")
    avg_frames = sum(l.shape[0] for l in latent_cache) / len(latent_cache)
    print(f"  Avg length: {avg_frames:.0f} frames ({avg_frames/12.5:.1f}s)")
    print(f"  Latent dim: {latent_cache[0].shape[-1]}  (should be 32)")
    print(f"  Encoder dim: {enc512_cache[0].shape[-1]}  (should be 512)")

    # Use Elise reference voice for sample generation (requires voice cloning)
    elise_ref = f"{ds_path}/elise_reference.wav"
    if os.path.exists(elise_ref):
        eval_voice = elise_ref
        print(f"  Voice for samples: Elise reference ({elise_ref})")
    else:
        eval_voice = "alba"
        print(f"  ⚠  No Elise reference found, using alba voice")

    # ── 4. Train ──────────────────────────────────────────────────────────
    PROMPT_FRAMES = 25      # ~2 seconds of voice conditioning
    N_NOISE = 4             # noise samples per frame for flow matching
    GRAD_ACCUM = 8          # effective batch size
    EOS_WEIGHT = 1.0        # weight for EOS loss (raised from 0.1 to fix EOS)
    EOS_TAIL = 3            # mark last N frames as EOS (reduces class imbalance)

    sample_every = max(1, max_steps // 6)
    print(f"\n[4/5] Training for {max_steps} steps  "
          f"(samples every {sample_every}, grad_accum={GRAD_ACCUM})")

    if device == "cuda":
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gpu_props.name}  "
              f"({round(gpu_props.total_memory / 1024**3, 1)} GB)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )

    # Linear LR schedule with warmup
    warmup_steps = 5

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, 1.0 - (step - warmup_steps) / max(max_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Checkpoint sample collection ─────────────────────────────────────
    all_samples = OrderedDict()

    print("\n  🔊  Generating baseline samples (step 0) …")
    print("    --- Alba voice (built-in reference) ---")
    all_samples["Step 0 — Baseline (alba)"] = _generate_samples(
        model, TEST_PROMPTS, voice="alba"
    )
    if eval_voice != "alba":
        print("    --- Elise voice (target) ---")
        all_samples["Step 0 — Baseline (elise)"] = _generate_samples(
            model, TEST_PROMPTS, voice=eval_voice
        )

    # ── Training loop ────────────────────────────────────────────────────
    model.flow_lm.train()
    model.mimi.eval()
    optimizer.zero_grad()

    t_start = time_module.time()

    for step in range(max_steps):
        idx = step % len(latent_cache)
        raw_latents_32 = latent_cache[idx].to(device)    # [T_total, 32]
        raw_enc_512 = enc512_cache[idx].to(device)       # [T_total, 512]
        text = text_cache[idx]

        T_total = raw_latents_32.shape[0]

        # ── Split into voice prompt + target ─────────────────────────
        # First ~2 seconds → voice conditioning (512-dim through speaker_proj)
        # Rest → target for flow matching loss (32-dim latents)
        if T_total > PROMPT_FRAMES + 10:
            prompt_enc512 = raw_enc_512[:PROMPT_FRAMES].unsqueeze(0)       # [1, 25, 512]
            target_latents = raw_latents_32[PROMPT_FRAMES:].unsqueeze(0)   # [1, T, 32]
        else:
            # Too short for splitting — no voice conditioning
            prompt_enc512 = None
            target_latents = raw_latents_32.unsqueeze(0)                   # [1, T, 32]

        B, T, ldim = target_latents.shape

        # ── Voice conditioning ───────────────────────────────────────
        # Project 512-dim encoder output through speaker_proj_weight → 1024-dim
        # (matches inference: TTSModel._encode_audio does
        #   F.linear(mimi_encoder_output, speaker_proj_weight))
        if prompt_enc512 is not None:
            audio_cond = F.linear(
                prompt_enc512.to(model.flow_lm.speaker_proj_weight.dtype),
                model.flow_lm.speaker_proj_weight,
            )   # [1, 25, 1024]
        else:
            audio_cond = torch.empty(
                1, 0, model.flow_lm.dim,
                device=device,
                dtype=model.flow_lm.speaker_proj_weight.dtype,
            )

        # ── Normalize target latents for backbone ────────────────────
        emb_mean = model.flow_lm.emb_mean
        emb_std = model.flow_lm.emb_std
        target_norm = (target_latents - emb_mean) / emb_std

        # ── Text embedding ───────────────────────────────────────────
        # conditioner.prepare(text) → TokenizedText(tokens=[1, S])
        # conditioner(prepared) → embed(tokens) → [1, S, 1024]
        prepared = model.flow_lm.conditioner.prepare(text)
        # Ensure tokens are Long (GPU round-trip during sample generation can cast)
        prepared = TokenizedText(prepared.tokens.long())
        text_emb = model.flow_lm.conditioner(prepared)   # [1, S_text, 1024]
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(0)              # safety: ensure 3D

        # ── Combine conditioning prefix ──────────────────────────────
        # Inference ordering: [audio_cond | text_emb | latents]
        # (audio prompt is processed first, then text, then generation)
        cond_prefix = torch.cat([audio_cond, text_emb], dim=1)

        # ── Teacher-forced shifted input (BOS + target[:-1]) ─────────
        bos = torch.full((B, 1, ldim), float("nan"), device=device)
        shifted = torch.cat([bos, target_norm[:, :-1]], dim=1)   # [1, T, 32]
        bos_emb = model.flow_lm.bos_emb                         # [32]
        shifted = torch.where(
            torch.isnan(shifted),
            bos_emb.view(1, 1, -1).expand_as(shifted),
            shifted,
        )
        input_ = model.flow_lm.input_linear(shifted)            # [1, T, 1024]

        # ── Full input: [cond_prefix | audio_latent_input] ───────────
        full_input = torch.cat([cond_prefix, input_], dim=1)

        # ── Causal transformer (full-sequence with KV cache) ────────
        # pocket-tts 1.1+ requires a proper model_state dict (not None).
        # We create a KV cache sized for the full sequence and process
        # everything in one shot — equivalent to non-streaming causal attn.
        from pocket_tts.modules.stateful_module import init_states as _init_states
        seq_len = full_input.shape[1]
        fwd_state = _init_states(model.flow_lm, batch_size=B, sequence_length=seq_len)
        # Move state tensors to the right device
        for k, v in fwd_state.items():
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                    fwd_state[k][kk] = vv.to(device)
        transformer_out = model.flow_lm.transformer(full_input, model_state=fwd_state)
        transformer_out = model.flow_lm.out_norm(transformer_out)
        backbone_out = transformer_out[:, -T:]                   # [1, T, 1024]
        backbone_out = backbone_out.to(torch.float32)

        # ── Flow Matching Loss (LSD 1-step) ─────────────────────────
        # Pocket-TTS uses LSD (Lagrangian Self-Distillation) with 1-step
        # decode at inference: flow_net(c, s=0, t=1, x=noise) → velocity.
        # The model predicts the full velocity from noise to clean data.
        #
        # For fine-tuning, we match the inference regime exactly:
        #   - s=0, t=1 (always — the model was distilled for 1-step)
        #   - x = pure noise x_0 ~ N(0, I)  (NOT interpolated x_t!)
        #   - target = x_1 - x_0  (full OT velocity)
        #   - Loss = MSE(predicted_velocity, true_velocity)
        bb_flat = backbone_out.reshape(-1, backbone_out.shape[-1])  # [T, 1024]

        loss_fm = torch.tensor(0.0, device=device)
        s_time = torch.zeros(B * T, 1, device=device)   # always 0
        t_time = torch.ones(B * T, 1, device=device)    # always 1

        for _ in range(N_NOISE):
            x_0 = torch.randn(B, T, ldim, device=device)        # pure noise
            x_1 = target_norm                                     # clean data
            velocity_target = x_1 - x_0                           # OT velocity

            x0_flat = x_0.reshape(-1, ldim)                       # [T, 32]

            pred = model.flow_lm.flow_net(
                bb_flat,
                s=s_time,
                t=t_time,
                x=x0_flat,
            )
            pred = pred.reshape(B, T, ldim)
            loss_fm = loss_fm + F.mse_loss(pred, velocity_target)

        loss_fm = loss_fm / N_NOISE

        # ── EOS Loss ─────────────────────────────────────────────────
        eos_logits = model.flow_lm.out_eos(backbone_out).squeeze(-1)   # [1, T]
        eos_labels = torch.zeros(B, T, device=device)
        eos_labels[:, -EOS_TAIL:] = 1.0   # last N frames are EOS
        loss_eos = F.binary_cross_entropy_with_logits(eos_logits, eos_labels)

        # ── Backward ─────────────────────────────────────────────────
        loss = (loss_fm + EOS_WEIGHT * loss_eos) / GRAD_ACCUM
        loss.backward()

        # ── Gradient step ────────────────────────────────────────────
        if (step + 1) % GRAD_ACCUM == 0 or step == max_steps - 1:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # ── Logging ──────────────────────────────────────────────────
        if step % 5 == 0 or step == max_steps - 1:
            elapsed = time_module.time() - t_start
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Step {step:4d}/{max_steps}  "
                f"loss={loss.item()*GRAD_ACCUM:.4f}  "
                f"fm={loss_fm.item():.4f}  "
                f"eos={loss_eos.item():.4f}  "
                f"lr={lr:.2e}  "
                f"T={T}  [{elapsed:.0f}s]"
            )

        # ── Checkpoint samples ───────────────────────────────────────
        if (step + 1) % sample_every == 0 and step > 0:
            print(f"\n  🔊  Generating samples at step {step + 1} …")
            all_samples[f"Step {step + 1}"] = _generate_samples(
                model, TEST_PROMPTS, voice=eval_voice
            )
            model.flow_lm.train()

    secs = time_module.time() - t_start
    if device == "cuda":
        peak = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
    else:
        peak = 0
    print(f"\n  ✅  Training done in {secs:.0f}s ({secs/60:.1f} min)")
    print(f"  Peak GPU memory: {peak} GB")

    # Final samples — generate with both voices for comparison
    final_key = f"Step {max_steps}"
    if final_key not in all_samples:
        print(f"\n  🔊  Generating final samples …")
        all_samples[f"Step {max_steps} — Final (elise)"] = _generate_samples(
            model, TEST_PROMPTS, voice=eval_voice
        )
        print("    --- Alba voice (comparison) ---")
        all_samples[f"Step {max_steps} — Final (alba)"] = _generate_samples(
            model, TEST_PROMPTS, voice="alba"
        )

    # ── 5. Save weights + build HTML ─────────────────────────────────────
    print(f"\n[5/5] Saving fine-tuned weights → {LORA_PATH}")
    save_finetuned_weights(model.flow_lm, LORA_PATH)
    volume.commit()

    html = _build_html(all_samples)
    print("  ✅  Done!  Deploy with:  modal deploy pocket_tts_modal.py")
    return html.encode("utf-8")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MERGE LoRA → FULL SAFETENSORS                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@app.function(
    image=image,
    gpu="L4",
    volumes={VOL: volume},
    secrets=[hf_secret],
    timeout=1800,
)
def merge_model():
    """Merge LoRA + fine-tuned weights into the base model and save."""
    import torch
    import safetensors.torch
    import os

    volume.reload()

    if not os.path.exists(f"{LORA_PATH}/finetuned_weights.safetensors"):
        raise FileNotFoundError(
            f"No fine-tuned weights at {LORA_PATH}. Train first!"
        )

    print("Loading base model …")
    from pocket_tts import TTSModel
    model = TTSModel.load_model()
    model = model.to("cuda")

    # Reconstruct LoRA structure
    print("Applying LoRA structure …")
    apply_lora(
        model.flow_lm,
        target_names=["in_proj", "out_proj", "linear1", "linear2"],
        r=16,
        alpha=16,
    )

    # Load fine-tuned weights
    print("Loading fine-tuned weights …")
    load_finetuned_weights(model.flow_lm, LORA_PATH)

    # Merge LoRA into base weights
    print("Merging LoRA …")
    merge_lora(model.flow_lm)

    # Also need to copy the fine-tuned flow_net, out_eos, speaker_proj_weight
    # (they were loaded by load_finetuned_weights above and are already merged
    #  since they were full-finetuned, not LoRA)

    # Save full model
    print(f"Saving merged model → {MERGED_PATH}")
    os.makedirs(MERGED_PATH, exist_ok=True)
    safetensors.torch.save_file(
        model.state_dict(),
        f"{MERGED_PATH}/tts_merged.safetensors",
    )
    volume.commit()

    files = os.listdir(MERGED_PATH)
    total_mb = sum(
        os.path.getsize(os.path.join(MERGED_PATH, f))
        for f in files
    ) / 1024 / 1024
    print(f"✅  Merged model saved ({total_mb:.0f} MB, {len(files)} files)")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  INFERENCE ENDPOINT                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@app.cls(
    image=image,
    gpu="L4",
    volumes={VOL: volume},
    secrets=[hf_secret],
    scaledown_window=300,
)
class PocketTTSInference:
    """Serves the fine-tuned Pocket-TTS model as a web endpoint."""

    @modal.enter()
    def startup(self):
        import torch
        import os

        volume.reload()

        if not os.path.exists(f"{LORA_PATH}/finetuned_weights.safetensors"):
            raise FileNotFoundError(
                f"No fine-tuned model at {LORA_PATH}. "
                "Train first: modal run pocket_tts_modal.py"
            )

        print("Loading Pocket-TTS + fine-tuned weights …")
        from pocket_tts import TTSModel

        self.model = TTSModel.load_model()

        # Reconstruct LoRA and load weights
        apply_lora(
            self.model.flow_lm,
            target_names=["in_proj", "out_proj", "linear1", "linear2"],
            r=16, alpha=16,
        )
        load_finetuned_weights(self.model.flow_lm, LORA_PATH)

        # pocket-tts is designed for CPU inference and runs well there
        # But we keep it on GPU since we have one for the endpoint
        self.model.eval()
        print("✅  Model loaded — ready for requests")

        # Pre-load default voice state
        self.default_voice_state = self.model.get_state_for_audio_prompt("alba")
        print("✅  Default voice state cached")

    @modal.fastapi_endpoint(method="POST")
    def speak(self, request: dict):
        """Generate speech from text.

        JSON body:
            text  – string to speak (required)
            voice – voice name or path (default: "alba")

        Returns: audio/wav
        """
        import torch
        import soundfile as sf
        import io
        from fastapi.responses import Response

        text = request.get("text", "Hello, this is a test.")
        voice = request.get("voice", "alba")

        print(f'Generating: "{text[:60]}…"  voice={voice}')

        if voice == "alba":
            voice_state = self.default_voice_state
        else:
            voice_state = self.model.get_state_for_audio_prompt(voice)

        with torch.no_grad():
            audio = self.model.generate_audio(
                voice_state, text, copy_state=True
            )

        audio_np = audio.numpy()
        buf = io.BytesIO()
        sf.write(buf, audio_np, self.model.sample_rate, format="WAV")
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="speech.wav"'
            },
        )

    @modal.method()
    def generate(self, text: str, voice: str = "alba") -> bytes:
        """Generate speech and return raw WAV bytes."""
        import torch
        import soundfile as sf
        import io

        if voice == "alba":
            voice_state = self.default_voice_state
        else:
            voice_state = self.model.get_state_for_audio_prompt(voice)

        with torch.no_grad():
            audio = self.model.generate_audio(
                voice_state, text, copy_state=True
            )

        audio_np = audio.numpy()
        buf = io.BytesIO()
        sf.write(buf, audio_np, self.model.sample_rate, format="WAV")
        buf.seek(0)
        return buf.read()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CLI ENTRYPOINT                                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@app.local_entrypoint()
def main(
    action: str = "train",
    max_steps: int = 1200,
    smoke_test: bool = False,
    text: str = "Hello! This is the fine-tuned Pocket TTS model speaking.",
):
    """
    --action train        Fine-tune Pocket-TTS  (default)
    --action merge        Merge LoRA → full safetensors
    --action test         Generate a test WAV file
    --smoke-test          Quick 8-example / 2-step run
    """
    if action == "train":
        if smoke_test:
            print("🔬  SMOKE TEST – 8 examples, 2 training steps")
        else:
            print(f"🚀  Starting Pocket-TTS fine-tuning ({max_steps} steps) …")
        html_bytes = train.remote(max_steps=max_steps, smoke_test=smoke_test)
        print("\n✅  Training complete!")
        if html_bytes:
            out = "training_samples.html"
            with open(out, "wb") as f:
                f.write(html_bytes)
            print(f"    Samples:  open {out}")
        print("    Deploy:   modal deploy pocket_tts_modal.py")
        print("    Merge:    modal run pocket_tts_modal.py --action merge")

    elif action == "merge":
        print("🔀  Merging LoRA into base model …")
        merge_model.remote()
        print("\n✅  Merged model on volume.")

    elif action == "test":
        print(f'🔊  Generating speech: "{text}"')
        wav_bytes = PocketTTSInference().generate.remote(text)
        with open("test_output.wav", "wb") as f:
            f.write(wav_bytes)
        print("✅  Saved → test_output.wav")

    else:
        raise SystemExit(
            f"Unknown action '{action}'.  Use 'train', 'merge', or 'test'."
        )
