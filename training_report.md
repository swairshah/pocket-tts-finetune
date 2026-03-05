# Pocket-TTS Fine-tuning: Training Report

**Model:** Kyutai Pocket-TTS (100M params)  
**Dataset:** MrDragonFox/Elise (1,195 examples, ~2.5 hours)  
**Infrastructure:** Modal cloud, NVIDIA L4 GPU  
**Date:** March 2026  

---

## 1. Goal

Fine-tune Pocket-TTS to speak in the voice of "Elise" from the MrDragonFox/Elise HuggingFace dataset. The pipeline should mirror our earlier CSM-1B fine-tuning work (`../csm-finetune/`) — single-file Modal deployment with training, merge, inference, and HTML sample evaluation.

## 2. Why Pocket-TTS Is Different

Pocket-TTS is architecturally unlike most TTS models. Where CSM-1B uses discrete audio tokens and cross-entropy loss (like a language model), Pocket-TTS uses **continuous 32-dimensional latents** and **flow matching** (predicting a velocity field to transform noise into audio). This meant we couldn't reuse any of the CSM training logic — everything had to be built from scratch by reverse-engineering the inference code.

Key architecture:
- **Mimi codec**: SEANet encoder/decoder + transformer. Encodes audio to 512-dim latents at 12.5 Hz, with a 32-dim bottleneck via `DummyQuantizer`
- **FlowLM backbone**: 6-layer causal transformer (d=1024, 16 heads) that processes text + voice conditioning + shifted audio latents
- **Flow MLP head** (`flow_net`): `SimpleMLPAdaLN` with 6 ResBlocks (d=512) that takes backbone output + noise and predicts velocity to clean audio
- **LSD (Lagrangian Self-Distillation)**: The model was distilled to generate in 1 step — `flow_net(s=0, t=1, x=noise)` directly predicts the full velocity

## 3. The Journey: What Broke and How We Fixed It

### 3.1 No Training Code Exists

Kyutai released inference-only code. There is no official training script, no loss function, no data pipeline. Everything was reverse-engineered from:
- The model architecture in `flow_lm.py`, `tts_model.py`, `mimi.py`
- The `StatefulModule` streaming system
- The config YAML (`b6369a24.yaml`)
- The inference flow in `lsd_decode()` and `generate_audio()`

### 3.2 PEFT/Unsloth Won't Work

**Problem:** Pocket-TTS uses a custom `StatefulModule` pattern where every module carries KV cache state via a special `model_state` dict. PEFT's wrapper breaks this by wrapping modules in a way that loses attribute access.

**Fix:** We implemented LoRA manually — a `LoRALinear` class that wraps `nn.Linear` with low-rank A/B matrices while proxying `.weight`, `.bias`, `.in_features`, `.out_features` to the original. This preserves the module tree structure that `StatefulModule` relies on.

### 3.3 The 512→32 Dimension Mismatch

**Problem:** Mimi's encoder outputs 512-dim latents, but FlowLM trains on 32-dim latents. The `DummyQuantizer` has a learned projection from 32→512, but no explicit reverse mapping exists in the codebase.

**Fix:** Compute the pseudo-inverse of the quantizer's output projection weight:
```python
W = mimi.quantizer.output_proj.weight.squeeze(-1)  # [512, 32]
W_pinv = torch.linalg.pinv(W)                       # [32, 512]
```
This gives the optimal least-squares 32-dim encoding for any 512-dim Mimi output.

### 3.4 KV Cache Required (model_state=None Crashes)

**Problem:** The transformer's forward pass in newer pocket-tts versions requires a `model_state` dict for KV cache — passing `None` caused crashes.

**Fix:** Use `init_states()` to create a properly-sized KV cache for the full sequence length, then process the entire sequence in one shot. This is mathematically equivalent to standard causal attention:
```python
fwd_state = init_states(model.flow_lm, batch_size=1, sequence_length=seq_len)
transformer_out = model.flow_lm.transformer(full_input, model_state=fwd_state)
```

### 3.5 TokenizedText Float Tensor Error

**Problem:** After moving model to GPU and back for sample generation, the tokenized text tensors sometimes ended up as Float instead of Long, causing `embedding lookup requires integer indices` errors.

**Fix:** Added explicit `.long()` cast:
```python
prepared = TokenizedText(prepared.tokens.long())
```

### 3.6 Critical Flow Matching Bug #1: Wrong Time Parameters

**Problem:** We initially used `s=t_flat, t=t_flat` — passing the same random time value for both the source and target time embeddings in the flow net. This doesn't match how the model was trained or how it runs at inference.

**Understanding the fix:** Pocket-TTS uses LSD (1-step distillation). At inference, `lsd_decode()` calls `flow_net(s=0, t=1, x=noise)` — always from time 0 to time 1 in a single step. The model was *distilled* to work in this regime. Standard CFM with random `t ∈ (0,1)` doesn't apply here.

**Fix:** Always use `s=0, t=1`:
```python
s_time = torch.zeros(B * T, 1, device=device)  # always 0
t_time = torch.ones(B * T, 1, device=device)   # always 1
```

### 3.7 Critical Flow Matching Bug #2: Wrong Input to Flow Net

**Problem:** We were feeding the interpolated `x_t = (1-t)*x_0 + t*x_1` to the flow net. This is correct for standard CFM, but wrong for the distilled 1-step model.

**Understanding the fix:** Since `s=0, t=1` always, the flow net expects the *starting point* of the flow — which is pure noise `x_0 ~ N(0,I)`. The model predicts the full velocity `x_1 - x_0` in one step.

**Fix:** Feed pure noise directly:
```python
x_0 = torch.randn(B, T, ldim, device=device)  # pure noise
pred = model.flow_lm.flow_net(backbone_out, s=0, t=1, x=x_0)
velocity_target = x_1 - x_0
loss = MSE(pred, velocity_target)
```

### 3.8 Flow Net Catastrophic Overfitting (8 Examples)

**Problem:** Early smoke tests used 8 cached examples. Unfreezing the flow_net caused it to memorize these completely — generated audio became a monotone drone with 0% silence, 5.7 dB dynamic range (vs healthy speech at 27-64 dB).

**Diagnosis:** The flow_net's 6-layer MLP with ~3M params easily memorized 8 training points. The flow matching loss dropped to near-zero but the model produced garbage at inference.

**Initial fix:** Freeze flow_net entirely. The backbone LoRA adapts instead. Added stale cache detection to reject caches with <20 examples for full runs.

### 3.9 Voice Cloning Model Not Available

**Problem:** HuggingFace hosts two Pocket-TTS model variants:
- `kyutai/pocket-tts` — full model with voice cloning (requires license acceptance)
- `kyutai/pocket-tts-without-voice-cloning` — `speaker_proj_weight` is **zeroed** (all zeros)

Without the license, `TTSModel.load_model()` silently downloads the restricted model. The `speaker_proj_weight` (shape [1024, 512]) is all zeros, which means custom audio conditioning produces near-silence (RMS ≈ 0.005).

The predefined voices (alba, marius, etc.) work because they use pre-computed `.safetensors` KV caches that bypass `speaker_proj_weight` entirely.

**Fix:**
1. Accept the HuggingFace license at `huggingface.co/kyutai/pocket-tts`
2. Add HF token to Modal: `modal secret create my-huggingface-secret HF_TOKEN=hf_xxx --force`
3. Login in the training function: `huggingface_hub.login(token=os.environ["HF_TOKEN"])`
4. The full model has non-zero `speaker_proj_weight` (std=0.0115, range [-0.076, 0.117])

## 4. Training Experiments

### 4.1 Experiment 1: Conservative Settings (Frozen flow_net, r=16, lr=5e-5)

| Setting | Value |
|---------|-------|
| LoRA rank | 16 |
| Learning rate | 5e-5 |
| Flow net | Frozen |
| Trainable params | 2.1M (1.8%) |
| Steps | 1,200 |
| Time | 70s on L4 |

**Result:** Speech remained intelligible (Whisper transcribed correctly at all checkpoints), but **nothing changed**. Baseline Elise voice and Step 1200 Elise voice were identical — same RMS (~0.025), same transcriptions, same characteristics. The FM loss dropped from 0.24 to 0.002, but this had zero effect on inference output.

**Diagnosis:** The LoRA was too small and the learning rate too low. The backbone barely moved. Training "succeeded" on paper (loss went down) but the changes were too small to affect generation.

### 4.2 Experiment 2: Aggressive Settings (Unfrozen flow_net, r=32, lr=2e-4)

| Setting | Value |
|---------|-------|
| LoRA rank | 32 |
| Learning rate | 2e-4 |
| Flow net | **Unfrozen** |
| Trainable params | 13.4M (11.2%) |
| Steps | 1,200 |
| Time | 173s on L4 |

**Result:** Complete model collapse by step 300. Whisper transcriptions:
- Step 0: ✅ "The quick brown fox jumps over the lazy dog."
- Step 300: ❌ "Blood quality"
- Step 600: ❌ "My nightmare mommy boom."
- Step 900: ❌ "Brrrrrrrrrrrrrrrrrrrkk"
- Step 1200: ❌ (empty — no detectable speech)

**Diagnosis:** The flow_net is catastrophically sensitive to fine-tuning. Even with 1,195 examples (not just 8), lr=2e-4 destroys it within 100-300 steps. The FM loss dropped to 0.0001, indicating severe overfitting.

### 4.3 Experiment 3: Middle Ground (Unfrozen flow_net, r=32, lr=1e-4)

Same as Experiment 2 but with halved learning rate (1e-4) and 600 steps.

**Result:** Same collapse pattern — garbage by step 100. "Live For är", empty transcriptions. The flow_net simply cannot be fine-tuned at any reasonable learning rate.

**Conclusion:** The flow_net is off-limits. It must stay frozen.

### 4.4 Experiment 4: Sweet Spot (Frozen flow_net, r=32, lr=1e-4) ✅

| Setting | Value |
|---------|-------|
| LoRA rank | 32 |
| Learning rate | 1e-4 |
| Flow net | **Frozen** |
| Trainable params | 3.7M (3.1%) |
| Steps | 600 |
| Time | 121s on L4 |
| Sample frequency | Every 100 steps |

**Result: Best outcome.** Speech stayed intelligible at every checkpoint, and the voice actually changed:

| Checkpoint | RMS | Whisper Transcription (prompt 1) |
|------------|-----|------|
| Baseline (alba) | 0.107 | "The quick brown fox jumps over the lazy dog." ✅ |
| Baseline (elise) | 0.017 | "Quick brown fox jumps over the lazy dog." ✅ |
| Step 100 | 0.013 | "A quick brown fox jumps over the lazy dog." ✅ |
| Step 200 | 0.012 | "A quick brown fox jumps over the lazy dog." ✅ |
| Step 300 | 0.022 | "The quick brown fox jumps over the lazy dog." ✅ |
| Step 400 | 0.029 | "A quick brown fox jumps over the lazy dog." ✅ |
| Step 500 | 0.024 | "Click from Fox jumps over the lazy dog." ⚠️ |
| **Step 600** | **0.037** | **"The quick brown fox jumps over the lazy dog."** ✅ |

Key observations:
- **Volume nearly tripled** from baseline (RMS 0.014 → 0.042 average across prompts)
- **Speech remained intelligible** at every checkpoint — Whisper got correct transcriptions throughout
- **One minor glitch** at step 500 ("Click from Fox" instead of "quick brown fox") but step 600 recovered
- The model found a better operating point for the Elise voice conditioning

## 5. Summary of What Works and What Doesn't

### ✅ What Works

| Technique | Why |
|-----------|-----|
| **Manual LoRA (not PEFT)** | Preserves StatefulModule attribute access patterns |
| **Pseudo-inverse for 512→32** | Optimal linear projection from Mimi encoder space to FlowLM latent space |
| **LSD 1-step flow matching (s=0, t=1)** | Matches the distilled model's inference regime exactly |
| **Pure noise input to flow_net** | Matches inference: model expects x_0 ~ N(0,I), not interpolated x_t |
| **Frozen flow_net + backbone LoRA** | Flow_net is catastrophically sensitive; backbone LoRA is sufficient |
| **LoRA rank 32 with lr=1e-4** | Sweet spot: enough capacity to change behavior without collapse |
| **HuggingFace auth for full model** | Non-VC model has zeroed speaker_proj_weight — voice cloning requires the gated model |
| **Frequent checkpointing + Whisper ASR** | Objectively measures speech quality without manual listening |

### ❌ What Doesn't Work

| Technique | Why |
|-----------|-----|
| **Unfreezing flow_net** | Catastrophic overfitting within 100-300 steps, even with 1195 examples |
| **Conservative lr=5e-5 with r=16** | Too small — training loss decreases but inference output doesn't change |
| **Standard CFM with random t ∈ (0,1)** | Wrong regime — model was distilled for 1-step (s=0, t=1) |
| **Interpolated x_t input** | Wrong — distilled model expects pure noise, not OT interpolation |
| **PEFT/Unsloth wrapping** | Breaks StatefulModule system |
| **Non-VC model for custom voices** | speaker_proj_weight is zeroed → custom audio conditioning produces silence |

## 6. Final Configuration

```
Model:            kyutai/pocket-tts (full, with voice cloning)
GPU:              NVIDIA L4 (24 GB) — peak usage 0.66 GB
LoRA rank:        32
LoRA alpha:       32
LoRA targets:     in_proj, out_proj, linear1, linear2
Learning rate:    1e-4 (linear decay with 5-step warmup)
Weight decay:     0.01
Gradient accum:   8
Max gradient norm: 1.0
Noise samples:    4 per frame
EOS weight:       1.0
EOS tail frames:  3
Voice prompt:     25 frames (~2 seconds)
Flow net:         FROZEN
Trainable params: 3.7M / 119.4M (3.1%)
Training time:    ~2 minutes for 600 steps
Cost:             ~$0.03 (2 min × $0.80/hr)
```

## 7. Voice Presets vs Fine-Tuning

An important realization during this project: **fine-tuning and voice identity are separate concerns**.

### How Built-in Voices Work

Pocket-TTS ships with 8 predefined voices (alba, marius, javert, etc.). Each is a `.safetensors` file containing a **pre-computed KV cache** — the model state after processing a reference audio clip. At inference, you load this state and generate from it. No model weights are changed.

```
alba.safetensors (5.9 MB) → KV cache state → generate_audio(state, text)
```

### What Fine-Tuning Does (and Doesn't Do)

LoRA fine-tuning modifies the model's **global weights**. This means:
- ✅ It changes how the model processes ALL voices (including alba)
- ❌ It does NOT add a selectable "Elise" voice to the voice roster
- ⚠️ It biases the model toward Elise's speech patterns for every voice

This is the wrong tool if you just want Elise as an additional voice option.

### The Right Approach: Save a Voice Preset

We added a `save-voice` action that does what the built-in voices do:

```bash
# Save Elise's voice as a reusable preset (no fine-tuning!)
modal run pocket_tts_modal.py --action save-voice --voice elise
```

This:
1. Loads the voice-cloning model (requires HF license)
2. Encodes the Elise reference audio through Mimi + speaker_proj
3. Runs the transformer to build a KV cache state
4. Exports it as `/vol/voices/elise.safetensors` (5.9 MB)

At inference, the endpoint loads all saved voice presets at startup:
```bash
# Use Elise voice
curl -X POST <endpoint> -d '{"text": "Hello!", "voice": "elise"}'

# Use built-in alba voice  
curl -X POST <endpoint> -d '{"text": "Hello!", "voice": "alba"}'
```

The voice preset approach is:
- **Additive** — doesn't change model weights, so other voices are unaffected
- **Instant** — no training needed, just encode the reference audio
- **Reusable** — saved state loads in milliseconds vs re-encoding audio each time

### When Fine-Tuning Still Makes Sense

Fine-tuning the backbone (LoRA) can improve how well the model handles a specific voice's characteristics — pronunciation quirks, speaking rhythm, tonal patterns. But it should be used **on top of** voice presets, not instead of them. And it affects all voices globally, so you'd want to evaluate whether other voices degrade.

## 8. Remaining Issues

1. **Elise voice is quieter than alba.** The voice-cloned output is about 3-5× quieter than the built-in alba voice. Training improved this (RMS went from 0.014 to 0.042) but it's still below alba's 0.12. This may be inherent to how voice cloning works in this model, or the reference clip volume may matter.

2. **Sibilant artifacts.** "She sells sea shells" consistently transcribes as "C cells C shells" with the Elise voice. This is a phonetic edge case that the model struggles with for this particular voice conditioning.

3. **Step 500 instability.** One sample degraded at step 500 ("Click from Fox") but recovered at step 600. This suggests the training is near the edge of stability. Going beyond 600 steps might eventually degrade.

## 10. Architecture Diagram

```
Training data flow:
─────────────────
Raw Audio (24kHz)
    │
    ▼  Mimi encoder (frozen)
512-dim latents [1, 512, T]
    │
    ├─→ First 25 frames → F.linear(·, speaker_proj_weight) → 1024-dim voice cond
    │
    └─→ Remaining frames → pseudo-inverse → 32-dim latents → normalize
                                                │
                                                ▼
    [voice_cond | text_embed | shifted_latents] → Transformer (LoRA) → backbone_out
                                                                            │
                                                                            ▼
                    noise x_0 ~ N(0,I) → flow_net(backbone_out, s=0, t=1, x_0) → predicted velocity
                                                                            │
                                                                            ▼
                                                        MSE(predicted, x_1 - x_0) = flow matching loss
                                                                     +
                                                        BCE(eos_logits, eos_labels) = EOS loss
```

## 11. Files

| File | Description |
|------|-------------|
| `pocket_tts_modal.py` | Complete Modal app (~1300 lines): training + voice saving + merge + inference + CLI |
| `training_samples.html` | Self-contained HTML with embedded audio at each checkpoint |
| `training_report.md` | This report |
| `REPORT.md` | Original project design document |
| `elise_test.wav` | Test sample generated with the saved Elise voice preset |

**On Modal volume (`/vol/`):**

| Path | Description |
|------|-------------|
| `/vol/voices/elise.safetensors` | Reusable Elise voice preset (5.9 MB KV cache) |
| `/vol/dataset_pocket/encoded.pt` | Pre-encoded dataset (1195 examples) |
| `/vol/dataset_pocket/elise_reference.wav` | Reference audio clip used for voice cloning |
| `/vol/pocket_lora/finetuned_weights.safetensors` | LoRA + fine-tuned weights |

## 12. How to Reproduce

```bash
# 0. Set up HuggingFace access
#    - Accept license at huggingface.co/kyutai/pocket-tts
#    - Get token from huggingface.co/settings/tokens
modal secret create my-huggingface-secret HF_TOKEN=hf_your_token --force

# 1. Save Elise as a reusable voice preset (no training needed!)
modal run pocket_tts_modal.py --action save-voice --voice elise

# 2. Deploy inference endpoint
modal deploy pocket_tts_modal.py

# 3. Generate speech with Elise's voice
curl -X POST <endpoint_url> \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, nice to meet you!", "voice": "elise"}' \
  --output speech.wav

# 4. (Optional) Fine-tune to improve Elise-specific speech patterns
modal run pocket_tts_modal.py --max-steps 600

# 5. (Optional) Smoke test first
modal run pocket_tts_modal.py --smoke-test
```
