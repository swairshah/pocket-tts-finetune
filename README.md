# Pocket-TTS Finetune (Modal)

Modal pipeline for fine-tuning and serving Kyutai Pocket-TTS.

## Quick start

```bash
# smoke test
modal run train_modal.py --smoke-test

# train
modal run train_modal.py --max-steps 600

# save voice preset
modal run train_modal.py --action save-voice --voice elise

# deploy endpoint
modal deploy train_modal.py

# side-by-side comparison (base vs cleaned vs fine-tuned)
modal run train_modal.py --action compare --voice elise
```

Main entrypoint: `train_modal.py`  
Core modules: `pocket_tts_finetune/`

Kyutai has no official fine-tuning code everything here is reverse-engineered
from inference code. The model (100M params) was distilled from a 300M teacher
via LSD (Lagrangian Self-Distillation) for 1-step generation. This tight
distillation makes it fast but leaves very little slack for weight adaptation.

### What works

- **Voice presets** (zero-shot cloning via KV cache export) — this is the primary
  customization path Kyutai designed. Clone any voice from ~5s of audio, export
  as `.safetensors`, and use it at inference. No weight changes needed.
- **Light backbone LoRA** (frozen flow-net, r=32, lr=1e-4) — trains without
  collapse and produces valid speech, but the perceptible difference from base
  model is minimal in practice.

I haven't found cases where LoRA improves on the voice presets. 

### What doesn't work

- **Unfreezing the flow-net** causes catastrophic collapse within 100–300 steps,
  even with 1,195 training examples. The flow-net is a 6-layer MLP (~3M params)
  that was distilled to operate in a single-step regime (s=0, t=1). Fine-tuning
  it breaks this delicate balance — loss drops to near-zero but inference
  produces monotone drones or silence. Likely cause: the flow-net's weights
  encode the entire distillation mapping from noise to audio in one step; even
  small perturbations destroy that mapping.
- **Conservative LoRA** (r=16, lr=5e-5) — loss decreases but output is
  identical to baseline. Too small to affect generation.
- **Standard CFM training** (random t, interpolated x_t) — wrong regime for
  this distilled model. Must use s=0, t=1 with pure noise input.

### More fine-tuning experiments can help

Voice presets control **who** speaks. Weights control **how** the model speaks —
pronunciation, prosody, text-handling logic. Cases where touching weights
should genuinely help:

1. **Pronunciation of domain terms** — the model consistently mispronounces
   technical vocabulary (e.g., "Kubernetes", "PyTorch", medical/legal terms).
   No voice preset change fixes this.
2. **New language phonemes** — sounds that don't exist in the English training
   data (Hindi nasals, Mandarin tones, Arabic pharyngeals). The model was
   trained on ~88k hours of English-only audio.
3. **Prosody/style not in training data** — sportscaster energy, ASMR whisper,
   bedtime story cadence. Voice presets carry timbre, not style.
4. **Systematic text-handling bugs** — repeated words, wrong expansion of
   abbreviations, bracket/symbol reading behavior.

## TODO: experiments

- [ ] **Unfreeze the text conditioner** — this maps text tokens to internal
      representations. If pronunciation is the target, this is where the signal
      lives. Test on a known mispronunciation and measure before/after with
      Whisper ASR.
- [ ] **Unfreeze `speaker_proj_weight`** — controls how voice conditioning
      enters the backbone. May improve voice cloning fidelity for voices that
      currently sound flat or quiet.
- [ ] **Targeted loss** — instead of flow-matching over all frames, penalize
      only specific failure patterns (e.g., repeated tokens, mispronounced
      segments). Requires alignment between text and audio frames.
- [ ] **Larger LoRA or full unfreeze of backbone** — with enough data, try r=64
      or full fine-tune of transformer layers (not flow-net) to see if there's a
      capacity threshold where behavior visibly changes.
- [ ] **Checkpoint-based accent/language experiments** — run controlled matrix
      of step counts (100/200/400/800) with different unfrozen components, score
      each with Whisper transcription accuracy and speaker similarity.
- [ ] **Compare with a non-distilled TTS model** — to confirm whether the
      adaptation difficulty is inherent to distilled models or to this
      architecture specifically.
- [ ] **Train on much more data** — our experiments used ~1,200 examples vs the
      model's 88,000 hours of pretraining. A few thousand examples may be
      noise-level signal. Try 10k–50k examples to see if backbone LoRA starts
      producing audible differences.

