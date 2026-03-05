import os
from pathlib import Path

from .sampling import build_multi_voice_demo_html


def load_model_with_optional_lora(paths):
    from pocket_tts import TTSModel

    from .lora import apply_lora, load_finetuned_weights
    from .workflows import login_hf_if_available

    login_hf_if_available(verbose=False)

    model = TTSModel.load_model()

    if os.path.exists(f"{paths.lora_path}/finetuned_weights.safetensors"):
        print("  Loading LoRA fine-tuned weights …")
        apply_lora(
            model.flow_lm,
            target_names=["in_proj", "out_proj", "linear1", "linear2"],
            r=32,
            alpha=32,
        )
        load_finetuned_weights(model.flow_lm, paths.lora_path)
        print("  ✅  LoRA weights loaded")
    else:
        print("  ℹ  No LoRA weights found — using base model")

    model.eval()
    return model


def load_voice_states(model, voices_path):
    voice_states = {}
    voice_states["alba"] = model.get_state_for_audio_prompt("alba")
    print("  Loaded voice: alba (built-in)")

    if os.path.isdir(voices_path):
        from pocket_tts.models.tts_model import _import_model_state

        for sf_file in sorted(Path(voices_path).glob("*.safetensors")):
            name = sf_file.stem
            if name.endswith("_test"):
                continue
            voice_states[name] = _import_model_state(sf_file)
            print(f"  Loaded voice: {name} (custom)")

    print(f"  Available voices: {list(voice_states.keys())}")
    return voice_states


def resolve_voice_state(model, voice_states, voices_path, voice: str):
    if voice in voice_states:
        return voice_states[voice]

    sf_path = f"{voices_path}/{voice}.safetensors"
    if os.path.exists(sf_path):
        from pocket_tts.models.tts_model import _import_model_state

        state = _import_model_state(sf_path)
        voice_states[voice] = state
        return state

    return model.get_state_for_audio_prompt(voice)


def run_multi_voice_demo(model, get_voice_state, voices, prompts):
    import io

    import numpy as np
    import soundfile as sf
    import torch

    results = {}

    for voice in voices:
        print(f"\n  🎤 Voice: {voice}")
        voice_state = get_voice_state(voice)
        results[voice] = []

        for i, prompt in enumerate(prompts):
            try:
                with torch.no_grad():
                    audio = model.generate_audio(voice_state, prompt, copy_state=True)
                audio_np = audio.numpy()
                rms = float(np.sqrt(np.mean(audio_np**2)))
                dur = len(audio_np) / model.sample_rate
                buf = io.BytesIO()
                sf.write(buf, audio_np, model.sample_rate, format="WAV")
                results[voice].append((prompt, buf.getvalue(), rms, dur))
                print(f"    [{i+1}/{len(prompts)}] {dur:.1f}s rms={rms:.3f}  {prompt[:50]}…")
            except Exception as e:
                print(f"    [{i+1}/{len(prompts)}] ⚠ FAILED: {e}")
                buf = io.BytesIO()
                sf.write(buf, np.zeros(4800, dtype=np.float32), 24000, format="WAV")
                results[voice].append((prompt, buf.getvalue(), 0.0, 0.2))

    return build_multi_voice_demo_html(results).encode("utf-8")
