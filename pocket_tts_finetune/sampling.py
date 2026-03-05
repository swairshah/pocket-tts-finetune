from collections import OrderedDict


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


def build_voice_state(model, audio_path):
    """Build a voice state from an audio file."""
    import torch
    from pathlib import Path

    if model.has_voice_cloning:
        return model.get_state_for_audio_prompt(Path(audio_path))

    import soundfile as sf_read
    from pocket_tts.data.audio_utils import convert_audio
    from pocket_tts.modules.stateful_module import increment_steps, init_states

    audio, sr = sf_read.read(audio_path)
    audio_t = torch.tensor(audio, dtype=torch.float32)
    if audio_t.dim() == 1:
        audio_t = audio_t.unsqueeze(0)
    if sr != model.sample_rate:
        audio_t = convert_audio(audio_t, sr, model.sample_rate, 1)

    with torch.no_grad():
        conditioning = model._encode_audio(audio_t.unsqueeze(0))

    model_state = init_states(model.flow_lm, batch_size=1, sequence_length=conditioning.shape[1])
    model._run_flow_lm_and_increment_step(model_state=model_state, audio_conditioning=conditioning)

    for mn in model_state:
        for k in model_state[mn]:
            if isinstance(model_state[mn][k], torch.Tensor):
                model_state[mn][k] = model_state[mn][k].detach().clone()

    return model_state


def generate_samples(model, prompts, voice="alba"):
    """Generate audio for each prompt. Returns [(prompt, wav_bytes), …]."""
    import io
    import os

    import soundfile as sf
    import torch

    was_training = model.training
    original_device = model.device
    model.eval()
    model.to("cpu")

    if isinstance(voice, str) and os.path.exists(voice):
        print(f"      Using voice file: {os.path.basename(voice)}")
        voice_state = build_voice_state(model, voice)
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
            import numpy as np

            buf = io.BytesIO()
            sf.write(buf, np.zeros(4800, dtype=np.float32), 24000, format="WAV")
            samples.append((prompt, buf.getvalue()))
        print(f"      [{i+1}/{len(prompts)}] {prompt[:40]}…")

    model.to(original_device)
    if was_training:
        model.train()
    return samples


def build_html(all_samples: OrderedDict):
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


def build_multi_voice_demo_html(results):
    """Build self-contained HTML for multi-voice comparisons."""
    import base64
    from datetime import datetime

    rows = ""
    for voice, samples in results.items():
        cards = ""
        for prompt, wav_bytes, rms, dur in samples:
            b64 = base64.b64encode(wav_bytes).decode()
            cards += (
                '<div class="sample">'
                f'<p class="prompt">"{prompt}"</p>'
                f'<p class="meta">{dur:.1f}s &middot; RMS {rms:.3f}</p>'
                f'<audio controls preload="none" '
                f'src="data:audio/wav;base64,{b64}"></audio>'
                "</div>\n"
            )
        rows += f'<div class="voice-section"><h2>🎤 {voice}</h2>\n{cards}</div>\n'

    voices = len(results)
    prompts = len(next(iter(results.values()))) if results else 0

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Pocket-TTS Multi-Voice Demo</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px;
         margin: 0 auto; padding: 24px; background: #fafafa; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
  .info {{ color: #666; font-size: 0.9em; margin-bottom: 24px; }}
  .voice-section {{ background: #fff; border: 1px solid #ddd;
                    border-radius: 10px; padding: 20px; margin: 20px 0;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .voice-section h2 {{ margin-top: 0; color: #c42; }}
  .sample {{ margin: 14px 0; }}
  .prompt {{ margin: 4px 0; color: #333; font-style: italic; }}
  .meta {{ margin: 2px 0; color: #999; font-size: 0.85em; }}
  audio {{ width: 100%; height: 36px; }}
</style></head><body>
<h1>🗣️ Pocket-TTS Multi-Voice Demo</h1>
<p class="info">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
 &mdash; {voices} voices &times; {prompts} prompts
 &mdash; Model: Pocket-TTS (100M)</p>
{rows}
</body></html>"""
