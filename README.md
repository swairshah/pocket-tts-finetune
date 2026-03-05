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
```

Main entrypoint: `train_modal.py`
Core modules: `pocket_tts_finetune/`
