from collections import OrderedDict

from .lora import apply_lora, load_finetuned_weights, merge_lora, save_finetuned_weights
from .sampling import TEST_PROMPTS, build_html, generate_samples


def login_hf_if_available(verbose=True):
    import os

    import huggingface_hub

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)
        if verbose:
            print("  ✅  Logged into HuggingFace (voice cloning enabled)")
        return True
    if verbose:
        print("  ⚠  No HF_TOKEN — voice cloning may not be available")
    return False


def train_workflow(
    max_steps: int,
    smoke_test: bool,
    paths,
    volume,
    dataset_name: str = "MrDragonFox/Elise",
    speaker_gender: str | None = None,
    translit_file: str | None = None,
):
    import os
    import time as time_module

    import torch
    import torch.nn.functional as F
    from datasets import Audio, ClassLabel, load_dataset
    from pocket_tts import TTSModel
    from pocket_tts.conditioners.base import TokenizedText

    def _safe_tag(text: str) -> str:
        return (
            text.lower()
            .replace("/", "_")
            .replace("-", "_")
            .replace(" ", "_")
            .replace(":", "_")
        )

    def _maybe_filter_speaker(ds):
        if not speaker_gender:
            return ds
        if "gender" not in ds.column_names:
            print(f"  ⚠  speaker_gender={speaker_gender} requested but dataset has no 'gender' column")
            return ds

        feat = ds.features.get("gender")
        target = speaker_gender.lower()

        if isinstance(feat, ClassLabel):
            name_to_id = {n.lower(): i for i, n in enumerate(feat.names)}
            if target in name_to_id:
                gid = name_to_id[target]
                ds = ds.filter(lambda ex: ex["gender"] == gid)
            else:
                print(f"  ⚠  gender='{speaker_gender}' not found in labels {feat.names}; skipping filter")
        else:
            ds = ds.filter(lambda ex: str(ex["gender"]).lower() == target)

        print(f"  Filtered by gender='{speaker_gender}' → {len(ds)} examples")
        return ds

    if smoke_test:
        max_steps = 2

    tag = "SMOKE TEST" if smoke_test else "Full"
    print("=" * 60)
    print(f"  Pocket-TTS  ·  LoRA Fine-tuning  ·  {tag}")
    print("=" * 60)
    print(f"  Dataset: {dataset_name}")
    if speaker_gender:
        print(f"  Speaker filter: gender={speaker_gender}")
    if translit_file:
        print(f"  Translit file: {translit_file}")

    login_hf_if_available(verbose=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    print("\n[2/5] Applying LoRA (r=16) to transformer …")

    for p in model.parameters():
        p.requires_grad = False

    replaced = apply_lora(
        model.flow_lm,
        target_names=["in_proj", "out_proj", "linear1", "linear2"],
        r=32,
        alpha=32,
    )
    print(f"  LoRA applied to {len(replaced)} layers")

    for p in model.flow_lm.out_eos.parameters():
        p.requires_grad = True

    if hasattr(model.flow_lm, "speaker_proj_weight"):
        model.flow_lm.speaker_proj_weight.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    print("\n[3/5] Preparing dataset …")

    SMOKE_N = 8
    ds_tag = "_smoke" if smoke_test else ""
    dataset_tag = _safe_tag(dataset_name)
    if speaker_gender:
        dataset_tag += f"_{_safe_tag(speaker_gender)}"
    if translit_file:
        dataset_tag += "_roman"
    ds_path = f"{paths.dataset_path}_{dataset_tag}{ds_tag}"
    encoded_file = f"{ds_path}/encoded.pt"

    if os.path.exists(encoded_file):
        cache = torch.load(encoded_file, weights_only=False)
        cached_n = len(cache["latents"])
        meta = cache.get("meta", {})
        cache_dataset = meta.get("dataset_name")
        cache_gender = meta.get("speaker_gender")
        cache_translit = meta.get("translit_file")

        cache_mismatch = (
            (cache_dataset != dataset_name)
            or (cache_gender != speaker_gender)
            or (cache_translit != translit_file)
        )

        if cache_mismatch:
            print("  ⚠  Cache metadata mismatch — re-encoding dataset")
            print(f"     cache dataset={cache_dataset}, gender={cache_gender}, translit={cache_translit}")
            print(f"     run   dataset={dataset_name}, gender={speaker_gender}, translit={translit_file}")
            os.remove(encoded_file)
            cache = None
        elif not smoke_test and cached_n < 20:
            print(f"  ⚠  Stale cache has only {cached_n} examples — re-encoding full dataset")
            os.remove(encoded_file)
            cache = None
        else:
            print(f"  ✅  Cached encoded dataset found ({cached_n} examples)")
    else:
        cache = None

    if cache is not None:
        latent_cache = cache["latents"]
        enc512_cache = cache["enc512"]
        text_cache = cache["texts"]

        ref_path = f"{ds_path}/reference.wav"
        if not os.path.exists(ref_path):
            print("  ⬇  Downloading reference clip …")
            raw_ds = load_dataset(dataset_name, split="train")
            raw_ds = _maybe_filter_speaker(raw_ds)
            raw_ds = raw_ds.cast_column("audio", Audio(sampling_rate=24000))
            import soundfile as sf_save

            for ex in raw_ds:
                arr = torch.tensor(ex["audio"]["array"], dtype=torch.float32)
                dur = arr.shape[0] / 24000
                if 5.0 <= dur <= 10.0:
                    sf_save.write(ref_path, arr.numpy(), 24000)
                    print(f"  Saved reference voice clip: {ref_path} ({dur:.1f}s)")
                    break
            else:
                arr = torch.tensor(raw_ds[0]["audio"]["array"], dtype=torch.float32)
                sf_save.write(ref_path, arr.numpy(), 24000)
                print(f"  Saved reference voice clip (fallback): {ref_path}")
            volume.commit()
    else:
        print(f"  ⬇  Downloading & encoding {dataset_name} …")
        raw_ds = load_dataset(dataset_name, split="train")
        raw_ds = _maybe_filter_speaker(raw_ds)

        if smoke_test:
            raw_ds = raw_ds.select(range(min(SMOKE_N, len(raw_ds))))
            print(f"  🔬  Smoke test: {len(raw_ds)} examples")

        translit_lines = None
        if translit_file:
            if not os.path.exists(translit_file):
                raise FileNotFoundError(f"Translit file not found: {translit_file}")
            with open(translit_file, "r", encoding="utf-8") as f:
                translit_lines = [x.rstrip("\n") for x in f]
            print(f"  Loaded transliterations: {len(translit_lines)} lines")
            if len(translit_lines) < len(raw_ds):
                print(
                    f"  ⚠  translit lines ({len(translit_lines)}) < dataset rows ({len(raw_ds)}); "
                    "remaining rows fallback to original text"
                )

        raw_ds = raw_ds.cast_column("audio", Audio(sampling_rate=24000))

        W = model.mimi.quantizer.output_proj.weight.squeeze(-1)
        W_pinv = torch.linalg.pinv(W)
        print(f"  Quantizer weight: {W.shape} → pseudo-inverse: {W_pinv.shape}")

        latent_cache = []
        enc512_cache = []
        text_cache = []
        skipped = 0

        best_ref_audio = None
        best_ref_len = 0

        model.mimi.eval()
        for i, example in enumerate(raw_ds):
            if translit_lines is not None and i < len(translit_lines):
                text_value = translit_lines[i]
            else:
                text_value = example.get("text", example.get("transcription", ""))
            if not isinstance(text_value, str) or not text_value.strip():
                skipped += 1
                continue

            audio_array = torch.tensor(example["audio"]["array"], dtype=torch.float32)

            if audio_array.shape[0] < 36000:
                skipped += 1
                continue

            dur_s = audio_array.shape[0] / 24000
            if 5.0 <= dur_s <= 10.0 and dur_s > best_ref_len:
                best_ref_audio = audio_array.clone()
                best_ref_len = dur_s

            max_samples = 30 * 24000
            if audio_array.shape[0] > max_samples:
                audio_array = audio_array[:max_samples]

            audio_tensor = audio_array.unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                enc_512 = model.mimi.encode_to_latent(audio_tensor)

            with torch.no_grad():
                latents_32 = F.conv1d(enc_512, W_pinv.unsqueeze(-1).to(enc_512.device))

            enc512_cache.append(enc_512.squeeze(0).transpose(0, 1).cpu())
            latent_cache.append(latents_32.squeeze(0).transpose(0, 1).cpu())
            text_cache.append(text_value)

            if (i + 1) % 100 == 0:
                print(f"    Encoded {i+1}/{len(raw_ds)} …")

        if best_ref_audio is None:
            best_ref_audio = torch.tensor(raw_ds[0]["audio"]["array"], dtype=torch.float32)
            best_ref_len = best_ref_audio.shape[0] / 24000

        print(f"  Encoded {len(latent_cache)} examples (skipped {skipped})")

        os.makedirs(ds_path, exist_ok=True)
        import soundfile as sf_save

        ref_path = f"{ds_path}/reference.wav"
        sf_save.write(ref_path, best_ref_audio.numpy(), 24000)
        print(f"  Saved reference voice clip: {ref_path} ({best_ref_len:.1f}s)")
        torch.save(
            {
                "latents": latent_cache,
                "enc512": enc512_cache,
                "texts": text_cache,
                "meta": {
                    "dataset_name": dataset_name,
                    "speaker_gender": speaker_gender,
                    "translit_file": translit_file,
                },
            },
            encoded_file,
        )
        volume.commit()

    print(f"  Dataset: {len(latent_cache)} examples")
    avg_frames = sum(l.shape[0] for l in latent_cache) / len(latent_cache)
    print(f"  Avg length: {avg_frames:.0f} frames ({avg_frames/12.5:.1f}s)")
    print(f"  Latent dim: {latent_cache[0].shape[-1]}  (should be 32)")
    print(f"  Encoder dim: {enc512_cache[0].shape[-1]}  (should be 512)")

    ref_voice = f"{ds_path}/reference.wav"
    if os.path.exists(ref_voice):
        eval_voice = ref_voice
        print(f"  Voice for samples: dataset reference ({ref_voice})")
    else:
        eval_voice = "alba"
        print(f"  ⚠  No dataset reference found, using alba voice")

    PROMPT_FRAMES = 25
    N_NOISE = 4
    GRAD_ACCUM = 8
    EOS_WEIGHT = 1.0
    EOS_TAIL = 3

    sample_every = max(1, max_steps // 6)
    print(
        f"\n[4/5] Training for {max_steps} steps  "
        f"(samples every {sample_every}, grad_accum={GRAD_ACCUM})"
    )

    if device == "cuda":
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gpu_props.name}  ({round(gpu_props.total_memory / 1024**3, 1)} GB)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )

    warmup_steps = 5

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, 1.0 - (step - warmup_steps) / max(max_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    all_samples = OrderedDict()

    print("\n  🔊  Generating baseline samples (step 0) …")
    print("    --- Alba voice (built-in reference) ---")
    all_samples["Step 0 — Baseline (alba)"] = generate_samples(model, TEST_PROMPTS, voice="alba")
    if eval_voice != "alba":
        print("    --- Elise voice (target) ---")
        all_samples["Step 0 — Baseline (elise)"] = generate_samples(
            model, TEST_PROMPTS, voice=eval_voice
        )

    model.flow_lm.train()
    model.mimi.eval()
    optimizer.zero_grad()

    t_start = time_module.time()

    for step in range(max_steps):
        idx = step % len(latent_cache)
        raw_latents_32 = latent_cache[idx].to(device)
        raw_enc_512 = enc512_cache[idx].to(device)
        text = text_cache[idx]

        T_total = raw_latents_32.shape[0]

        if T_total > PROMPT_FRAMES + 10:
            prompt_enc512 = raw_enc_512[:PROMPT_FRAMES].unsqueeze(0)
            target_latents = raw_latents_32[PROMPT_FRAMES:].unsqueeze(0)
        else:
            prompt_enc512 = None
            target_latents = raw_latents_32.unsqueeze(0)

        B, T, ldim = target_latents.shape

        if prompt_enc512 is not None:
            audio_cond = F.linear(
                prompt_enc512.to(model.flow_lm.speaker_proj_weight.dtype),
                model.flow_lm.speaker_proj_weight,
            )
        else:
            audio_cond = torch.empty(
                1,
                0,
                model.flow_lm.dim,
                device=device,
                dtype=model.flow_lm.speaker_proj_weight.dtype,
            )

        emb_mean = model.flow_lm.emb_mean
        emb_std = model.flow_lm.emb_std
        target_norm = (target_latents - emb_mean) / emb_std

        prepared = model.flow_lm.conditioner.prepare(text)
        prepared = TokenizedText(prepared.tokens.long())
        text_emb = model.flow_lm.conditioner(prepared)
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(0)

        cond_prefix = torch.cat([audio_cond, text_emb], dim=1)

        bos = torch.full((B, 1, ldim), float("nan"), device=device)
        shifted = torch.cat([bos, target_norm[:, :-1]], dim=1)
        bos_emb = model.flow_lm.bos_emb
        shifted = torch.where(
            torch.isnan(shifted),
            bos_emb.view(1, 1, -1).expand_as(shifted),
            shifted,
        )
        input_ = model.flow_lm.input_linear(shifted)

        full_input = torch.cat([cond_prefix, input_], dim=1)

        from pocket_tts.modules.stateful_module import init_states as _init_states

        seq_len = full_input.shape[1]
        fwd_state = _init_states(model.flow_lm, batch_size=B, sequence_length=seq_len)
        for k, v in fwd_state.items():
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                    fwd_state[k][kk] = vv.to(device)
        transformer_out = model.flow_lm.transformer(full_input, model_state=fwd_state)
        transformer_out = model.flow_lm.out_norm(transformer_out)
        backbone_out = transformer_out[:, -T:]
        backbone_out = backbone_out.to(torch.float32)

        bb_flat = backbone_out.reshape(-1, backbone_out.shape[-1])

        loss_fm = torch.tensor(0.0, device=device)
        s_time = torch.zeros(B * T, 1, device=device)
        t_time = torch.ones(B * T, 1, device=device)

        for _ in range(N_NOISE):
            x_0 = torch.randn(B, T, ldim, device=device)
            x_1 = target_norm
            velocity_target = x_1 - x_0

            x0_flat = x_0.reshape(-1, ldim)

            pred = model.flow_lm.flow_net(bb_flat, s=s_time, t=t_time, x=x0_flat)
            pred = pred.reshape(B, T, ldim)
            loss_fm = loss_fm + F.mse_loss(pred, velocity_target)

        loss_fm = loss_fm / N_NOISE

        eos_logits = model.flow_lm.out_eos(backbone_out).squeeze(-1)
        eos_labels = torch.zeros(B, T, device=device)
        eos_labels[:, -EOS_TAIL:] = 1.0
        loss_eos = F.binary_cross_entropy_with_logits(eos_logits, eos_labels)

        loss = (loss_fm + EOS_WEIGHT * loss_eos) / GRAD_ACCUM
        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0 or step == max_steps - 1:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

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

        if (step + 1) % sample_every == 0 and step > 0:
            print(f"\n  🔊  Generating samples at step {step + 1} …")
            all_samples[f"Step {step + 1}"] = generate_samples(model, TEST_PROMPTS, voice=eval_voice)
            model.flow_lm.train()

    secs = time_module.time() - t_start
    peak = round(torch.cuda.max_memory_reserved() / 1024**3, 2) if device == "cuda" else 0
    print(f"\n  ✅  Training done in {secs:.0f}s ({secs/60:.1f} min)")
    print(f"  Peak GPU memory: {peak} GB")

    final_key = f"Step {max_steps}"
    if final_key not in all_samples:
        print(f"\n  🔊  Generating final samples …")
        all_samples[f"Step {max_steps} — Final (elise)"] = generate_samples(
            model, TEST_PROMPTS, voice=eval_voice
        )
        print("    --- Alba voice (comparison) ---")
        all_samples[f"Step {max_steps} — Final (alba)"] = generate_samples(
            model, TEST_PROMPTS, voice="alba"
        )

    print(f"\n[5/5] Saving fine-tuned weights → {paths.lora_path}")
    save_finetuned_weights(model.flow_lm, paths.lora_path)
    volume.commit()

    html = build_html(all_samples)
    print("  ✅  Done!  Deploy with:  modal deploy pocket_tts_modal.py")
    return html.encode("utf-8")


def merge_workflow(paths, volume):
    import os

    import safetensors.torch
    from pocket_tts import TTSModel

    volume.reload()

    if not os.path.exists(f"{paths.lora_path}/finetuned_weights.safetensors"):
        raise FileNotFoundError(f"No fine-tuned weights at {paths.lora_path}. Train first!")

    print("Loading base model …")
    model = TTSModel.load_model()
    model = model.to("cuda")

    print("Applying LoRA structure …")
    apply_lora(
        model.flow_lm,
        target_names=["in_proj", "out_proj", "linear1", "linear2"],
        r=16,
        alpha=16,
    )

    print("Loading fine-tuned weights …")
    load_finetuned_weights(model.flow_lm, paths.lora_path)

    print("Merging LoRA …")
    merge_lora(model.flow_lm)

    print(f"Saving merged model → {paths.merged_path}")
    os.makedirs(paths.merged_path, exist_ok=True)
    safetensors.torch.save_file(model.state_dict(), f"{paths.merged_path}/tts_merged.safetensors")
    volume.commit()

    files = os.listdir(paths.merged_path)
    total_mb = sum(os.path.getsize(os.path.join(paths.merged_path, f)) for f in files) / 1024 / 1024
    print(f"✅  Merged model saved ({total_mb:.0f} MB, {len(files)} files)")


def save_voice_workflow(
    paths,
    volume,
    voice_name="elise",
    audio_url="",
    dataset_name: str = "MrDragonFox/Elise",
    speaker_gender: str | None = None,
):
    import io
    import os
    import urllib.request
    from pathlib import Path

    import numpy as np
    import soundfile as sf
    import torch
    from datasets import Audio, ClassLabel, load_dataset
    from pocket_tts import TTSModel
    from pocket_tts.models.tts_model import _import_model_state, export_model_state

    login_hf_if_available(verbose=False)

    print("Loading Pocket-TTS model …")
    model = TTSModel.load_model()
    print(f"  Voice cloning: {'✅ enabled' if model.has_voice_cloning else '❌ disabled'}")

    if not model.has_voice_cloning:
        raise RuntimeError(
            "Voice cloning not available! Accept the license at "
            "huggingface.co/kyutai/pocket-tts and set HF_TOKEN."
        )

    if audio_url:
        print(f"  Downloading reference audio from: {audio_url}")
        tmp_audio = f"/tmp/voice_ref_{voice_name}.wav"
        urllib.request.urlretrieve(audio_url, tmp_audio)
        ref_path = Path(tmp_audio)
    else:
        dataset_tag = (
            dataset_name.lower()
            .replace("/", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )
        if speaker_gender:
            dataset_tag += f"_{speaker_gender.lower()}"
        ref_dir = f"{paths.dataset_path}_{dataset_tag}"
        ref_path_str = f"{ref_dir}/reference.wav"

        if not os.path.exists(ref_path_str):
            print(f"  Downloading {dataset_name} to extract reference clip …")
            raw_ds = load_dataset(dataset_name, split="train")
            if speaker_gender and "gender" in raw_ds.column_names:
                feat = raw_ds.features.get("gender")
                target = speaker_gender.lower()
                if isinstance(feat, ClassLabel):
                    name_to_id = {n.lower(): i for i, n in enumerate(feat.names)}
                    if target in name_to_id:
                        raw_ds = raw_ds.filter(lambda ex: ex["gender"] == name_to_id[target])
                else:
                    raw_ds = raw_ds.filter(lambda ex: str(ex["gender"]).lower() == target)
                print(f"  Filtered by gender='{speaker_gender}' → {len(raw_ds)} examples")

            raw_ds = raw_ds.cast_column("audio", Audio(sampling_rate=24000))

            os.makedirs(ref_dir, exist_ok=True)
            for ex in raw_ds:
                arr = torch.tensor(ex["audio"]["array"], dtype=torch.float32)
                dur = arr.shape[0] / 24000
                if 5.0 <= dur <= 10.0:
                    sf.write(ref_path_str, arr.numpy(), 24000)
                    print(f"  Saved reference: {ref_path_str} ({dur:.1f}s)")
                    break
            else:
                arr = torch.tensor(raw_ds[0]["audio"]["array"], dtype=torch.float32)
                sf.write(ref_path_str, arr.numpy(), 24000)

            volume.commit()

        ref_path = Path(ref_path_str)

    print(f"  Reference audio: {ref_path}")
    audio_data, sr = sf.read(str(ref_path))
    print(f"  Duration: {len(audio_data)/sr:.1f}s  Sample rate: {sr} Hz")

    print(f"\nEncoding voice state for '{voice_name}' …")
    voice_state = model.get_state_for_audio_prompt(ref_path)

    total_bytes = 0
    for _, module_state in voice_state.items():
        for _, tensor in module_state.items():
            if isinstance(tensor, torch.Tensor):
                total_bytes += tensor.numel() * tensor.element_size()
    print(f"  State size: {total_bytes / 1024 / 1024:.1f} MB")

    os.makedirs(paths.voices_path, exist_ok=True)
    out_path = f"{paths.voices_path}/{voice_name}.safetensors"
    export_model_state(voice_state, out_path)
    volume.commit()

    file_size = os.path.getsize(out_path)
    print(f"\n✅  Voice '{voice_name}' saved!")
    print(f"    Path: {out_path}")
    print(f"    Size: {file_size / 1024 / 1024:.1f} MB")

    print(f"\n🔊  Testing voice '{voice_name}' …")
    loaded_state = _import_model_state(out_path)

    test_text = "Hello! This is a test of the saved voice preset."
    with torch.no_grad():
        audio = model.generate_audio(loaded_state, test_text, copy_state=True)

    test_wav_path = f"{paths.voices_path}/{voice_name}_test.wav"
    sf.write(test_wav_path, audio.numpy(), model.sample_rate)
    volume.commit()

    rms = float(np.sqrt(np.mean(audio.numpy() ** 2)))
    dur = len(audio) / model.sample_rate
    print(f"  Generated: {dur:.1f}s, RMS={rms:.4f}")
    print(f"  Test WAV: {test_wav_path}")

    buf = io.BytesIO()
    sf.write(buf, audio.numpy(), model.sample_rate, format="WAV")
    return buf.getvalue()
