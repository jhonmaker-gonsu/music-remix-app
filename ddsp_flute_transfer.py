#!/usr/bin/env python3
"""
ddsp_flute_transfer.py — DDSP / DDSP_VST で歌声をリード楽器化
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, sosfiltfilt


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


SAMPLE_RATE = 16000
VST_FRAME_SIZE = 1024
VST_FRAME_RATE = 50
VST_HOP_SIZE = int(SAMPLE_RATE / VST_FRAME_RATE)
DEFAULT_VST_PLUGIN_CANDIDATES = [
    Path.home() / "Library/Audio/Plug-Ins/Components/DDSP Effect.component/Contents/MacOS/DDSP Effect",
    Path.home() / "Library/Audio/Plug-Ins/VST3/DDSP Effect.vst3/Contents/MacOS/DDSP Effect",
]
VST_MODEL_INDEX = {
    "bassoon": 2,
    "clarinet": 3,
    "flute": 4,
    "saxophone": 5,
    "sax": 5,
    "trombone": 6,
    "tuba": 10,
    "violin": 11,
}

ddsp = None
gin = None
librosa = None
tf = None
detect_notes = None
fit_quantile_transform = None
Autoencoder = None


def load_runtime() -> None:
    global ddsp, gin, librosa, tf, detect_notes, fit_quantile_transform, Autoencoder
    try:
        import ddsp as _ddsp  # noqa: PLC0415
        import gin as _gin  # noqa: PLC0415
        import librosa as _librosa  # noqa: PLC0415
        import tensorflow.compat.v2 as _tf  # noqa: PLC0415
        from ddsp.training.postprocessing import detect_notes as _detect_notes  # noqa: PLC0415
        from ddsp.training.postprocessing import fit_quantile_transform as _fit_quantile_transform  # noqa: PLC0415
        from ddsp.training.models import Autoencoder as _Autoencoder  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "DDSP実行環境が未準備です。先に ddsp_setup.py を通すか、"
            "GUIの「DDSPフルート化」から初回セットアップを実行してください。"
        ) from exc

    ddsp = _ddsp
    gin = _gin
    librosa = _librosa
    tf = _tf
    detect_notes = _detect_notes
    fit_quantile_transform = _fit_quantile_transform
    Autoencoder = _Autoencoder


def squeeze(x: np.ndarray) -> np.ndarray:
    return np.squeeze(x) if np.ndim(x) > 1 else x


def get_tuning_factor(f0_midi: np.ndarray, f0_confidence: np.ndarray, mask_on: np.ndarray) -> float:
    tuning_factors = np.linspace(-0.5, 0.5, 101)
    midi_diffs = (f0_midi[mask_on][:, np.newaxis] - tuning_factors[np.newaxis, :]) % 1.0
    midi_diffs[midi_diffs > 0.5] -= 1.0
    weights = f0_confidence[mask_on][:, np.newaxis]
    cost_diffs = np.mean(weights * np.abs(midi_diffs), axis=0)
    f0_at = f0_midi[mask_on][:, np.newaxis] - midi_diffs
    deltas = (np.diff(f0_at, axis=0) != 0.0).astype(float)
    cost_deltas = np.mean(weights[:-1] * deltas, axis=0)
    norm = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-7)
    return tuning_factors[np.argmin(norm(cost_deltas) + norm(cost_diffs))]


def auto_tune(f0_midi: np.ndarray, tuning_factor: float, mask_on: np.ndarray, amount: float = 0.0) -> np.ndarray:
    major_scale = np.ravel([np.array([0, 2, 4, 5, 7, 9, 11]) + 12 * i for i in range(10)])
    all_scales = np.stack([major_scale + i for i in range(12)])
    f0_on = f0_midi[mask_on] - tuning_factor
    f0_diff_tsn = f0_on[:, np.newaxis, np.newaxis] - all_scales[np.newaxis, :, :]
    f0_diff_ts = np.min(np.abs(f0_diff_tsn), axis=-1)
    scale_idx = np.argmin(np.mean(f0_diff_ts, axis=0))
    f0_diff_tn = f0_midi[:, np.newaxis] - all_scales[scale_idx][np.newaxis, :]
    note_idx = np.argmin(np.abs(f0_diff_tn), axis=-1)
    midi_diff = np.take_along_axis(f0_diff_tn, note_idx[:, np.newaxis], axis=-1)[:, 0]
    return f0_midi - amount * midi_diff


def shift_ld(audio_features: dict[str, np.ndarray], ld_shift: float = 0.0) -> dict[str, np.ndarray]:
    audio_features["loudness_db"] += ld_shift
    return audio_features


def shift_f0(audio_features: dict[str, np.ndarray], pitch_shift: float = 0.0) -> dict[str, np.ndarray]:
    audio_features["f0_hz"] *= 2.0 ** float(pitch_shift)
    audio_features["f0_hz"] = np.clip(audio_features["f0_hz"], 0.0, librosa.midi_to_hz(110.0))
    return audio_features


def load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    return audio.astype(np.float32)


def latest_checkpoint(model_dir: Path) -> str:
    ckpt = tf.train.latest_checkpoint(str(model_dir))
    if ckpt:
        return ckpt
    ckpt_file = model_dir / "checkpoint"
    if ckpt_file.exists():
        text = ckpt_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("model_checkpoint_path:"):
                name = line.split('"')[1]
                return str(model_dir / name)
    index_files = sorted(model_dir.glob("ckpt-*.index"))
    if index_files:
        latest = max(index_files, key=lambda p: int(p.stem.split("-")[-1]))
        return str(latest.with_suffix(""))
    raise FileNotFoundError(f"checkpoint が見つかりません: {model_dir}")


def compute_audio_features(audio: np.ndarray, frame_rate: int) -> dict[str, np.ndarray]:
    mono = squeeze(audio)
    hop_length = max(1, int(round(SAMPLE_RATE / float(frame_rate))))
    frame_length = 2048
    loudness_db = ddsp.spectral_ops.compute_loudness(mono, SAMPLE_RATE, frame_rate)
    loudness_db = loudness_db.numpy() if hasattr(loudness_db, "numpy") else np.asarray(loudness_db)
    loudness_db = squeeze(np.asarray(loudness_db)).astype(np.float32)

    f0_hz, _, voiced_prob = librosa.pyin(
        mono,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=SAMPLE_RATE,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )
    if voiced_prob is None:
        voiced_prob = np.zeros_like(f0_hz, dtype=np.float32)
    f0_hz = np.nan_to_num(f0_hz, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    f0_confidence = np.nan_to_num(voiced_prob, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    length = min(len(loudness_db), len(f0_hz), len(f0_confidence))
    return {
        "audio": audio,
        "loudness_db": loudness_db[:length],
        "f0_hz": f0_hz[:length],
        "f0_confidence": f0_confidence[:length],
    }


def find_ddsp_vst_plugin(plugin_binary: str | None = None) -> Path | None:
    if plugin_binary:
        path = Path(plugin_binary).expanduser().resolve()
        return path if path.exists() else None
    for candidate in DEFAULT_VST_PLUGIN_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def find_all_offsets(blob: bytes, needle: bytes) -> list[int]:
    offsets: list[int] = []
    start = 0
    while True:
        index = blob.find(needle, start)
        if index == -1:
            return offsets
        offsets.append(index)
        start = index + 1


def read_embedded_metadata(blob: bytes, offset: int) -> dict[str, float] | None:
    eocd_offset = blob.find(b"PK\x05\x06", offset)
    if eocd_offset == -1:
        return None

    for extra_bytes in range(22, 22 + 2048):
        archive = BytesIO(blob[offset : eocd_offset + extra_bytes])
        if not zipfile.is_zipfile(archive):
            continue
        with zipfile.ZipFile(archive) as zf:
            return json.loads(zf.read("metadata.json").decode("utf-8"))
    return None


def extract_vst_bundle(plugin_binary: Path) -> dict[str, object]:
    blob = plugin_binary.read_bytes()
    zip_offsets = find_all_offsets(blob, b"PK\x03\x04")
    model_offsets = [offset - 4 for offset in find_all_offsets(blob, b"TFL3")]
    metadata = [read_embedded_metadata(blob, offset) for offset in zip_offsets]
    return {
        "blob": blob,
        "zip_offsets": zip_offsets,
        "model_offsets": model_offsets,
        "metadata": metadata,
    }


def extract_vst_model_blob(bundle: dict[str, object], model_index: int) -> bytes:
    blob = bundle["blob"]
    model_offsets = bundle["model_offsets"]
    zip_offsets = bundle["zip_offsets"]
    model_offset = model_offsets[model_index]
    next_zip = next((offset for offset in zip_offsets if offset > model_offset), len(blob))
    return blob[model_offset:next_zip]


def normalize_vst_model_name(model_name: str) -> str:
    key = model_name.strip().lower()
    if key not in VST_MODEL_INDEX:
        choices = ", ".join(sorted({name.title() for name in VST_MODEL_INDEX}))
        raise ValueError(f"未知の VST モデルです: {model_name} (選択肢: {choices})")
    return key


def load_vst_interpreters(plugin_binary: Path, model_name: str) -> tuple[object, object, dict[str, float] | None]:
    bundle = extract_vst_bundle(plugin_binary)
    pitch_blob = extract_vst_model_blob(bundle, 0)
    model_key = normalize_vst_model_name(model_name)
    synth_blob = extract_vst_model_blob(bundle, VST_MODEL_INDEX[model_key])
    metadata = bundle["metadata"][VST_MODEL_INDEX[model_key]]

    pitch_interpreter = tf.lite.Interpreter(model_content=pitch_blob)
    pitch_interpreter.allocate_tensors()

    synth_interpreter = tf.lite.Interpreter(model_content=synth_blob)
    synth_interpreter.allocate_tensors()
    return pitch_interpreter, synth_interpreter, metadata


def run_pitch_model(pitch_interpreter: object, audio_frame: np.ndarray) -> tuple[float, float, float, float]:
    input_details = pitch_interpreter.get_input_details()
    output_details = pitch_interpreter.get_output_details()
    pitch_interpreter.set_tensor(input_details[0]["index"], np.asarray(audio_frame, dtype=np.float32))
    pitch_interpreter.invoke()
    outputs = [float(pitch_interpreter.get_tensor(detail["index"]).ravel()[0]) for detail in output_details]
    return outputs[0], outputs[1], outputs[2], outputs[3]


def round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return int(np.ceil(float(value) / float(multiple)) * multiple)


def round_down_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return int(np.floor(float(value) / float(multiple)) * multiple)


def repair_boolean_mask(mask: np.ndarray, max_gap_frames: int, min_run_frames: int) -> np.ndarray:
    fixed = np.asarray(mask, dtype=bool).copy()
    length = len(fixed)

    i = 0
    while i < length:
        if fixed[i]:
            i += 1
            continue
        start = i
        while i < length and not fixed[i]:
            i += 1
        if 0 < start < length and i < length and (i - start) <= max_gap_frames:
            fixed[start:i] = True

    i = 0
    while i < length:
        if not fixed[i]:
            i += 1
            continue
        start = i
        while i < length and fixed[i]:
            i += 1
        if (i - start) < min_run_frames:
            fixed[start:i] = False

    return fixed


def refine_f0_conditioning(
    f0_hz: np.ndarray,
    f0_confidence: np.ndarray,
    loudness_db: np.ndarray,
    frame_rate: int,
    min_confidence: float,
    activity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f0_hz = np.asarray(f0_hz, dtype=np.float32)
    f0_confidence = np.asarray(f0_confidence, dtype=np.float32)
    loudness_db = np.asarray(loudness_db, dtype=np.float32)

    n_frames = len(f0_hz)
    if n_frames == 0:
        zeros = np.zeros(0, dtype=np.float32)
        return zeros, zeros, np.zeros(0, dtype=bool), zeros

    loud_floor = float(np.percentile(loudness_db, 20))
    loud_peak = float(np.percentile(loudness_db, 95))
    loud_norm = np.clip((loudness_db - loud_floor) / max(loud_peak - loud_floor, 1.0), 0.0, 1.0)

    confidence_sigma = max(frame_rate * 0.010, 0.5)
    activity_sigma = max(frame_rate * 0.015, 0.5)
    confidence_smooth = gaussian_filter1d(f0_confidence, sigma=confidence_sigma, mode="nearest")
    activity = gaussian_filter1d((confidence_smooth ** 1.5) * loud_norm, sigma=activity_sigma, mode="nearest")

    mask_on = activity > activity_threshold
    mask_on = repair_boolean_mask(
        mask_on,
        max_gap_frames=max(1, int(round(frame_rate * 0.10))),
        min_run_frames=max(2, int(round(frame_rate * 0.03))),
    )
    valid = (f0_hz > 0.0) & ((confidence_smooth > min_confidence) | mask_on)

    f0_midi = librosa.hz_to_midi(np.maximum(f0_hz, 1e-6)).astype(np.float32)
    f0_midi[~valid] = np.nan

    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return (
            np.zeros_like(f0_hz, dtype=np.float32),
            confidence_smooth.astype(np.float32),
            mask_on,
            activity.astype(np.float32),
        )

    prev = f0_midi[valid_idx[0]]
    for idx in valid_idx[1:]:
        current = f0_midi[idx]
        candidates = np.array(
            [current - 24.0, current - 12.0, current, current + 12.0, current + 24.0],
            dtype=np.float32,
        )
        best = candidates[np.argmin(np.abs(candidates - prev))]
        if abs(best - prev) < abs(current - prev) - 2.5:
            f0_midi[idx] = best
        prev = f0_midi[idx]

    gap_limit = max(1, int(round(frame_rate * 0.10)))
    filled = f0_midi.copy()
    valid_idx = np.flatnonzero(~np.isnan(filled))
    for left, right in zip(valid_idx[:-1], valid_idx[1:]):
        gap = right - left - 1
        if 0 < gap <= gap_limit:
            step = (filled[right] - filled[left]) / float(gap + 1)
            for gap_idx in range(1, gap + 1):
                filled[left + gap_idx] = filled[left] + (step * gap_idx)

    valid_idx = np.flatnonzero(~np.isnan(filled))
    interpolated = np.interp(np.arange(n_frames), valid_idx, filled[valid_idx]).astype(np.float32)

    median_window = max(3, int(round(frame_rate * 0.020)))
    if median_window % 2 == 0:
        median_window += 1
    smoothed = median_filter(interpolated, size=median_window, mode="nearest")
    smoothed = gaussian_filter1d(smoothed, sigma=max(frame_rate * 0.008, 0.6), mode="nearest")

    blend = np.clip((confidence_smooth * 0.65) + (activity * 0.80), 0.0, 1.0)
    refined_midi = (interpolated * (1.0 - blend)) + (smoothed * blend)
    refined_f0_hz = librosa.midi_to_hz(refined_midi).astype(np.float32)
    refined_f0_hz[~mask_on] = 0.0

    return refined_f0_hz, confidence_smooth.astype(np.float32), mask_on, activity.astype(np.float32)


def apply_loudness_conditioning(
    loudness_db: np.ndarray,
    mask_on: np.ndarray,
    note_activity: np.ndarray,
    dataset_stats: dict[str, np.ndarray] | None,
    quiet_db: float,
) -> np.ndarray:
    loudness_db = np.asarray(loudness_db, dtype=np.float32)
    mask_on = np.asarray(mask_on, dtype=bool)
    note_activity = np.asarray(note_activity, dtype=np.float32)
    loudness_out = np.copy(loudness_db)

    if dataset_stats is not None and np.count_nonzero(mask_on) >= 8:
        _, loudness_norm = fit_quantile_transform(
            loudness_db,
            mask_on,
            inv_quantile=dataset_stats["quantile_transform"],
        )
        loudness_out = np.reshape(loudness_norm, loudness_db.shape).astype(np.float32)

    mask_off = np.logical_not(mask_on)
    loudness_out[mask_off] -= quiet_db * (1.0 - note_activity[mask_off])
    return loudness_out.astype(np.float32)


def condition_audio_features(
    audio_features: dict[str, np.ndarray],
    dataset_stats: dict[str, np.ndarray] | None,
    args: argparse.Namespace,
    frame_rate: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    audio_features_mod = {
        key: np.copy(value) if isinstance(value, np.ndarray) else value
        for key, value in audio_features.items()
    }

    refined_f0_hz, refined_confidence, mask_on, note_activity = refine_f0_conditioning(
        audio_features_mod["f0_hz"],
        audio_features_mod["f0_confidence"],
        audio_features_mod["loudness_db"],
        frame_rate=frame_rate,
        min_confidence=args.f0_min_confidence,
        activity_threshold=args.activity_threshold,
    )
    audio_features_mod["f0_hz"] = refined_f0_hz
    audio_features_mod["f0_confidence"] = refined_confidence

    if dataset_stats is not None and np.any(mask_on):
        target_mean_pitch = float(dataset_stats["mean_pitch"])
        pitch_midi = ddsp.core.hz_to_midi(np.maximum(audio_features_mod["f0_hz"], 1e-6))
        mean_pitch = float(np.mean(pitch_midi[mask_on]))
        octave_shift = int(np.clip(np.round((target_mean_pitch - mean_pitch) / 12.0), -1, 1))
        if octave_shift != 0:
            audio_features_mod = shift_f0(audio_features_mod, pitch_shift=float(octave_shift * 12.0))

    audio_features_mod["loudness_db"] = apply_loudness_conditioning(
        audio_features_mod["loudness_db"],
        mask_on=mask_on,
        note_activity=note_activity,
        dataset_stats=dataset_stats,
        quiet_db=args.quiet,
    )

    if args.autotune > 0 and np.any(mask_on):
        f0_midi = np.asarray(ddsp.core.hz_to_midi(np.maximum(audio_features_mod["f0_hz"], 1e-6)))
        tuning_factor = get_tuning_factor(f0_midi, audio_features_mod["f0_confidence"], mask_on)
        f0_midi_at = auto_tune(f0_midi, tuning_factor, mask_on, amount=args.autotune)
        tuned_hz = ddsp.core.midi_to_hz(f0_midi_at)
        audio_features_mod["f0_hz"] = np.where(mask_on, tuned_hz, 0.0).astype(np.float32)

    audio_features_mod = shift_ld(audio_features_mod, args.loudness_shift)
    audio_features_mod = shift_f0(audio_features_mod, args.pitch_shift)

    return audio_features_mod, note_activity


def build_sample_envelope(note_activity: np.ndarray, n_samples: int) -> np.ndarray:
    note_activity = np.asarray(note_activity, dtype=np.float32)
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(note_activity) == 0:
        return np.zeros(n_samples, dtype=np.float32)
    if len(note_activity) == 1:
        return np.full(n_samples, float(note_activity[0]), dtype=np.float32)

    frame_positions = np.linspace(0.0, n_samples - 1, num=len(note_activity), dtype=np.float64)
    sample_positions = np.arange(n_samples, dtype=np.float64)
    envelope = np.interp(sample_positions, frame_positions, note_activity).astype(np.float32)
    return np.clip(envelope, 0.0, 1.0)


def interpolate_sample_curve(values: np.ndarray, n_samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(values) == 0:
        return np.zeros(n_samples, dtype=np.float32)
    if len(values) == 1:
        return np.full(n_samples, float(values[0]), dtype=np.float32)

    frame_positions = np.linspace(0.0, n_samples - 1, num=len(values), dtype=np.float64)
    sample_positions = np.arange(n_samples, dtype=np.float64)
    return np.interp(sample_positions, frame_positions, values).astype(np.float32)


def synthesize_pitch_anchor(
    f0_hz: np.ndarray,
    envelope: np.ndarray,
    n_samples: int,
    base_mix: float,
    low_pitch_boost: np.ndarray | None = None,
) -> np.ndarray:
    if base_mix <= 0.0 or n_samples <= 0:
        return np.zeros(max(n_samples, 0), dtype=np.float32)

    sample_f0 = interpolate_sample_curve(f0_hz, n_samples)
    sample_env = interpolate_sample_curve(envelope, n_samples)
    if low_pitch_boost is None:
        boost = np.ones(n_samples, dtype=np.float32)
    else:
        boost = np.clip(interpolate_sample_curve(low_pitch_boost, n_samples), 0.0, 1.0)

    voiced = sample_f0 > 0.0
    phase = np.cumsum((2.0 * np.pi * sample_f0) / SAMPLE_RATE, dtype=np.float64)
    fundamental = np.sin(phase)
    second = np.sin(phase * 2.0)
    third = np.sin(phase * 3.0)

    anchor = (0.82 * fundamental) + (0.14 * second) + (0.04 * third)
    anchor *= sample_env * (base_mix * (0.45 + (0.55 * boost)))
    anchor[~voiced] = 0.0
    return anchor.astype(np.float32)


def active_rms(signal: np.ndarray, envelope: np.ndarray, threshold: float = 0.20) -> float:
    active = envelope > threshold
    if not np.any(active):
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal[active]), dtype=np.float64) + 1e-8))


def match_active_rms(signal: np.ndarray, reference: np.ndarray, envelope: np.ndarray, max_gain: float = 4.0) -> np.ndarray:
    signal_rms = active_rms(signal, envelope)
    reference_rms = active_rms(reference, envelope)
    if signal_rms <= 1e-6 or reference_rms <= 1e-6:
        return signal
    gain = min(max_gain, reference_rms / signal_rms)
    return (signal * gain).astype(np.float32)


def apply_filter(audio: np.ndarray, cutoff_hz: float, kind: str) -> np.ndarray:
    if cutoff_hz <= 0.0:
        return audio.astype(np.float32)
    sos = butter(2 if kind == "highpass" else 4, cutoff_hz, btype=kind, fs=SAMPLE_RATE, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def hz_to_vst_scaled(f0_hz: np.ndarray) -> np.ndarray:
    midi = librosa.hz_to_midi(np.maximum(np.asarray(f0_hz, dtype=np.float32), 1e-6))
    scaled = np.clip(midi / 127.0, 0.0, 1.0)
    scaled[np.asarray(f0_hz) <= 0.0] = 0.0
    return scaled.astype(np.float32)


def postprocess_output_audio(
    audio: np.ndarray,
    envelope: np.ndarray,
    args: argparse.Namespace,
    reference_signal: np.ndarray | None = None,
) -> np.ndarray:
    processed = np.asarray(audio, dtype=np.float32)

    if args.post_highpass > 0.0:
        processed = apply_filter(processed, args.post_highpass, "highpass")
    if 0.0 < args.post_lowpass < (SAMPLE_RATE / 2.0):
        processed = apply_filter(processed, args.post_lowpass, "lowpass")

    envelope = np.clip(envelope, 0.0, 1.0)
    processed *= (0.18 + (0.82 * envelope))

    if reference_signal is not None:
        processed = match_active_rms(processed, reference_signal, envelope, max_gain=8.0)

    peak = float(np.max(np.abs(processed)))
    if peak > 1e-5:
        processed = processed / peak

    if args.output_drive > 1.0:
        processed = np.tanh(processed * args.output_drive).astype(np.float32)

    return processed.astype(np.float32)


def smooth_vst_pitch(
    f0_hz: np.ndarray,
    f0_scaled: np.ndarray,
    pw_scaled: np.ndarray,
    power_db: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_on = (power_db > -58.0) & (pw_scaled > 0.45)
    mask_on = repair_boolean_mask(mask_on, max_gap_frames=3, min_run_frames=2)

    f0_hz = np.asarray(f0_hz, dtype=np.float32)
    f0_scaled = np.asarray(f0_scaled, dtype=np.float32)
    pw_scaled = np.asarray(pw_scaled, dtype=np.float32)
    power_db = np.asarray(power_db, dtype=np.float32)

    if not np.any(mask_on):
        zeros = np.zeros_like(f0_hz, dtype=np.float32)
        return zeros, zeros, zeros

    midi = librosa.hz_to_midi(np.maximum(f0_hz, 1e-6)).astype(np.float32)
    midi[~mask_on] = np.nan

    valid_idx = np.flatnonzero(~np.isnan(midi))
    filled = np.interp(np.arange(len(midi)), valid_idx, midi[valid_idx]).astype(np.float32)
    filled = median_filter(filled, size=5, mode="nearest")
    filled = gaussian_filter1d(filled, sigma=0.8, mode="nearest")

    smoothed_hz = librosa.midi_to_hz(filled).astype(np.float32)
    smoothed_hz[~mask_on] = 0.0

    smoothed_f0_scaled = gaussian_filter1d(f0_scaled, sigma=0.8, mode="nearest").astype(np.float32)
    smoothed_pw_scaled = gaussian_filter1d(pw_scaled, sigma=0.6, mode="nearest").astype(np.float32)
    smoothed_f0_scaled[~mask_on] = 0.0
    smoothed_pw_scaled[~mask_on] = 0.0
    return smoothed_hz, smoothed_f0_scaled, smoothed_pw_scaled


def compute_vst_conditioning(
    audio: np.ndarray,
    pitch_interpreter: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mono = squeeze(audio)
    if mono.ndim != 1:
        mono = np.mean(mono, axis=0)

    padded = np.pad(mono, (VST_FRAME_SIZE // 2, VST_FRAME_SIZE // 2), mode="reflect")
    n_frames = 1 + max(0, int(np.ceil(len(mono) / VST_HOP_SIZE)))

    f0_scaled = np.zeros(n_frames, dtype=np.float32)
    pw_scaled = np.zeros(n_frames, dtype=np.float32)
    f0_hz = np.zeros(n_frames, dtype=np.float32)
    power_db = np.zeros(n_frames, dtype=np.float32)

    for frame_index in range(n_frames):
        start = frame_index * VST_HOP_SIZE
        frame = padded[start : start + VST_FRAME_SIZE]
        if len(frame) < VST_FRAME_SIZE:
            frame = np.pad(frame, (0, VST_FRAME_SIZE - len(frame)))
        f0_scaled[frame_index], pw_scaled[frame_index], f0_hz[frame_index], power_db[frame_index] = run_pitch_model(
            pitch_interpreter,
            frame,
        )

    f0_hz, f0_scaled, pw_scaled = smooth_vst_pitch(f0_hz, f0_scaled, pw_scaled, power_db)
    return f0_hz, f0_scaled, pw_scaled, power_db


def adapt_vst_conditioning_pitch(
    f0_hz: np.ndarray,
    metadata: dict[str, float] | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    voiced = f0_hz > 0.0
    if metadata is None or not np.any(voiced):
        return f0_hz.astype(np.float32), hz_to_vst_scaled(f0_hz), 0.0

    meta_min_hz = float(metadata.get("mean_min_pitch_note_hz", 0.0))
    meta_max_hz = float(metadata.get("mean_max_pitch_note_hz", 0.0))
    if meta_min_hz <= 0.0 or meta_max_hz <= 0.0:
        return f0_hz.astype(np.float32), hz_to_vst_scaled(f0_hz), 0.0

    voiced_f0 = f0_hz[voiced]
    median_f0 = float(np.median(voiced_f0))
    target_floor = meta_min_hz * 1.02
    if median_f0 >= target_floor:
        return f0_hz.astype(np.float32), hz_to_vst_scaled(f0_hz), 0.0

    shift_semitones = float(np.clip(12.0 * np.log2(target_floor / max(median_f0, 1e-6)), 0.0, 12.0))
    conditioning_hz = (f0_hz * (2.0 ** (shift_semitones / 12.0))).astype(np.float32)
    conditioning_hz[voiced] = np.minimum(conditioning_hz[voiced], meta_max_hz * 0.98)
    conditioning_hz[~voiced] = 0.0
    conditioning_scaled = hz_to_vst_scaled(conditioning_hz)
    return conditioning_hz.astype(np.float32), conditioning_scaled.astype(np.float32), shift_semitones


def run_vst_synth(
    synth_interpreter: object,
    f0_scaled: np.ndarray,
    pw_scaled: np.ndarray,
    f0_hz: np.ndarray,
    noise_mix: float,
) -> np.ndarray:
    input_details = synth_interpreter.get_input_details()
    output_details = synth_interpreter.get_output_details()

    state = np.zeros(input_details[0]["shape"], dtype=np.float32)
    n_frames = len(f0_hz)
    amplitudes = np.zeros((1, n_frames, 1), dtype=np.float32)
    harmonic_distribution = np.zeros((1, n_frames, 60), dtype=np.float32)
    noise_magnitudes = np.zeros((1, n_frames, 65), dtype=np.float32)

    for frame_index in range(n_frames):
        synth_interpreter.set_tensor(input_details[0]["index"], state.astype(np.float32).reshape(input_details[0]["shape"]))
        synth_interpreter.set_tensor(input_details[1]["index"], np.array([f0_scaled[frame_index]], dtype=np.float32))
        synth_interpreter.set_tensor(input_details[2]["index"], np.array([pw_scaled[frame_index]], dtype=np.float32))
        synth_interpreter.invoke()

        outputs = [synth_interpreter.get_tensor(detail["index"]).ravel() for detail in output_details]
        harmonic_distribution[0, frame_index] = outputs[0]
        state = outputs[1]
        amplitudes[0, frame_index, 0] = float(outputs[2][0])
        noise_magnitudes[0, frame_index] = outputs[3]

    f0_hz = f0_hz.reshape(1, n_frames, 1).astype(np.float32)
    n_samples = n_frames * VST_HOP_SIZE

    harmonic = ddsp.synths.Harmonic(n_samples=n_samples, sample_rate=SAMPLE_RATE)
    filtered_noise = ddsp.synths.FilteredNoise(n_samples=n_samples)
    harmonic_audio = squeeze(np.asarray(harmonic.get_signal(**harmonic.get_controls(amplitudes, harmonic_distribution, f0_hz)))).astype(np.float32)
    noise_audio = squeeze(np.asarray(filtered_noise.get_signal(**filtered_noise.get_controls(noise_magnitudes)))).astype(np.float32)
    return (harmonic_audio + (noise_mix * noise_audio)).astype(np.float32)


def transfer_audio_vst(
    audio: np.ndarray,
    plugin_binary: Path,
    model_name: str,
    args: argparse.Namespace,
) -> np.ndarray:
    print(f"Using DDSP_VST backend: {plugin_binary.name} / {model_name}")
    pitch_interpreter, synth_interpreter, metadata = load_vst_interpreters(plugin_binary, model_name)
    if metadata is not None:
        print(f"VST model metadata: pitch {metadata.get('mean_min_pitch_note_hz', 0):.1f}-{metadata.get('mean_max_pitch_note_hz', 0):.1f} Hz")

    f0_hz, f0_scaled, pw_scaled, power_db = compute_vst_conditioning(audio, pitch_interpreter)
    conditioning_hz, conditioning_scaled, shift_semitones = adapt_vst_conditioning_pitch(f0_hz, metadata)
    if shift_semitones > 0.01:
        print(f"Adaptive conditioning shift: +{shift_semitones:.2f} st")
    instrument_audio = run_vst_synth(
        synth_interpreter,
        f0_scaled=conditioning_scaled,
        pw_scaled=pw_scaled,
        f0_hz=f0_hz,
        noise_mix=args.noise_mix,
    )

    target_samples = audio.shape[1]
    instrument_audio = instrument_audio[:target_samples]
    envelope = build_sample_envelope((f0_hz > 0.0).astype(np.float32), len(instrument_audio))
    meta_min_hz = float(metadata.get("mean_min_pitch_note_hz", 0.0)) if metadata else 0.0
    low_pitch_boost = np.ones_like(f0_hz, dtype=np.float32)
    if meta_min_hz > 0.0:
        low_pitch_boost = np.clip((meta_min_hz / np.maximum(f0_hz, 1e-6)) - 0.75, 0.0, 1.0).astype(np.float32)
        low_pitch_boost[f0_hz <= 0.0] = 0.0
    anchor = synthesize_pitch_anchor(
        f0_hz=f0_hz,
        envelope=(f0_hz > 0.0).astype(np.float32),
        n_samples=len(instrument_audio),
        base_mix=args.pitch_anchor_mix,
        low_pitch_boost=low_pitch_boost,
    )
    if len(anchor) == len(instrument_audio):
        instrument_audio = (instrument_audio + anchor).astype(np.float32)
    instrument_audio = postprocess_output_audio(
        instrument_audio,
        envelope=envelope,
        args=args,
        reference_signal=None,
    )
    peak = float(np.max(np.abs(instrument_audio)))
    if peak > 1e-5:
        instrument_audio = (0.85 * instrument_audio / peak).astype(np.float32)
    return instrument_audio.astype(np.float32)


def configure_model(gin_file: Path, n_samples: int, time_steps: int) -> Autoencoder:
    with gin.unlock_config():
        gin.clear_config()
        gin.parse_config_file(str(gin_file), skip_unknown=True)
        gin.parse_config(
            [
                f"Harmonic.n_samples = {n_samples}",
                f"FilteredNoise.n_samples = {n_samples}",
                f"F0LoudnessPreprocessor.time_steps = {time_steps}",
                "oscillator_bank.use_angular_cumsum = True",
            ]
        )

    model = Autoencoder()
    return model


def process_chunk(
    audio_chunk: np.ndarray,
    model: Autoencoder,
    dataset_stats: dict[str, np.ndarray] | None,
    args: argparse.Namespace,
    frame_rate: int,
    hop_size: int,
) -> np.ndarray:
    chunk_n_samples = audio_chunk.shape[1]
    time_steps = int(chunk_n_samples / hop_size)

    audio_features = compute_audio_features(audio_chunk, frame_rate=frame_rate)
    audio_features["audio"] = audio_features["audio"][:, :chunk_n_samples]
    for key in ("f0_hz", "f0_confidence", "loudness_db"):
        audio_features[key] = audio_features[key][:time_steps]

    audio_features_mod, note_activity = condition_audio_features(
        audio_features,
        dataset_stats=dataset_stats,
        args=args,
        frame_rate=frame_rate,
    )

    outputs = model(audio_features_mod, training=False)
    processor_outputs = model.processor_group.get_controls(outputs)

    dry_signal = squeeze(np.asarray(processor_outputs["add"]["signal"])).astype(np.float32)
    harmonic_signal = squeeze(np.asarray(processor_outputs["harmonic"]["signal"])).astype(np.float32)
    noise_signal = squeeze(np.asarray(processor_outputs["filtered_noise"]["signal"])).astype(np.float32)
    wet_signal = squeeze(np.asarray(processor_outputs["reverb"]["signal"])).astype(np.float32)
    wet_only = wet_signal - dry_signal

    instrument_audio = harmonic_signal + (args.noise_mix * noise_signal) + (args.reverb_mix * wet_only)
    envelope = build_sample_envelope(note_activity, len(instrument_audio))
    instrument_audio = match_active_rms(instrument_audio, dry_signal, envelope, max_gain=4.0)
    instrument_audio = postprocess_output_audio(
        instrument_audio,
        envelope,
        args,
        reference_signal=dry_signal,
    )
    return instrument_audio.astype(np.float32)


def transfer_audio(
    audio: np.ndarray,
    gin_file: Path,
    checkpoint: str,
    dataset_stats: dict[str, np.ndarray] | None,
    args: argparse.Namespace,
) -> np.ndarray:
    time_steps_train = gin.query_parameter("F0LoudnessPreprocessor.time_steps")
    n_samples_train = gin.query_parameter("Harmonic.n_samples")
    hop_size = int(n_samples_train / time_steps_train)
    frame_rate = int(round(SAMPLE_RATE / hop_size))

    audio_len = audio.shape[1]
    max_chunk_samples = round_up_to_multiple(int(round(args.chunk_seconds * SAMPLE_RATE)), hop_size)
    overlap_samples = round_down_to_multiple(int(round(args.chunk_overlap_seconds * SAMPLE_RATE)), hop_size)
    overlap_samples = max(0, min(overlap_samples, max_chunk_samples - hop_size))

    if audio_len <= max_chunk_samples:
        chunk_n_samples = max(hop_size, round_up_to_multiple(audio_len, hop_size))
    else:
        chunk_n_samples = max_chunk_samples

    step_samples = max(hop_size, chunk_n_samples - overlap_samples)
    time_steps = int(chunk_n_samples / hop_size)

    model = configure_model(gin_file, n_samples=chunk_n_samples, time_steps=time_steps)
    print("Loading model...")
    model.restore(checkpoint, verbose=False)

    output = np.zeros(audio_len, dtype=np.float32)
    weights = np.zeros(audio_len, dtype=np.float32)

    starts = list(range(0, audio_len, step_samples))
    if starts:
        last_start = max(0, audio_len - chunk_n_samples)
        starts = [start for start in starts if start <= last_start]
        if not starts or starts[-1] != last_start:
            starts.append(last_start)

    for index, start in enumerate(sorted(set(starts)), start=1):
        end = min(audio_len, start + chunk_n_samples)
        actual_len = end - start
        chunk = np.zeros((1, chunk_n_samples), dtype=np.float32)
        chunk[:, :actual_len] = audio[:, start:end]

        print(f"Processing chunk {index}/{len(set(starts))} ({start / SAMPLE_RATE:.1f}s - {end / SAMPLE_RATE:.1f}s)")
        chunk_out = process_chunk(
            chunk,
            model=model,
            dataset_stats=dataset_stats,
            args=args,
            frame_rate=frame_rate,
            hop_size=hop_size,
        )[:actual_len]

        weight = np.ones(actual_len, dtype=np.float32)
        fade_len = min(overlap_samples, actual_len)
        if fade_len > 0 and start > 0:
            weight[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        if fade_len > 0 and end < audio_len:
            weight[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

        output[start:end] += chunk_out * weight
        weights[start:end] += weight

    output /= np.maximum(weights, 1e-6)
    peak = float(np.max(np.abs(output)))
    if peak > 0.98:
        output *= 0.95 / peak
    return output.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="DDSP lead instrument transfer")
    parser.add_argument("--input", required=True, help="入力 vocal stem")
    parser.add_argument("--output", required=True, help="出力 wav")
    parser.add_argument("--model-dir", required=True, help="legacy DDSP model dir")
    parser.add_argument("--backend", choices=["auto", "vst", "legacy"], default="auto")
    parser.add_argument("--plugin-binary", default=None, help="DDSP_VST の Effect バイナリ")
    parser.add_argument("--vst-model", default="Flute", help="DDSP_VST の埋め込みモデル名")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--quiet", type=float, default=28.0)
    parser.add_argument("--autotune", type=float, default=0.0)
    parser.add_argument("--pitch-shift", type=float, default=0.0)
    parser.add_argument("--loudness-shift", type=float, default=2.0)
    parser.add_argument("--noise-mix", type=float, default=0.02)
    parser.add_argument("--reverb-mix", type=float, default=0.0)
    parser.add_argument("--post-lowpass", type=float, default=4000.0)
    parser.add_argument("--post-highpass", type=float, default=90.0)
    parser.add_argument("--output-drive", type=float, default=1.0)
    parser.add_argument("--pitch-anchor-mix", type=float, default=0.18)
    parser.add_argument("--f0-min-confidence", type=float, default=0.08)
    parser.add_argument("--activity-threshold", type=float, default=0.16)
    parser.add_argument("--chunk-seconds", type=float, default=12.0)
    parser.add_argument("--chunk-overlap-seconds", type=float, default=0.6)
    args = parser.parse_args()
    load_runtime()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = load_audio(input_path)
    gin_file = model_dir / "operative_config-0.gin"
    dataset_stats_file = model_dir / "dataset_statistics.pkl"
    ckpt = latest_checkpoint(model_dir)

    dataset_stats = None
    if dataset_stats_file.exists():
        with dataset_stats_file.open("rb") as f:
            dataset_stats = pickle.load(f)

    use_vst_backend = False
    vst_plugin = find_ddsp_vst_plugin(args.plugin_binary)
    if args.backend == "vst":
        if vst_plugin is None:
            raise FileNotFoundError("DDSP_VST の Effect バイナリが見つかりません。")
        use_vst_backend = True
    elif args.backend == "auto" and vst_plugin is not None:
        use_vst_backend = True

    start = time.time()
    if use_vst_backend:
        audio_out = transfer_audio_vst(
            audio=audio,
            plugin_binary=vst_plugin,
            model_name=args.vst_model,
            args=args,
        )
    else:
        with gin.unlock_config():
            gin.clear_config()
            gin.parse_config_file(str(gin_file), skip_unknown=True)
        audio_out = transfer_audio(
            audio=audio,
            gin_file=gin_file,
            checkpoint=ckpt,
            dataset_stats=dataset_stats,
            args=args,
        )
    print(f"Prediction took {time.time() - start:.1f}s")

    sf.write(output_path, audio_out, SAMPLE_RATE, subtype="PCM_24")
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
