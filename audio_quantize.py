#!/usr/bin/env python3

import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


_DENSE_TIMEMAP_PARAMS = {
    "label": "dense",
    "tol_ms": 15.0,
    "min_anchor_ms": 180.0,
    "shift_gate_ms": 10.0,
}
_SPARSE_TIMEMAP_PARAMS = {
    "label": "sparse",
    "tol_ms": 8.0,
    "min_anchor_ms": 100.0,
    "shift_gate_ms": 6.0,
}
_ANCHOR_DENSITY_THRESHOLD = 500
_RUBBERBAND_ARGS = [
    "rubberband",
    "--fine",
    "--centre-focus",
]


def compute_quantize_targets(
    audio: np.ndarray,
    sr: int,
    grid: str = "16th",
    strength: float = 1.0,
):
    """Return onset targets and clamped shifts for quantization."""
    is_stereo = audio.ndim == 2
    mono = librosa.to_mono(audio.T) if is_stereo else audio.copy()

    tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=sr)
    tempo = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beat_times) < 2:
        return None

    divs = {"4th": 1, "8th": 2, "16th": 4, "32nd": 8, "64th": 16}
    d = divs.get(grid, 4)
    grid_times = []
    for i in range(len(beat_times) - 1):
        for j in range(d):
            grid_times.append(
                beat_times[i] + (beat_times[i + 1] - beat_times[i]) * j / d
            )
    grid_times.append(beat_times[-1])
    grid_samples = librosa.time_to_samples(np.array(grid_times), sr=sr)

    onset_frames = librosa.onset.onset_detect(y=mono, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    if len(onset_times) < 2:
        return None

    onset_samples = librosa.time_to_samples(onset_times, sr=sr)
    min_gap = max(1, int(round(sr * 0.020)))
    filtered_onsets = []
    last_onset = -min_gap
    for onset_sample in onset_samples:
        onset_sample = int(onset_sample)
        if onset_sample - last_onset >= min_gap:
            filtered_onsets.append(onset_sample)
            last_onset = onset_sample
    if len(filtered_onsets) < 2:
        return None

    abs_max_shift = int(round(sr * 0.095))
    shifts = []
    for onset_sample in filtered_onsets:
        insert_idx = int(np.searchsorted(grid_samples, onset_sample))
        candidates = []
        if insert_idx > 0:
            candidates.append(insert_idx - 1)
        if insert_idx < len(grid_samples):
            candidates.append(insert_idx)
        if not candidates:
            shifts.append(0)
            continue

        nearest_idx = min(
            candidates,
            key=lambda idx: abs(int(grid_samples[idx]) - onset_sample),
        )
        nearest_grid = int(grid_samples[nearest_idx])

        neighbor_steps = []
        if nearest_idx > 0:
            neighbor_steps.append(nearest_grid - int(grid_samples[nearest_idx - 1]))
        if nearest_idx + 1 < len(grid_samples):
            neighbor_steps.append(int(grid_samples[nearest_idx + 1]) - nearest_grid)
        local_max = (
            max(1, int(round(min(neighbor_steps) / 2))) if neighbor_steps else 0
        )
        max_shift = min(local_max, abs_max_shift) if local_max else 0

        raw = int(round((nearest_grid - onset_sample) * strength))
        clamped = int(np.clip(raw, -max_shift, max_shift)) if max_shift else 0
        shifts.append(clamped)

    return {
        "mono": mono,
        "tempo": tempo,
        "grid_count": len(grid_samples),
        "onset_count": len(onset_times),
        "filtered_onsets": np.array(filtered_onsets, dtype=np.int64),
        "shifts": np.array(shifts, dtype=np.int64),
    }


def _rdp_indices(points: np.ndarray, tolerance: float) -> np.ndarray:
    if len(points) <= 2:
        return np.arange(len(points))

    keep = np.zeros(len(points), dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(points) - 1)]

    while stack:
        start, end = stack.pop()
        if end - start <= 1:
            continue

        x1, y1 = points[start]
        x2, y2 = points[end]
        xs = points[start + 1:end, 0]
        ys = points[start + 1:end, 1]
        if x2 == x1:
            interp = np.full_like(xs, y1)
        else:
            interp = y1 + (y2 - y1) * (xs - x1) / (x2 - x1)
        error = np.abs(ys - interp)
        rel_idx = int(np.argmax(error))
        if error[rel_idx] > tolerance:
            split_idx = start + 1 + rel_idx
            keep[split_idx] = True
            stack.append((start, split_idx))
            stack.append((split_idx, end))

    return np.flatnonzero(keep)


def _select_timemap_params(targets: dict) -> dict:
    onset_count = len(targets["filtered_onsets"])
    if onset_count >= _ANCHOR_DENSITY_THRESHOLD:
        return _DENSE_TIMEMAP_PARAMS
    return _SPARSE_TIMEMAP_PARAMS


def _build_timemap(
    n_samples: int,
    sr: int,
    targets: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    params = _select_timemap_params(targets)
    source = np.concatenate(
        ([0], targets["filtered_onsets"], [n_samples - 1])
    ).astype(np.int64)
    shifts = np.concatenate(([0], targets["shifts"], [0])).astype(np.int64)
    target = np.maximum.accumulate(source + shifts)
    for i in range(1, len(target)):
        if target[i] <= target[i - 1]:
            target[i] = target[i - 1] + 1

    must_keep = np.flatnonzero(
        np.abs(shifts) >= int(round(sr * params["shift_gate_ms"] / 1000.0))
    )
    reduced = _rdp_indices(
        np.column_stack([source, target]).astype(np.float64),
        sr * params["tol_ms"] / 1000.0,
    )
    reduced = np.unique(
        np.concatenate(([0, len(source) - 1], reduced, must_keep))
    )

    min_gap = int(round(sr * params["min_anchor_ms"] / 1000.0))
    pruned = [int(reduced[0])]
    last_source = int(source[reduced[0]])
    for idx in reduced[1:-1]:
        idx = int(idx)
        if int(source[idx]) - last_source >= min_gap:
            pruned.append(idx)
            last_source = int(source[idx])
    pruned.append(int(reduced[-1]))
    pruned = np.array(pruned, dtype=np.int64)

    return source[pruned], target[pruned], params


def _rubberband_quantize(
    audio: np.ndarray,
    sr: int,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="quantize_rb_") as tmpdir:
        tmp = Path(tmpdir)
        infile = tmp / "input.wav"
        outfile = tmp / "output.wav"
        timemap = tmp / "timemap.txt"

        sf.write(infile, audio, sr, subtype="FLOAT")
        with timemap.open("w", encoding="utf-8") as handle:
            for src, dst in zip(source, target):
                handle.write(f"{int(src)} {int(dst)}\n")

        cmd = [
            *_RUBBERBAND_ARGS,
            "--timemap",
            str(timemap),
            "-t",
            "1.0",
            str(infile),
            str(outfile),
        ]
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        output, _ = sf.read(outfile, dtype="float64")

    if len(output) < len(audio):
        pad = len(audio) - len(output)
        if output.ndim == 2:
            output = np.pad(output, ((0, pad), (0, 0)))
        else:
            output = np.pad(output, (0, pad))
    return output[:len(audio)]


def _quality_metrics(
    original: np.ndarray,
    candidate: np.ndarray,
    sr: int,
) -> tuple[int, int, float]:
    orig = original.mean(axis=1) if original.ndim == 2 else original
    quant = candidate.mean(axis=1) if candidate.ndim == 2 else candidate

    hop = max(1, int(sr * 0.01))
    new_silence = 0
    for start in range(0, len(orig) - hop, hop):
        ro = float(np.sqrt(np.mean(orig[start:start + hop] ** 2)))
        rq = float(np.sqrt(np.mean(quant[start:start + hop] ** 2)))
        if ro > 0.01 and rq < 0.0001:
            new_silence += 1

    do = np.abs(np.diff(orig))
    dq = np.abs(np.diff(quant))
    threshold = float(np.percentile(do, 99.9) * 3.0)
    new_clicks = max(0, int(np.sum(dq > threshold)) - int(np.sum(do > threshold)))

    rms_ratio = (
        float(np.sqrt(np.mean(quant ** 2))) /
        max(float(np.sqrt(np.mean(orig ** 2))), 1e-12) * 100.0
    )
    return new_silence, new_clicks, rms_ratio


def _crossfade_splice_quantize(audio: np.ndarray, sr: int, targets: dict) -> np.ndarray:
    is_stereo = audio.ndim == 2
    n_samples = audio.shape[0]
    filtered_onsets = targets["filtered_onsets"]
    shifts = targets["shifts"]

    original = audio.astype(np.float64)
    output = original.copy()
    fade = int(sr * 0.005)

    for i, onset_sample in enumerate(filtered_onsets):
        shift = int(shifts[i])
        if shift == 0:
            continue

        prev_mid = (filtered_onsets[i - 1] + onset_sample) // 2 if i > 0 else 0
        next_mid = (
            (onset_sample + filtered_onsets[i + 1]) // 2
            if i < len(filtered_onsets) - 1 else n_samples
        )

        segment = original[prev_mid:next_mid].copy()
        dst_start = prev_mid + shift
        dst_end = next_mid + shift

        clip_start = max(0, dst_start)
        clip_end = min(n_samples, dst_end)
        if clip_end <= clip_start:
            continue

        offset = clip_start - dst_start
        copy_len = clip_end - clip_start
        seg_slice = segment[offset:offset + copy_len]

        fade_len = min(fade, copy_len // 4)
        window = np.ones(copy_len, dtype=np.float64)
        if fade_len > 1:
            window[:fade_len] = np.linspace(0.0, 1.0, fade_len)
            window[-fade_len:] = np.linspace(1.0, 0.0, fade_len)

        if is_stereo:
            w = window[:, np.newaxis]
            output[clip_start:clip_end] = (
                output[clip_start:clip_end] * (1.0 - w) + seg_slice * w
            )
        else:
            output[clip_start:clip_end] = (
                output[clip_start:clip_end] * (1.0 - window) + seg_slice * window
            )

    return output.astype(audio.dtype, copy=False)


def quantize_stem(
    audio: np.ndarray,
    sr: int,
    grid: str = "16th",
    strength: float = 1.0,
) -> np.ndarray:
    targets = compute_quantize_targets(audio, sr, grid, strength)
    if targets is None:
        print("    (ビート/オンセット不足 → スキップ)")
        return audio

    print(
        f"    tempo={targets['tempo']:.1f}BPM, "
        f"onsets={targets['onset_count']}, "
        f"grid={grid}({targets['grid_count']}pts)"
    )

    shift_abs = [abs(int(s)) / sr * 1000 for s in targets["shifts"]]
    print(
        f"    shift: mean={np.mean(shift_abs):.1f}ms, "
        f"max={np.max(shift_abs):.1f}ms"
    )

    if shutil.which(_RUBBERBAND_ARGS[0]) is None:
        print("    rubberband not found -> fallback=crossfade-splice")
        return _crossfade_splice_quantize(audio, sr, targets)

    try:
        source, target, params = _build_timemap(audio.shape[0], sr, targets)
        quantized = _rubberband_quantize(audio, sr, source, target)
        new_silence, new_clicks, rms_ratio = _quality_metrics(audio, quantized, sr)
        print(
            f"    rubberband={params['label']} anchors={len(source)}, "
            f"new_silence={new_silence}, new_clicks={new_clicks}, "
            f"rms={rms_ratio:.1f}%"
        )
        if new_silence == 0 and new_clicks == 0 and rms_ratio >= 85.0:
            return quantized.astype(audio.dtype, copy=False)
        print("    rubberband quality gate failed -> fallback=crossfade-splice")
    except Exception as exc:
        print(f"    rubberband failed -> fallback=crossfade-splice ({exc})")

    return _crossfade_splice_quantize(audio, sr, targets)
