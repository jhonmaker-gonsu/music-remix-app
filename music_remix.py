#!/usr/bin/env python3
"""
music_remix.py — 曲の楽器分離・クオンタイズ・音量調整・フォルマント・
楽器風ボーカル変換・リバーブ処理

機能:
  1. demucs で楽器分離 (drums, bass, vocals, other)
  2. 各ステムをビートグリッドにクオンタイズ
  3. 楽器ごとの音量調整
  4. ボーカル/メロディのフォルマントシフト
  5. ボーカルの「歌詞感」を落として楽器風に変換
  6. 全体にリバーブ

Usage:
  python music_remix.py input.wav -o output.wav \
      --volumes drums=1.2 bass=0.8 vocals=1.0 other=0.6 \
      --quantize 16th --quantize-strength 0.8 \
      --formant-shift 2.0 --formant-target vocals \
      --reverb-size 0.85 --reverb-wet 0.6

  # クオンタイズなし、リバーブ強め
  python music_remix.py input.wav -o output.wav --reverb-size 0.95 --reverb-wet 0.7

  # フォルマントを other (メロディ楽器) にも適用
  python music_remix.py input.wav -o output.wav --formant-target vocals other --formant-shift 3.0

  # ボーカルを歌メロ楽器化して、Suno参照向けのガイドを作る
  python music_remix.py input.wav -o output.wav --instrumentize-vocals 0.8 \
      --instrumentize-breath 0.9 --instrumentize-tone 0.55 \
      --instrumentize-consonants 0.8 --instrumentize-modblur 0.7 \
      --instrumentize-grit 0.85 --instrumentize-robot 0.55
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
from pedalboard import Pedalboard, Reverb
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, sosfiltfilt

from audio_quantize import quantize_stem as _shared_quantize_stem


# ──────────────────────────────────────────────
# 1. 音源分離 (demucs)
# ──────────────────────────────────────────────
def separate_stems(input_path: str, output_dir: str) -> dict:
    """demucs (htdemucs) で4ステムに分離"""
    print("[1/5] 音源分離中 (demucs htdemucs)...")
    cmd = [sys.executable, "-m", "demucs", "-o", output_dir, input_path]
    subprocess.run(cmd, check=True)

    track_name = Path(input_path).stem
    stem_dir = Path(output_dir) / "htdemucs" / track_name

    stems = {}
    for stem_file in sorted(stem_dir.glob("*.wav")):
        audio, sr = sf.read(stem_file, dtype="float64")
        stems[stem_file.stem] = audio
        ch = "stereo" if audio.ndim == 2 else "mono"
        dur = len(audio) / sr
        print(f"  -> {stem_file.stem}: {dur:.1f}s, {sr}Hz, {ch}")

    return stems, sr


# ──────────────────────────────────────────────
# 2. クオンタイズ
# ──────────────────────────────────────────────
def quantize_stem(audio: np.ndarray, sr: int,
                  grid: str = "16th", strength: float = 1.0) -> np.ndarray:
    return _shared_quantize_stem(audio, sr, grid, strength)


# ──────────────────────────────────────────────
# 3. フォルマントシフト (WORLD vocoder)
# ──────────────────────────────────────────────
def shift_formant(audio: np.ndarray, sr: int, shift_semitones: float = 2.0) -> np.ndarray:
    """
    WORLD vocoder でスペクトル包絡を周波数方向にシフト（フォルマントを変える）
    ピッチは変えずに声質だけ変化する。

    Parameters:
        shift_semitones: 正=高く、負=低く (半音単位)
    """
    is_stereo = audio.ndim == 2
    ratio = 2.0 ** (shift_semitones / 12.0)

    def _process_mono(mono):
        mono = mono.astype(np.float64)
        # WORLD 分析
        f0, t = pw.harvest(mono, sr)
        f0 = pw.stonemask(mono, f0, t, sr)
        sp = pw.cheaptrick(mono, f0, t, sr)
        ap = pw.d4c(mono, f0, t, sr)

        # スペクトル包絡を周波数方向にシフト (= フォルマントシフト)
        freq_axis = np.arange(sp.shape[1])
        shifted_sp = np.zeros_like(sp)
        for i in range(sp.shape[0]):
            new_axis = freq_axis / ratio
            shifted_sp[i] = np.interp(freq_axis, new_axis, sp[i],
                                       left=sp[i, 0], right=sp[i, -1])

        # 再合成 (ピッチは元のまま)
        out = pw.synthesize(f0, shifted_sp, ap, sr)
        # 長さを元に揃える
        if len(out) > len(mono):
            out = out[:len(mono)]
        elif len(out) < len(mono):
            out = np.pad(out, (0, len(mono) - len(out)))
        return out

    if is_stereo:
        left = _process_mono(audio[:, 0])
        right = _process_mono(audio[:, 1])
        return np.column_stack([left, right])
    else:
        return _process_mono(audio)


def bridge_short_unvoiced_gaps(
    f0_hz: np.ndarray,
    max_gap_frames: int,
    max_jump_semitones: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """短い無声音ギャップを補間し、母音のつながりだけを保つ。"""
    bridged = np.asarray(f0_hz, dtype=np.float64).copy()
    bridge_mask = np.zeros_like(bridged, dtype=bool)
    voiced = bridged > 0.0
    n_frames = len(bridged)

    index = 0
    while index < n_frames:
        if voiced[index]:
            index += 1
            continue

        start = index
        while index < n_frames and not voiced[index]:
            index += 1
        end = index
        gap = end - start

        if start == 0 or end >= n_frames or gap <= 0 or gap > max_gap_frames:
            continue

        left = bridged[start - 1]
        right = bridged[end]
        if left <= 0.0 or right <= 0.0:
            continue

        jump_semitones = abs(12.0 * np.log2(max(right, 1e-6) / max(left, 1e-6)))
        if jump_semitones <= max_jump_semitones:
            bridged[start:end] = np.linspace(left, right, gap + 2, dtype=np.float64)[1:-1]
        else:
            bridged[start:end] = np.sqrt(left * right)
        bridge_mask[start:end] = True

    return bridged, bridge_mask


def cleanup_instrumentized_low_end(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """WORLD再合成で出やすい低域のうなり/DCを軽く掃除する。"""
    mono = np.asarray(audio, dtype=np.float64)
    cutoff_hz = float(np.clip(38.0 + (26.0 * amount), 35.0, 72.0))
    sos = butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
    filtered = sosfiltfilt(sos, mono).astype(np.float64)
    return filtered


def bandlimit_melody_core(audio: np.ndarray, sr: int) -> np.ndarray:
    """元ボーカルから、子音を拾いにくいメロディ芯だけを抜き出す。"""
    mono = np.asarray(audio, dtype=np.float64)
    low_sos = butter(3, 2400.0, btype="lowpass", fs=sr, output="sos")
    high_sos = butter(2, 110.0, btype="highpass", fs=sr, output="sos")
    shaped = sosfiltfilt(low_sos, mono)
    shaped = sosfiltfilt(high_sos, shaped)
    return shaped.astype(np.float64)


def bandlimit_high_melody_core(audio: np.ndarray, sr: int) -> np.ndarray:
    """高音域の輪郭を守るため、少し広めの帯域で元ボーカル芯を抜き出す。"""
    mono = np.asarray(audio, dtype=np.float64)
    low_sos = butter(3, 5500.0, btype="lowpass", fs=sr, output="sos")
    high_sos = butter(2, 160.0, btype="highpass", fs=sr, output="sos")
    shaped = sosfiltfilt(low_sos, mono)
    shaped = sosfiltfilt(high_sos, shaped)
    return shaped.astype(np.float64)


def soft_hpss(
    spectrum: np.ndarray,
    harmonic_kernel: int = 31,
    percussive_kernel: int = 31,
    harmonic_margin: float = 1.0,
    percussive_margin: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Median filtering based HPSS without sklearn dependency."""
    magnitude = np.abs(spectrum).astype(np.float64)
    power = magnitude ** 2.0

    harmonic_med = median_filter(power, size=(1, harmonic_kernel), mode="nearest")
    percussive_med = median_filter(power, size=(percussive_kernel, 1), mode="nearest")

    harmonic_score = harmonic_med / np.maximum(harmonic_med + (percussive_margin * percussive_med), 1e-12)
    percussive_score = percussive_med / np.maximum((harmonic_margin * harmonic_med) + percussive_med, 1e-12)

    total = np.maximum(harmonic_score + percussive_score, 1e-12)
    harmonic_mask = harmonic_score / total
    percussive_mask = percussive_score / total
    return harmonic_mask.astype(np.float64), percussive_mask.astype(np.float64)


def frame_activity_from_audio(audio: np.ndarray, sr: int, frame_times: np.ndarray) -> np.ndarray:
    """元音声の短時間RMSから歌唱アクティビティを推定する。"""
    mono = np.asarray(audio, dtype=np.float64)
    if len(mono) == 0 or len(frame_times) == 0:
        return np.zeros(len(frame_times), dtype=np.float64)

    half_window = max(64, int(round(sr * 0.016)))
    rms = np.zeros(len(frame_times), dtype=np.float64)
    for idx, frame_time in enumerate(frame_times):
        center = int(round(float(frame_time) * sr))
        start = max(0, center - half_window)
        end = min(len(mono), center + half_window)
        if end <= start:
            continue
        segment = mono[start:end]
        rms[idx] = np.sqrt(np.mean(np.square(segment), dtype=np.float64) + 1e-10)

    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    floor = float(np.percentile(rms_db, 20))
    peak = float(np.percentile(rms_db, 98))
    norm = np.clip((rms_db - floor) / max(peak - floor, 1e-3), 0.0, 1.0)
    return gaussian_filter1d(norm, sigma=0.8, mode="nearest")


def frame_envelope_to_samples(frame_values: np.ndarray, frame_times: np.ndarray, n_samples: int, sr: int) -> np.ndarray:
    frame_values = np.asarray(frame_values, dtype=np.float64)
    frame_times = np.asarray(frame_times, dtype=np.float64)
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float64)
    if len(frame_values) == 0:
        return np.zeros(n_samples, dtype=np.float64)
    if len(frame_values) == 1:
        return np.full(n_samples, float(frame_values[0]), dtype=np.float64)

    sample_points = np.clip(np.round(frame_times * sr).astype(np.int64), 0, max(n_samples - 1, 0))
    sample_points = np.maximum.accumulate(sample_points)
    unique_points, unique_indices = np.unique(sample_points, return_index=True)
    unique_values = frame_values[unique_indices]
    if unique_points[0] != 0:
        unique_points = np.insert(unique_points, 0, 0)
        unique_values = np.insert(unique_values, 0, unique_values[0])
    if unique_points[-1] != n_samples - 1:
        unique_points = np.append(unique_points, n_samples - 1)
        unique_values = np.append(unique_values, unique_values[-1])

    return np.interp(np.arange(n_samples, dtype=np.float64), unique_points, unique_values).astype(np.float64)


def instrumentize_vocal(
    audio: np.ndarray,
    sr: int,
    amount: float = 0.7,
    breath_reduction: float = 0.75,
    tone_darken: float = 0.35,
    consonant_suppress: float = 0.65,
    modulation_blur: float = 0.45,
    grit_drive: float = 0.0,
    robot_mod: float = 0.0,
) -> np.ndarray:
    """
    ボーカルのメロディを保ちながら、言葉っぽさを減らして楽器風に寄せる。

    WORLDで再合成する際に、
      - スペクトル包絡を平滑化してフォルマントの細かい揺れを減らす
      - 非周期成分(ap)を抑えて息・子音感を弱める
      - 高域を少し暗くして言葉の明瞭さを落とす

    Parameters:
        amount: 全体の効き具合 0.0-1.0
        breath_reduction: 息・子音成分の抑制量 0.0-1.0
        tone_darken: 高域を落として楽器っぽくする量 0.0-1.0
        consonant_suppress: 無声音/子音っぽいフレームを落とす量 0.0-1.0
        modulation_blur: 高域の時間包絡をぼかす量 0.0-1.0
        grit_drive: 波形折り返し/サンプルレート劣化を混ぜる量 0.0-1.0
        robot_mod: リング変調寄りのロボ感を混ぜる量 0.0-1.0
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    breath_reduction = float(np.clip(breath_reduction, 0.0, 1.0))
    tone_darken = float(np.clip(tone_darken, 0.0, 1.0))
    consonant_suppress = float(np.clip(consonant_suppress, 0.0, 1.0))
    modulation_blur = float(np.clip(modulation_blur, 0.0, 1.0))
    grit_drive = float(np.clip(grit_drive, 0.0, 1.0))
    robot_mod = float(np.clip(robot_mod, 0.0, 1.0))
    if amount <= 0.0:
        return audio

    is_stereo = audio.ndim == 2

    def _apply_robot_grit(mono: np.ndarray) -> np.ndarray:
        shaped = mono.astype(np.float64)

        if robot_mod > 0.0:
            carrier_hz = 28.0 + (172.0 * robot_mod)
            t_axis = np.arange(len(shaped), dtype=np.float64) / float(sr)
            carrier = np.sin(2.0 * np.pi * carrier_hz * t_axis)
            ring = shaped * carrier
            mix = 0.18 + (0.72 * robot_mod)
            shaped = ((1.0 - mix) * shaped) + (mix * ring)

        if grit_drive > 0.0:
            drive = 1.0 + (22.0 * grit_drive)
            folded = np.sin(shaped * drive)
            clipped = np.tanh(folded * (1.4 + (7.0 * grit_drive)))

            hold = 1 + int(round(4 + (44 * grit_drive)))
            crushed = np.repeat(clipped[::hold], hold)[: len(clipped)]
            if len(crushed) < len(clipped):
                crushed = np.pad(crushed, (0, len(clipped) - len(crushed)), mode="edge")

            mix = 0.25 + (0.75 * grit_drive)
            shaped = ((1.0 - mix) * shaped) + (mix * crushed)

        peak = np.max(np.abs(shaped))
        if peak > 1e-8:
            shaped = shaped / max(1.0, peak / 0.92)
        return shaped

    def _normalize_curve(values: np.ndarray, floor_pct: float, ceil_pct: float) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        low = float(np.percentile(values, floor_pct))
        high = float(np.percentile(values, ceil_pct))
        return np.clip((values - low) / max(high - low, 1e-6), 0.0, 1.0)

    def _process_mono(mono: np.ndarray) -> np.ndarray:
        mono = mono.astype(np.float64)
        n_fft = 4096 if sr >= 32000 else 2048
        hop_length = n_fft // 8

        spectrum = librosa.stft(mono.astype(np.float32), n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
        harmonic_mask, percussive_mask = soft_hpss(
            spectrum,
            harmonic_kernel=31,
            percussive_kernel=31,
            harmonic_margin=1.0 + (2.4 * amount),
            percussive_margin=1.0 + (6.0 * max(breath_reduction, consonant_suppress)),
        )
        harmonic = spectrum * harmonic_mask
        percussive = spectrum * percussive_mask

        magnitude = np.abs(spectrum) + 1e-8
        harmonic_mag = np.abs(harmonic)
        percussive_mag = np.abs(percussive)
        residual_mag = np.maximum(magnitude - harmonic_mag - percussive_mag, 0.0)

        frame_rms = np.sqrt(np.mean(np.square(magnitude), axis=0) + 1e-10)
        frame_flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
        onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(magnitude, ref=np.max), sr=sr, hop_length=hop_length)
        onset_env = np.pad(onset_env, (0, max(0, magnitude.shape[1] - len(onset_env))), mode="edge")[: magnitude.shape[1]]

        harmonic_ratio = np.sum(harmonic_mag, axis=0) / np.maximum(np.sum(magnitude, axis=0), 1e-8)
        percussive_ratio = np.sum(percussive_mag, axis=0) / np.maximum(np.sum(magnitude, axis=0), 1e-8)
        residual_ratio = np.sum(residual_mag, axis=0) / np.maximum(np.sum(magnitude, axis=0), 1e-8)
        f0_hz, _, _ = librosa.pyin(
            mono.astype(np.float32),
            sr=sr,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            frame_length=n_fft,
            hop_length=hop_length,
        )
        if f0_hz is None:
            f0_hz = np.zeros(magnitude.shape[1], dtype=np.float64)
        else:
            f0_hz = np.nan_to_num(f0_hz, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        if len(f0_hz) < magnitude.shape[1]:
            f0_hz = np.pad(f0_hz, (0, magnitude.shape[1] - len(f0_hz)))
        elif len(f0_hz) > magnitude.shape[1]:
            f0_hz = f0_hz[: magnitude.shape[1]]
        gap_frames = max(3, int(round((0.075 * sr) / hop_length)))
        f0_track, bridge_mask = bridge_short_unvoiced_gaps(
            f0_hz,
            max_gap_frames=gap_frames,
            max_jump_semitones=6.5,
        )

        activity = gaussian_filter1d(_normalize_curve(frame_rms, 20, 98), sigma=0.8, mode="nearest")
        flatness_score = gaussian_filter1d(_normalize_curve(frame_flatness, 20, 98), sigma=0.6, mode="nearest")
        onset_score = gaussian_filter1d(_normalize_curve(onset_env, 30, 98), sigma=0.7, mode="nearest")
        percussive_score = gaussian_filter1d(_normalize_curve(percussive_ratio + (0.75 * residual_ratio), 20, 98), sigma=0.7, mode="nearest")
        sustain_score = gaussian_filter1d(np.clip(harmonic_ratio * (0.65 + (0.35 * activity)), 0.0, 1.0), sigma=1.0, mode="nearest")

        frame_articulation = np.clip(
            (0.55 * percussive_score)
            + (0.45 * flatness_score)
            + (0.35 * onset_score)
            + (0.18 * (1.0 - activity)),
            0.0,
            1.0,
        )
        high_pitch_score = gaussian_filter1d(np.clip((f0_track - 180.0) / 40.0, 0.0, 1.0), sigma=1.0, mode="nearest")
        high_pitch_seed = np.clip((f0_track - 185.0) / 85.0, 0.0, 1.0)
        high_pitch_seed = np.maximum(high_pitch_seed, 0.55 * bridge_mask.astype(np.float64))
        high_pitch_focus = np.clip(high_pitch_score * np.clip(0.52 + (0.48 * sustain_score), 0.0, 1.0), 0.0, 1.0)
        high_pitch_guard = gaussian_filter1d(
            np.maximum(high_pitch_focus, high_pitch_seed * np.clip(0.48 + (0.52 * activity), 0.0, 1.0)),
            sigma=1.8,
            mode="nearest",
        )
        high_pitch_guard = np.clip(np.maximum(high_pitch_guard, high_pitch_focus), 0.0, 1.0)
        # スペクトル重心フォールバック: pyin が f0=0 を返す高音フレームも保護
        centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        unvoiced_mask = (f0_track == 0.0).astype(np.float64)
        centroid_guard = gaussian_filter1d(
            np.clip((centroid - 1500.0) / 1500.0, 0.0, 1.0) * unvoiced_mask * 0.6,
            sigma=1.2,
            mode="nearest",
        )
        high_pitch_guard = np.clip(np.maximum(high_pitch_guard, centroid_guard), 0.0, 1.0)
        high_pitch_relief = 1.0 - (0.98 * high_pitch_guard[np.newaxis, :])

        freq_hz = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        highband = np.clip((freq_hz - 1600.0) / 3200.0, 0.0, 1.0)
        lowband = np.clip((220.0 - freq_hz) / 220.0, 0.0, 1.0)
        harmonic_protect = np.zeros((len(freq_hz), magnitude.shape[1]), dtype=np.float64)
        voiced_frames = f0_track > 0.0
        if np.any(voiced_frames):
            for harmonic_index in range(1, 10):
                harmonic_hz = f0_track * harmonic_index
                valid = voiced_frames & (harmonic_hz < (sr * 0.47))
                if not np.any(valid):
                    continue
                width_hz = np.maximum(90.0, (0.16 * harmonic_hz) + 24.0)
                dist_hz = np.abs(freq_hz[:, np.newaxis] - harmonic_hz[np.newaxis, :])
                protect = np.exp(-0.5 * np.square(dist_hz / np.maximum(width_hz[np.newaxis, :], 1.0)))
                protect *= valid[np.newaxis, :]
                protect *= (1.0 / (harmonic_index ** 0.55))
                harmonic_protect = np.maximum(harmonic_protect, protect)
        harmonic_protect *= (0.45 + (0.55 * high_pitch_guard[np.newaxis, :]))
        harmonic_protect = np.clip(harmonic_protect, 0.0, 1.0)
        highband_matrix = highband[:, np.newaxis]
        protected_highband = highband_matrix * (1.0 - (0.88 * harmonic_protect))
        suppression_highband = protected_highband * (1.0 - (0.74 * high_pitch_guard[np.newaxis, :]))
        dynamic_amount = amount * (1.0 - (0.85 * high_pitch_guard[np.newaxis, :]))
        dynamic_breath = breath_reduction * (1.0 - (0.90 * high_pitch_guard[np.newaxis, :]))
        dynamic_consonant = consonant_suppress * (1.0 - (0.88 * high_pitch_guard[np.newaxis, :]))
        dynamic_blur = modulation_blur * (1.0 - (0.92 * high_pitch_guard[np.newaxis, :]))
        dynamic_tone = tone_darken * (1.0 - (0.97 * high_pitch_guard[np.newaxis, :]))

        harmonic_gain = 1.0 + (0.05 * dynamic_amount * (1.0 - highband[:, np.newaxis]))
        residual_gain = 1.0 - (
            (0.70 * dynamic_amount * dynamic_breath * suppression_highband * high_pitch_relief)
            + (0.62 * dynamic_amount * dynamic_consonant * frame_articulation[np.newaxis, :] * suppression_highband * high_pitch_relief)
            + (0.26 * dynamic_amount * dynamic_blur * onset_score[np.newaxis, :] * suppression_highband)
        )
        residual_gain = np.clip(residual_gain, 0.04, 1.0)

        percussive_gain = 1.0 - (
            (0.82 * dynamic_amount * dynamic_breath * (0.30 + (0.70 * suppression_highband)) * high_pitch_relief)
            + (0.78 * dynamic_amount * dynamic_consonant * frame_articulation[np.newaxis, :] * (1.0 - (0.68 * harmonic_protect)) * high_pitch_relief)
        )
        percussive_gain = np.clip(percussive_gain, 0.02, 0.85)

        if tone_darken > 0.0:
            tone_gain = 10.0 ** ((-14.0 * dynamic_amount * dynamic_tone * suppression_highband) / 20.0)
        else:
            tone_gain = 1.0

        # 低域のうなりは percussive / residual 側だけ積極的に抑える。
        low_cleanup = 1.0 - (0.88 * lowband[:, np.newaxis] * (0.45 + (0.55 * frame_articulation[np.newaxis, :])))
        low_cleanup = np.clip(low_cleanup, 0.12, 1.0)

        out_spec = (
            (harmonic * harmonic_gain)
            + (percussive * percussive_gain * tone_gain * low_cleanup)
            + ((spectrum - harmonic - percussive) * residual_gain * tone_gain * low_cleanup)
        )

        # sustain 区間だけ、元ボーカルのメロディ芯を薄く残す。
        melody_core = bandlimit_melody_core(mono, sr)
        frame_keep = np.clip((0.78 * sustain_score) + (0.24 * activity) - (0.14 * frame_articulation), 0.0, 1.0)
        sample_keep = np.interp(
            np.arange(len(mono), dtype=np.float64),
            np.linspace(0.0, len(mono) - 1, num=len(frame_keep), dtype=np.float64),
            frame_keep,
        )

        out = librosa.istft(out_spec, hop_length=hop_length, win_length=n_fft, length=len(mono)).astype(np.float64)
        core_mix = (0.10 + (0.12 * amount)) * sample_keep
        out = ((1.0 - core_mix) * out) + (core_mix * melody_core)

        high_pitch_keep = np.interp(
            np.arange(len(mono), dtype=np.float64),
            np.linspace(0.0, len(mono) - 1, num=len(high_pitch_guard), dtype=np.float64),
            high_pitch_guard,
        )
        high_core = bandlimit_high_melody_core(mono, sr)
        high_core_mix = (0.22 + (0.18 * amount)) * sample_keep * high_pitch_keep
        out = ((1.0 - high_core_mix) * out) + (high_core_mix * high_core)

        # 高音ではほぼ dry 側に寄せて、メロディ輪郭を優先して残す。
        conservative_mix = np.clip((0.62 + (0.28 * amount)) * sample_keep * high_pitch_keep, 0.0, 0.96)
        out = ((1.0 - conservative_mix) * out) + (conservative_mix * mono)

        # 歌っていない区間は residual/percussive を落とすが、母音の持続は残す。
        silence_gate = np.clip((sample_keep - 0.05) / 0.95, 0.0, 1.0)
        out *= (0.06 + (0.94 * silence_gate))

        # 仕上げの de-ess / low-end cleanup
        if tone_darken > 0.0 or breath_reduction > 0.0:
            high_sos = butter(2, 5200.0, btype="lowpass", fs=sr, output="sos")
            softened = sosfiltfilt(high_sos, out)
            mix = (0.12 * amount * max(tone_darken, breath_reduction)) * (1.0 - (0.94 * high_pitch_keep))
            out = ((1.0 - mix) * out) + (mix * softened)

        out = cleanup_instrumentized_low_end(out, sr, amount)
        return _apply_robot_grit(out)

    if is_stereo:
        left = _process_mono(audio[:, 0])
        right = _process_mono(audio[:, 1])
        return np.column_stack([left, right])
    return _process_mono(audio)


# ──────────────────────────────────────────────
# 4. 音量調整
# ──────────────────────────────────────────────
def adjust_volume(audio: np.ndarray, gain: float) -> np.ndarray:
    """線形ゲイン適用 (1.0 = 変化なし)"""
    return audio * gain


# ──────────────────────────────────────────────
# 5. ミックス + リバーブ
# ──────────────────────────────────────────────
def mix_stems(stems: dict, sr: int) -> np.ndarray:
    """全ステムを加算ミックス"""
    arrays = list(stems.values())
    min_len = min(a.shape[0] for a in arrays)
    result = sum(a[:min_len] for a in arrays)
    return result


def apply_reverb(audio: np.ndarray, sr: int,
                 room_size: float = 0.85, wet: float = 0.5, dry: float = 0.5) -> np.ndarray:
    """pedalboard で全体にリバーブ"""
    board = Pedalboard([
        Reverb(room_size=room_size, wet_level=wet, dry_level=dry, width=1.0)
    ])
    # pedalboard は (channels, samples) の float32 を期待
    if audio.ndim == 1:
        buf = audio[np.newaxis, :].astype(np.float32)
    else:
        buf = audio.T.astype(np.float32)

    out = board(buf, sr)

    if audio.ndim == 1:
        return out[0].astype(np.float64)
    else:
        return out.T.astype(np.float64)


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────
def parse_volumes(volume_args: list[str] | None) -> dict:
    """'drums=1.2 bass=0.8' 形式をパース"""
    vols = {"drums": 1.0, "bass": 1.0, "vocals": 1.0, "other": 1.0}
    if volume_args:
        for v in volume_args:
            name, val = v.split("=")
            vols[name.strip()] = float(val.strip())
    return vols


def main():
    parser = argparse.ArgumentParser(
        description="曲の楽器分離・クオンタイズ・音量調整・フォルマント・楽器風ボーカル変換・リバーブ処理"
    )
    parser.add_argument("input", help="入力音声ファイル (wav, mp3, flac, etc.)")
    parser.add_argument("-o", "--output", default="remix_output.wav",
                        help="出力ファイル (default: remix_output.wav)")

    # 音量
    parser.add_argument("--volumes", nargs="*", metavar="STEM=GAIN",
                        help="楽器別音量 (例: drums=1.2 bass=0.8 vocals=1.0 other=0.6)")

    # クオンタイズ
    parser.add_argument("--quantize", default=None,
                        choices=["4th", "8th", "16th", "32nd", "64th"],
                        help="クオンタイズのグリッド (省略時: クオンタイズしない)")
    parser.add_argument("--quantize-strength", type=float, default=1.0,
                        help="クオンタイズの強さ 0.0-1.0 (default: 1.0)")

    # フォルマント
    parser.add_argument("--formant-shift", type=float, default=0.0,
                        help="フォルマントシフト量 (半音単位, 正=高く) (default: 0 = 無効)")
    parser.add_argument("--formant-target", nargs="*", default=["vocals"],
                        help="フォルマントを適用するステム (default: vocals)")

    # ボーカル楽器化
    parser.add_argument("--instrumentize-vocals", type=float, default=0.0,
                        help="ボーカルを楽器風に寄せる量 0.0-1.0 (default: 0 = 無効)")
    parser.add_argument("--instrumentize-breath", type=float, default=0.75,
                        help="息・子音成分の抑制量 0.0-1.0 (default: 0.75)")
    parser.add_argument("--instrumentize-tone", type=float, default=0.35,
                        help="高域を暗くして声感を減らす量 0.0-1.0 (default: 0.35)")
    parser.add_argument("--instrumentize-consonants", type=float, default=0.65,
                        help="無声音/子音フレームの抑制量 0.0-1.0 (default: 0.65)")
    parser.add_argument("--instrumentize-modblur", type=float, default=0.45,
                        help="高域の時間包絡ぼかし量 0.0-1.0 (default: 0.45)")
    parser.add_argument("--instrumentize-grit", type=float, default=0.0,
                        help="波形折り返し/サンプルレート劣化の量 0.0-1.0 (default: 0.0)")
    parser.add_argument("--instrumentize-robot", type=float, default=0.0,
                        help="リング変調寄りのロボ感の量 0.0-1.0 (default: 0.0)")

    # リバーブ
    parser.add_argument("--reverb-size", type=float, default=0.85,
                        help="リバーブの広さ 0.0-1.0 (default: 0.85)")
    parser.add_argument("--reverb-wet", type=float, default=0.5,
                        help="リバーブ Wet レベル 0.0-1.0 (default: 0.5)")
    parser.add_argument("--reverb-dry", type=float, default=0.5,
                        help="リバーブ Dry レベル 0.0-1.0 (default: 0.5)")

    # その他
    parser.add_argument("--no-separate", action="store_true",
                        help="分離済みステムを再利用 (前回の出力ディレクトリ)")
    parser.add_argument("--stems-dir", default=None,
                        help="分離済みステムのディレクトリ")

    args = parser.parse_args()
    input_path = str(Path(args.input).resolve())
    volumes = parse_volumes(args.volumes)

    print("=" * 60)
    print("  Music Remix Pipeline")
    print("=" * 60)
    print(f"  入力: {args.input}")
    print(f"  出力: {args.output}")
    print(f"  音量: {volumes}")
    print(f"  クオンタイズ: {args.quantize or '無効'}"
          f" (strength={args.quantize_strength})" if args.quantize else "")
    print(f"  フォルマント: {args.formant_shift:+.1f}半音"
          f" -> {args.formant_target}" if args.formant_shift != 0 else "")
    print(
        f"  ボーカル楽器化: amount={args.instrumentize_vocals:.2f}, "
        f"breath={args.instrumentize_breath:.2f}, tone={args.instrumentize_tone:.2f}, "
        f"cons={args.instrumentize_consonants:.2f}, blur={args.instrumentize_modblur:.2f}, "
        f"grit={args.instrumentize_grit:.2f}, robot={args.instrumentize_robot:.2f}"
        if args.instrumentize_vocals > 0
        else ""
    )
    print(f"  リバーブ: size={args.reverb_size}, wet={args.reverb_wet}")
    print("=" * 60)

    # --- Step 1: 音源分離 ---
    if args.no_separate and args.stems_dir:
        stem_dir = Path(args.stems_dir)
        stems = {}
        for f in sorted(stem_dir.glob("*.wav")):
            audio, sr = sf.read(f, dtype="float64")
            stems[f.stem] = audio
        print(f"[1/6] 既存ステム読み込み: {list(stems.keys())}")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="remix_")
        stems, sr = separate_stems(input_path, tmp_dir)

    # --- Step 2: クオンタイズ ---
    if args.quantize:
        print(f"\n[2/6] クオンタイズ ({args.quantize}, strength={args.quantize_strength})...")
        for name in stems:
            print(f"  {name}:")
            stems[name] = quantize_stem(
                stems[name], sr,
                grid=args.quantize,
                strength=args.quantize_strength
            )
    else:
        print("\n[2/6] クオンタイズ: スキップ")

    # --- Step 3: フォルマントシフト ---
    if args.formant_shift != 0:
        print(f"\n[3/6] フォルマントシフト ({args.formant_shift:+.1f}半音)...")
        for name in stems:
            if name in args.formant_target:
                print(f"  {name}: 処理中...")
                stems[name] = shift_formant(stems[name], sr, args.formant_shift)
                print(f"  {name}: 完了")
            else:
                print(f"  {name}: スキップ")
    else:
        print("\n[3/6] フォルマントシフト: スキップ")

    # --- Step 4: ボーカル楽器化 ---
    if args.instrumentize_vocals > 0:
        print(
            f"\n[4/6] ボーカル楽器化 "
            f"(amount={args.instrumentize_vocals:.2f}, "
            f"breath={args.instrumentize_breath:.2f}, "
            f"tone={args.instrumentize_tone:.2f})..."
        )
        if "vocals" in stems:
            stems["vocals"] = instrumentize_vocal(
                stems["vocals"],
                sr,
                amount=args.instrumentize_vocals,
                breath_reduction=args.instrumentize_breath,
                tone_darken=args.instrumentize_tone,
                consonant_suppress=args.instrumentize_consonants,
                modulation_blur=args.instrumentize_modblur,
                grit_drive=args.instrumentize_grit,
                robot_mod=args.instrumentize_robot,
            )
            print("  vocals: 完了")
        else:
            print("  vocals: スキップ (ステムなし)")
    else:
        print("\n[4/6] ボーカル楽器化: スキップ")

    # --- Step 5: 音量調整 + ミックス ---
    print(f"\n[5/6] 音量調整 & ミックス...")
    for name in stems:
        gain = volumes.get(name, 1.0)
        stems[name] = adjust_volume(stems[name], gain)
        print(f"  {name}: x{gain:.2f}")

    mixed = mix_stems(stems, sr)

    # クリッピング防止 (ピークノーマライズ)
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)
        print(f"  ピークノーマライズ: {peak:.3f} -> 0.95")

    # --- Step 6: リバーブ ---
    print(f"\n[6/6] リバーブ (size={args.reverb_size}, wet={args.reverb_wet})...")
    output = apply_reverb(
        mixed, sr,
        room_size=args.reverb_size,
        wet=args.reverb_wet,
        dry=args.reverb_dry
    )

    # 最終ノーマライズ
    peak = np.max(np.abs(output))
    if peak > 0.99:
        output = output * (0.95 / peak)

    # 書き出し
    sf.write(args.output, output, sr, subtype="PCM_24")
    dur = len(output) / sr
    print(f"\n完了: {args.output} ({dur:.1f}s, {sr}Hz, 24bit)")


if __name__ == "__main__":
    main()
