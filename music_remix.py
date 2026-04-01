#!/usr/bin/env python3
"""
music_remix.py — 曲の楽器分離・クオンタイズ・音量調整・フォルマント・リバーブ処理

機能:
  1. demucs で楽器分離 (drums, bass, vocals, other)
  2. 各ステムをビートグリッドにクオンタイズ
  3. 楽器ごとの音量調整
  4. ボーカル/メロディのフォルマントシフト
  5. 全体にリバーブ

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
from scipy.interpolate import interp1d

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
        description="曲の楽器分離・クオンタイズ・音量調整・フォルマント・リバーブ処理"
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
    print(f"  リバーブ: size={args.reverb_size}, wet={args.reverb_wet}")
    print("=" * 60)

    # --- Step 1: 音源分離 ---
    if args.no_separate and args.stems_dir:
        stem_dir = Path(args.stems_dir)
        stems = {}
        for f in sorted(stem_dir.glob("*.wav")):
            audio, sr = sf.read(f, dtype="float64")
            stems[f.stem] = audio
        print(f"[1/5] 既存ステム読み込み: {list(stems.keys())}")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="remix_")
        stems, sr = separate_stems(input_path, tmp_dir)

    # --- Step 2: クオンタイズ ---
    if args.quantize:
        print(f"\n[2/5] クオンタイズ ({args.quantize}, strength={args.quantize_strength})...")
        for name in stems:
            print(f"  {name}:")
            stems[name] = quantize_stem(
                stems[name], sr,
                grid=args.quantize,
                strength=args.quantize_strength
            )
    else:
        print("\n[2/5] クオンタイズ: スキップ")

    # --- Step 3: フォルマントシフト ---
    if args.formant_shift != 0:
        print(f"\n[3/5] フォルマントシフト ({args.formant_shift:+.1f}半音)...")
        for name in stems:
            if name in args.formant_target:
                print(f"  {name}: 処理中...")
                stems[name] = shift_formant(stems[name], sr, args.formant_shift)
                print(f"  {name}: 完了")
            else:
                print(f"  {name}: スキップ")
    else:
        print("\n[3/5] フォルマントシフト: スキップ")

    # --- Step 4: 音量調整 + ミックス ---
    print(f"\n[4/5] 音量調整 & ミックス...")
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

    # --- Step 5: リバーブ ---
    print(f"\n[5/5] リバーブ (size={args.reverb_size}, wet={args.reverb_wet})...")
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
