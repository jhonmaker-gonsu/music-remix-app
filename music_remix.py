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
from scipy.ndimage import gaussian_filter1d

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
            # ゆるいリング変調で人声らしい倍音関係を崩す。
            carrier_hz = 28.0 + (172.0 * robot_mod)
            t_axis = np.arange(len(shaped), dtype=np.float64) / float(sr)
            carrier = np.sin(2.0 * np.pi * carrier_hz * t_axis)
            ring = shaped * carrier
            mix = 0.18 + (0.72 * robot_mod)
            shaped = ((1.0 - mix) * shaped) + (mix * ring)

        if grit_drive > 0.0:
            # wavefold + sample-and-hold で、声の口周りのニュアンスをさらに壊す。
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

    def _process_mono(mono: np.ndarray) -> np.ndarray:
        mono = mono.astype(np.float64)
        f0, t = pw.harvest(mono, sr)
        f0 = pw.stonemask(mono, f0, t, sr)
        sp = pw.cheaptrick(mono, f0, t, sr)
        ap = pw.d4c(mono, f0, t, sr)

        # 細かなフォルマント変化を丸めて、歌詞の明瞭さを少し落とす。
        log_sp = np.log(np.maximum(sp, 1e-8))
        sigma = 1.0 + (amount * 7.0)
        smoothed_sp = np.exp(gaussian_filter1d(log_sp, sigma=sigma, axis=1, mode="nearest"))
        detail_keep = 1.0 - (0.88 * amount)
        shaped_sp = (detail_keep * sp) + ((1.0 - detail_keep) * smoothed_sp)

        # 高域だけ時間方向にぼかして、音節/子音の動きの手掛かりを減らす。
        if modulation_blur > 0.0:
            blur_sigma_frames = 0.5 + (5.5 * amount * modulation_blur)
            time_smoothed = np.exp(
                gaussian_filter1d(
                    np.log(np.maximum(shaped_sp, 1e-8)),
                    sigma=blur_sigma_frames,
                    axis=0,
                    mode="nearest",
                )
            )
            freq_hz = np.linspace(0.0, sr / 2.0, shaped_sp.shape[1], dtype=np.float64)
            highband_weight = np.clip((freq_hz - 1400.0) / 2600.0, 0.0, 1.0)
            blur_mix = amount * modulation_blur * highband_weight[np.newaxis, :]
            shaped_sp = (shaped_sp * (1.0 - blur_mix)) + (time_smoothed * blur_mix)

        # 高域を少し暗くして、声の子音・歯擦音が前に出すぎるのを抑える。
        if tone_darken > 0.0:
            freq_hz = np.linspace(0.0, sr / 2.0, shaped_sp.shape[1], dtype=np.float64)
            ramp = np.clip((freq_hz - 1200.0) / max((sr / 2.0) - 1200.0, 1.0), 0.0, 1.0)
            tilt_db = -18.0 * amount * tone_darken * ramp
            shaped_sp *= 10.0 ** (tilt_db[np.newaxis, :] / 20.0)

        # 息・子音に相当する非周期成分を落として、シンセ/リード寄りへ。
        if breath_reduction > 0.0:
            freq_ratio = np.linspace(0.0, 1.0, ap.shape[1], dtype=np.float64)
            reduction_curve = 0.20 + (0.80 * freq_ratio)
            reduction = 1.0 - (0.95 * amount * breath_reduction * reduction_curve)
            shaped_ap = np.clip(ap * reduction[np.newaxis, :], 0.0, 1.0)
        else:
            shaped_ap = ap

        # 無声音や高aperiodicityフレームを減衰して、歌詞の聞き取りをさらに落とす。
        if consonant_suppress > 0.0:
            voiced_mask = (f0 > 0.0).astype(np.float64)
            aperiodic_mean = np.mean(shaped_ap, axis=1)
            consonant_score = np.clip(
                (1.0 - voiced_mask) * 0.85 + (aperiodic_mean ** 0.8) * 0.9,
                0.0,
                1.0,
            )
            attenuation_db = -34.0 * amount * consonant_suppress * consonant_score
            frame_gain = 10.0 ** (attenuation_db / 20.0)
            shaped_sp *= frame_gain[:, np.newaxis]
            shaped_ap *= (0.30 + 0.70 * frame_gain[:, np.newaxis])

        out = pw.synthesize(f0, shaped_sp, shaped_ap, sr)
        if len(out) > len(mono):
            out = out[:len(mono)]
        elif len(out) < len(mono):
            out = np.pad(out, (0, len(mono) - len(out)))

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
