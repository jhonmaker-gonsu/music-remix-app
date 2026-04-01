# Music Remix App

楽曲を楽器ごとに分離し、各ステムにエフェクトを適用してリミックスできるGUIアプリケーション。

## 機能

- **音源分離**: demucs (htdemucs) で drums / bass / vocals / other に分離
- **楽器別エフェクト**: Volume / Distortion / Lowpass / Delay / Reverb / Compressor / Gain
- **フォルマントシフト**: ボーカルのフォルマントを変更（WORLD vocoder, harvest F0推定）
- **マスターエフェクト**: Reverb / Limiter
- **プレビュー**: 任意の位置から30秒間再生
- **ミュート**: ステムごとにON/OFF
- **ボーカルなし書き出し**: ワンクリックでカラオケ版を出力
- **WAV/MP3出力**: 24bit WAV または 320kbps MP3

## セットアップ

```bash
# 依存パッケージのインストール
pip install -r requirements.txt

# rubberband (クオンタイズ機能に必要)
brew install rubberband  # macOS
```

## 使い方

```bash
python3 remix_gui.py
```

1. **「MP3/WAV 読み込み」** で曲を選択 → demucs が自動で楽器分離（数分）
2. 各タブ（DRUMS / BASS / VOCALS / OTHER / MASTER）でスライダー調整
3. **「プレビュー (30秒)」** で確認（開始位置を指定可能）
4. **「書き出し」** で WAV/MP3 を出力

### 2回目以降

- **「ステム読み込み」** で前回の `remix_曲名/` フォルダを選べば即座に調整開始

### CLI

```bash
python3 music_remix.py input.mp3 -o output.wav \
    --volumes drums=2.0 bass=1.0 vocals=0.9 other=0.7 \
    --quantize 16th \
    --formant-shift -4.0 \
    --reverb-size 0.4 --reverb-wet 0.25
```

## 技術スタック

| ライブラリ | 用途 |
|-----------|------|
| PySide6 | GUI (Qt) |
| demucs | 音源分離 |
| pedalboard | エフェクト処理 |
| pyworld | フォルマントシフト |
| rubberband | オーディオクオンタイズ |
| sounddevice | プレビュー再生 |
| librosa | ビート/オンセット検出 |
