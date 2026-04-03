# Music Remix App

楽曲を楽器ごとに分離し、各ステムにエフェクトを適用してリミックスできるGUIアプリケーション。

## 機能

- **音源分離**: demucs (htdemucs) で drums / bass / vocals / other に分離
- **楽器別エフェクト**: Volume / Distortion / Lowpass / Delay / Reverb / Compressor / Gain
- **フォルマントシフト**: ボーカルのフォルマントを変更（WORLD vocoder, harvest F0推定）
- **歌メロ楽器化**: ボーカルのメロディは残しつつ、子音・息成分を減らして楽器っぽく変換
- **DDSP VST 楽器化**: DDSP_VST の埋め込みモデルを優先し、ボーカルステムをリード楽器へ置換
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
   - `VOCALS` タブの `楽器化 %` を上げると、歌詞感を落としてリード楽器っぽくできます
   - `子音抑制 %` と `暗さ %` で、言葉の聞こえやすさをさらに下げられます
   - `子音ゲート %` と `高域ぼかし %` を上げると、子音や音節の輪郭がさらに崩れて歌詞が聞き取りにくくなります
   - `激歪み %` は wavefold + サンプルレート劣化で音色自体を強く壊し、`ロボ変調 %` はロボ/シンセ寄りにします
   - 迷ったら上部の **「歌メロ楽器化」** ボタンで出発点プリセットを読み込めます
3. **「プレビュー (30秒)」** で確認（開始位置を指定可能）
4. **「書き出し」** で WAV/MP3 を出力

### DDSP VST 楽器化

- 上部の **「DDSP VST 楽器化」** を押すと、`VOCALS` ステムを DDSP_VST の Effect モデルで変換して置換します
- 初回だけ `.venv-ddsp/` を自動作成し、専用依存と `ddsp_models/solo_flute_ckpt/` を準備します
- `~/Library/Audio/Plug-Ins/Components/DDSP Effect.component` または `~/Library/Audio/Plug-Ins/VST3/DDSP Effect.vst3` があれば、埋め込み VST モデルを自動優先します
- 変換後は `VOCALS` タブの音量やEQをそのまま使って、ドラム/ベース/その他ステムと一緒にミックスできます

### 2回目以降

- **「ステム読み込み」** で前回の `remix_曲名/` フォルダを選べば即座に調整開始

### CLI

```bash
python3 music_remix.py input.mp3 -o output.wav \
    --volumes drums=2.0 bass=1.0 vocals=0.9 other=0.7 \
    --quantize 16th \
    --formant-shift -4.0 \
    --reverb-size 0.4 --reverb-wet 0.25

python3 music_remix.py input.mp3 -o guide.wav \
    --instrumentize-vocals 0.82 \
    --instrumentize-breath 0.88 \
    --instrumentize-tone 0.55 \
    --instrumentize-consonants 0.82 \
    --instrumentize-modblur 0.68 \
    --instrumentize-grit 0.78 \
    --instrumentize-robot 0.52 \
    --formant-shift -4.0
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
