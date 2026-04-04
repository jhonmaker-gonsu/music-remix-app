# 次AI向けプロンプト

このリポジトリの開発を引き継いでください。

## プロジェクト

- ローカルパス: `/Users/gon/Documents/AI_Generated_Apps/music-remix-app`
- GitHub: `https://github.com/jhonmaker-gonsu/music-remix-app`
- 最新コミット: `94e5180`

## 必須条件

- 日本語で応答する
- まず現状コードを読んでから判断する
- 実装だけで終わらず、短い音声で実検証する
- 詳細は `HANDOFF_NEXT_AI.md` を読むこと

## 最優先課題

「歌メロ楽器化」の高音域保護をさらに強化する。

**ユーザーの要求**: 「発音だけ減らして、元メロディを最大限保つ。高音で潰れないことを優先する」

## いまの状況

1. `music_remix.py` の `instrumentize_vocal()` が本命  
   (`DDSP VST 楽器化` は音色は良いが、メロディ保持で不満のため主戦場でない)

2. **OOM バグ修正済み** (コミット `94e5180`)  
   `harmonic_gain[:, np.newaxis]` → `harmonic_gain` に修正。  
   以前は実行のたびに大量の RAM を消費していた。

3. **高音保護 v2 が Colab でテスト済みだがまだ未マージ**  
   以下の改善を `instrumentize_vocal_v2` として Colab 検証した:
   - `high_pitch_guard` 発動を 205Hz → 180Hz に早める
   - スペクトル重心ベースの f0-independent フォールバック追加
   - `high_pitch_relief` 係数 0.96 → 0.98
   - `dynamic_*` 削減係数をより保守的に
   - `bandlimit_high_melody_core` LP を 4200Hz → 5500Hz に拡大
   - `conservative_mix` 係数と上限を引き上げ

## まずやってほしいこと

1. `HANDOFF_NEXT_AI.md` を読む
2. `music_remix.py` を読み、現在の `instrumentize_vocal()` を確認する
3. **Colab の v2 変更を `music_remix.py` 本体にマージする**  
   (`HANDOFF_NEXT_AI.md` のセクション 3 に変更内容の詳細あり)
4. `/tmp/ddsp_backnumber_45_53.wav` の `4.19s–5.63s` 区間で実検証する  
   (ファイルがない場合は合成信号で代替)
5. まだ潰れが残る場合は `HANDOFF_NEXT_AI.md` セクション 6 の次ステップを実施する
6. 変更・検証結果・残課題を報告してコミットする
