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

## 現状 (2026-04-04 更新)

1. `music_remix.py` の `instrumentize_vocal()` が本命
2. **OOM バグ修正済み** (コミット `94e5180`)
3. **高音保護 v2 マージ済み** - 以下の改善が本体に反映済み:
   - `high_pitch_guard` 発動を 205Hz → 180Hz に早める
   - スペクトル重心ベースの f0-independent フォールバック追加
   - `high_pitch_relief` 係数 0.96 → 0.98
   - `dynamic_*` 削減係数をより保守的に
   - `bandlimit_high_melody_core` LP を 4200Hz → 5500Hz に拡大
   - `conservative_mix` 係数と上限を引き上げ

## 最優先課題

実音源での聴感検証と残課題対処。

**ユーザーの要求**: 「発音だけ減らして、元メロディを最大限保つ。高音で潰れないことを優先する」

## まずやってほしいこと

1. `HANDOFF_NEXT_AI.md` を読む
2. 実音源 `/tmp/ddsp_backnumber_45_53.wav` の `4.19s–5.63s` 区間で聴感検証
   (ファイルがない場合は合成信号で代替)
3. まだ潰れが残る場合は `HANDOFF_NEXT_AI.md` セクション 6 の次ステップを実施する
4. 変更・検証結果・残課題を報告してコミットする
