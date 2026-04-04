# ハイブリッド歌メロ楽器化アプリ 仕様書・実装タスク

## 1. 文書の目的

本書は、`music-remix-app` をベースにした次世代の「歌メロ楽器化」アプリの仕様と実装タスクを整理するための開発ドキュメントである。

今回の主目的は、歌声の `歌詞感・発音感` を減らしつつ、`メロディ・高音の輪郭・ビブラート・持続感` をできるだけ保持した半楽器化ボーカル処理を、ローカル中心で実用化することにある。


## 2. 背景

現行の `music_remix.py` にある `instrumentize_vocal()` は、`HPSS + 高域抑制 + 元音声の再注入` を軸に改善されてきたが、ユーザー評価では次の課題が残っている。

- 高音域でメロディ輪郭がまだ潰れる
- 子音を減らしたいが、倍音や持続まで削れやすい
- DDSP 系は音色面では魅力があるが、メロディ保持では不安定
- 「別の楽器に作り替える」より「言葉っぽさだけ減らしたい」という要件が強い

このため、本アプリでは以下 3 系統の発想を統合したハイブリッド設計を採用する。

- `NANSY 系`: linguistic / structural-noise 的な発音感の分離と抑制
- `VibE-SVC 系`: 高音ビブラートと細かな F0 輪郭の保護
- `Serenade 系`: source-guided な F0 / harmonic 骨格の再注入


## 3. 目的

- ボーカルから子音・息・無声音・発音のエッジを減らす
- 高音のメロディ輪郭、ビブラート、母音の持続を残す
- 実運用はローカルで完結できるようにする
- Colab は重い実験、比較、学習モデル検証の補助に限定する
- GUI から簡単に操作できる
- 既存の `music-remix-app` の延長として実装可能な範囲に留める


## 4. 非目的

- 完全な歌手変換
- 歌詞の認識や編集
- クラウド専用運用
- 学習必須の巨大モデルに全面依存する実装
- DAW プラグイン化の即時対応


## 5. 想定ユースケース

### 5.1 メインユースケース

ユーザーは楽曲を読み込み、Demucs で `drums / bass / vocals / other` に分離した後、`VOCALS` に対して歌メロ楽器化を適用する。結果として、歌詞の聞き取りやすさは低下するが、メロディは維持されたまま、フルートやシンセのようなリード楽器に近い印象のボーカルステムを得る。

### 5.2 成功条件

- 高音で潰れにくい
- 旋律を口ずさめる程度に保持される
- 子音や息が適度に減る
- 不自然な低域うなりがない
- GUI から試行錯誤しやすい


## 6. 運用方針

### 6.1 ローカルを主軸にする理由

- GUI アプリとして安定運用しやすい
- 音声ファイルの入出力が多い
- セッション切断や再起動に弱くない
- CPU ベースの DSP 処理が主戦場になる

### 6.2 Colab の位置づけ

- 重い比較実験
- 学習モデルの試作
- 可視化を伴う分析ノートブック
- バッチ検証

### 6.3 運用方針まとめ

- `制作`: ローカル中心
- `実運用`: ローカル中心
- `研究/検証`: Colab 併用


## 7. システム全体像

本アプリは次の 4 層で構成する。

### 7.1 入出力層

- 楽曲読み込み
- stem 読み込み
- Demucs による分離
- プレビュー
- WAV/MP3 書き出し

### 7.2 DSP 解析層

- STFT / ISTFT
- HPSS
- RMS / flatness / onset / harmonic ratio 解析
- F0 推定
- 高音区間検出

### 7.3 編集・保護層

- de-articulation
- vibrato / fine pitch contour 保護
- source-guided harmonic / dry 再注入
- 低域 cleanup

### 7.4 実験拡張層

- DDSP 系色付け
- Colab 実験ノートブック
- 将来の軽量 ML 補助モジュール


## 8. 機能仕様

### 8.1 入力

- 単曲の `mp3 / wav`
- 既存 stem フォルダ
- 単独 `vocals.wav` の検証入力

### 8.2 出力

- 加工済み `vocals`
- ミックス済みフル曲
- プレビュー音声
- 実験比較用ファイル

### 8.3 必須機能

- 歌メロ楽器化
- 高音保護
- 母音保持
- dry 再注入
- プレビュー
- WAV/MP3 書き出し
- 元 VOCALS への復帰

### 8.4 追加候補

- A/B 比較再生
- 高音区間の可視化
- パラメータの保存/復元
- 実験プリセット切り替え


## 9. コアアルゴリズム仕様

処理名は `Hybrid Vocal Instrumentize` とする。

### 9.1 Stage A: Pre Analysis

入力波形に対して以下を抽出する。

- STFT
- harmonic / percussive / residual 比率
- frame RMS
- spectral flatness
- onset strength
- pyin 由来 F0
- 短い無声音ギャップの補間
- sustain score
- articulation score
- high-pitch guard

### 9.2 Stage B: De-Articulation Engine

発音感の主要因を `percussive / residual / onset-rich frame` とみなし、以下を適用する。

- percussive の抑制
- residual の抑制
- articulation score に応じた動的減衰
- breath / consonant / modulation の個別制御

このステージは `歌詞感を落とすが、旋律そのものは壊しすぎない` ことを目的とする。

### 9.3 Stage C: High-Pitch Protection

高音の輪郭を守るため、以下を適用する。

- F0 ベースの high-pitch guard
- harmonic ridge 保護マスク
- 高音時の effect amount 自動低減
- 高音時の suppression 緩和
- 高音 sustain の dry 比率増加

### 9.4 Stage D: Vibrato / Fine Contour Preservation

高音区間では平均 F0 だけではなく、細かなピッチ変動も保持対象とする。

第一段階では以下を行う。

- bridged F0 の安定化
- 高音 sustain での保守的な処理
- 原音の harmonic core 再注入

将来的には以下を追加可能とする。

- vibrato depth / rate 推定
- fine contour 保護エンベロープ

### 9.5 Stage E: Source-Guided Resynthesis Lite

元のボーカルから以下を再注入する。

- melody core
- high melody core
- dry signal の一部

目的は次の通り。

- 母音の芯を戻す
- 高音倍音を戻す
- メロディ輪郭を再確保する

### 9.6 Stage F: Style Color Layer

必要に応じて軽い色付けを行う。

- tone darken
- grit drive
- robot modulation
- 将来の DDSP 後段色付け

この層は必須ではなく、常に bypass 可能とする。


## 10. GUI 仕様

### 10.1 基本要素

- `MP3/WAV 読み込み`
- `ステム読み込み`
- `プレビュー`
- `書き出し`
- `歌メロ楽器化` プリセット
- `元 VOCALS に戻す` ボタン

### 10.2 操作モード

#### 簡単モード

- 楽器化 %
- 発音抑制 %
- 高音保護 %
- 母音保持 %

#### 詳細モード

- breath reduction
- consonant suppress
- modulation blur
- tone darken
- melody core mix
- high core mix
- high-pitch guard
- high dry mix
- vibrato preserve
- low-end cleanup
- grit / robot

### 10.3 UX 方針

- デフォルトは保守的な音作り
- 高音保護を最優先
- 極端な設定でも破綻しにくいレンジにする
- プリセットを「出発点」として使えるようにする


## 11. データ構造と状態管理

### 11.1 セッション状態

- `stems_raw`
- `stems_original`
- `stem_params`
- `preview cache`
- `last_export_path`

### 11.2 重要要件

- DDSP や特殊処理後でも `元 VOCALS` を保持する
- `歌メロ楽器化` は常に original に戻せるようにする
- 実験処理と最終出力を混同しない


## 12. モジュール設計

### 12.1 既存中心

- [music_remix.py](/Users/gon/Documents/AI_Generated_Apps/music-remix-app/music_remix.py)
- [remix_gui.py](/Users/gon/Documents/AI_Generated_Apps/music-remix-app/remix_gui.py)
- [ddsp_flute_transfer.py](/Users/gon/Documents/AI_Generated_Apps/music-remix-app/ddsp_flute_transfer.py)

### 12.2 推奨分割

将来的には以下のような分離を検討する。

- `vocal_analysis.py`
- `vocal_protection.py`
- `vocal_resynthesis.py`
- `vocal_presets.py`
- `experiments/`
- `notebooks/`

第一段階では、無理な分離よりも既存コードの整理を優先する。


## 13. 品質要件

### 13.1 音質

- 高音で輪郭が明確に崩れない
- 低域のうなりが少ない
- 歌詞感が適度に落ちる
- 破綻ノイズが少ない

### 13.2 速度

- CPU ベースでも短尺検証が現実的に回る
- GUI のプレビュー待ちが過剰に長くならない

### 13.3 保守性

- パラメータの意味が明確
- 実験コードと本番コードが分離できる
- Colab 実験結果をローカルに還元しやすい


## 14. 評価方法

### 14.1 主観評価

- 高音で潰れないか
- メロディ輪郭が残るか
- 子音が減ったか
- 母音の持続が自然か
- 不自然な残響やノイズがないか

### 14.2 補助指標

- 高域エネルギー比
- 低域不要成分比
- F0 continuity
- harmonic retention
- onset harshness reduction


## 15. 実装フェーズ

### Phase 0: 現状固定と検証基盤

目的:
現状の品質を比較できるようにし、改善前後を検証しやすくする。

成果物:

- テスト用音声セット
- 比較用出力ファイル命名規則
- 検証コマンドの整理
- ベースラインの簡易評価メモ

### Phase 1: DSP コアの保守化

目的:
既存 `instrumentize_vocal()` の高音破綻を減らす。

成果物:

- high-pitch guard 強化
- harmonic protect 改善
- dynamic amount の見直し
- de-ess / softening の高音緩和

### Phase 2: Vibrato / contour 保護

目的:
高音 sustain とビブラートの保持を改善する。

成果物:

- fine contour 保護の追加
- 高音区間の dry / core 再注入強化
- 必要ならビブラート推定の試作

### Phase 3: Source-guided 再構成

目的:
元ボーカルの芯をより自然に戻し、潰れを減らす。

成果物:

- melody core 再注入の整理
- high melody core 再注入の整理
- dynamic dry blend の明文化

### Phase 4: GUI 統合

目的:
新しい保護パラメータを GUI から触れるようにする。

成果物:

- 元 VOCALS に戻すボタン
- 簡単モードの新パラメータ
- 詳細モードの新パラメータ
- 新プリセット

### Phase 5: Colab 実験基盤

目的:
重い比較実験や将来モデル検証のための入口を整える。

成果物:

- Colab セットアップ手順
- 実験用 notebook ひな形
- ローカルと Colab の処理一致確認

### Phase 6: 実験的ニューラル拡張

目的:
将来の軽量 ML 支援または DDSP 色付けを後段として試す。

成果物:

- optional backend の追加
- DSP 主体のパイプラインを壊さない統合方法


## 16. 実装タスク一覧

### 16.1 Phase 0

#### T01 ベースライン音声セットを固定する

- 既存の `vocals.wav` から短尺テスト区間を選定
- 高音で問題が出る区間を優先
- 比較対象ファイルを記録する

完了条件:

- 少なくとも 2 曲、各 1〜3 区間の検証対象が決まっている

#### T02 比較コマンドを標準化する

- CLI 実行コマンドを README または補助文書に整理
- 出力ファイル名にバージョン識別子を入れる

完了条件:

- 同じコマンドで before / after 比較が再現できる

#### T03 簡易評価スクリプトを用意する

- 高域比率
- 低域不要成分比
- F0 continuity

完了条件:

- 主観評価と合わせて毎回同じ指標を取れる


### 16.2 Phase 1

#### T04 `instrumentize_vocal()` の責務を整理する

- 解析部
- 抑制部
- 再注入部
- 仕上げ部

完了条件:

- 関数内の主要ブロックが読みやすく整理されている

#### T05 high-pitch guard を再設計する

- 高音判定閾値
- sustain との組み合わせ
- 補間ギャップの扱い

完了条件:

- 高音フレーズで guard が途切れにくい

#### T06 harmonic protect を拡張する

- 保護倍音の本数
- 帯域幅
- guard との連動

完了条件:

- 高音倍音の削れが現状より減る

#### T07 suppression を高音時に弱める

- dynamic amount
- dynamic breath
- dynamic consonant
- dynamic blur
- dynamic tone

完了条件:

- 高音で effect がかかりすぎない

#### T08 仕上げ softening の高音緩和を調整する

- de-ess 的 lowpass 混合を guard に連動

完了条件:

- 高音の抜けが過度に消えない


### 16.3 Phase 2

#### T09 vibrato 保護の簡易実装を入れる

- F0 微細変動の保持
- high sustain 時の contour 重視

完了条件:

- ビブラートが平坦化しにくい

#### T10 high dry blend を動的制御する

- 高音 sustain では dry を強める
- 子音が戻りすぎない上限を設ける

完了条件:

- 高音の輪郭保持が主観的に改善する

#### T11 melody core 再注入量を分離制御する

- low/mid core
- high core
- dry

完了条件:

- 何をどこで戻しているかが制御しやすい


### 16.4 Phase 3

#### T12 source-guided resynthesis lite を整理する

- 元音声を戻すロジックを段階化
- mix の意図を明確化

完了条件:

- 高音で潰れた時の逃げ道がコード上で明確

#### T13 low-end cleanup の副作用を見直す

- 母音の痩せすぎ防止
- 不要低域だけ除去

完了条件:

- 低域 cleanup が高音保持を邪魔しない


### 16.5 Phase 4

#### T14 `元 VOCALS に戻す` を GUI に追加する

- `stems_original["vocals"]` を復元
- 状態更新
- UI 表示更新

完了条件:

- DDSP などを適用後でも original に戻せる

#### T15 簡単モード用パラメータを追加する

- 発音抑制
- 高音保護
- 母音保持

完了条件:

- 初心者でも方向性を触りやすい

#### T16 詳細モードに新パラメータを追加する

- high guard
- high dry
- high core
- vibrato preserve

完了条件:

- 現在の実装要素を GUI から制御できる

#### T17 新プリセットを作る

- 保守的
- 標準
- より楽器寄り

完了条件:

- 3 種類以上の出発点がある


### 16.6 Phase 5

#### T18 Colab 実験導線を追加する

- セットアップスクリプト
- 実験手順
- notebook ひな形

完了条件:

- Colab 再起動後でも比較的早く復旧できる

#### T19 ローカル/Colab 一致確認を行う

- 同じ入力
- 同じパラメータ
- 出力差の確認

完了条件:

- 誤差の理由を説明できる


### 16.7 Phase 6

#### T20 DDSP 後段色付けを optional 化する

- DSP 主体を崩さない
- backend 選択を明確にする

完了条件:

- DDSP を使わなくても本命機能が成立する

#### T21 軽量 ML 補助の検証枠を作る

- vibrato 推定器
- articulation 推定器
- structural noise 推定器

完了条件:

- DSP と置き換えず、補助として使える


## 17. 優先順位

### 最優先

- T01
- T02
- T04
- T05
- T06
- T07
- T10
- T14

### 中優先

- T03
- T08
- T09
- T11
- T15
- T16
- T17
- T18

### 低優先

- T19
- T20
- T21


## 18. MVP 定義

以下を満たした時点で第一段階の完成とする。

- 高音の潰れが現行より明確に改善している
- GUI で `元 VOCALS に戻す` が使える
- GUI から高音保護の方向を調整できる
- 短尺検証素材で before / after 比較ができる
- ローカルで安定運用できる
- Colab 実験入口が用意されている


## 19. リスク

- 高音保護を強めすぎると子音も戻りやすい
- dry 再注入を増やすと歌詞感が戻る
- パラメータが増えすぎると GUI が複雑化する
- Colab 側の実験結果を本番コードに戻す際に差異が出る可能性がある


## 20. 直近の着手順

開発は次の順で進める。

1. Phase 0 の基盤を固める
2. `instrumentize_vocal()` の保守化を進める
3. 高音 dry / core / vibrato 保護を調整する
4. GUI に復元と保護パラメータを追加する
5. Colab 実験導線を整える


## 21. 関連ファイル

- [music_remix.py](/Users/gon/Documents/AI_Generated_Apps/music-remix-app/music_remix.py)
- [remix_gui.py](/Users/gon/Documents/AI_Generated_Apps/music-remix-app/remix_gui.py)
- [ddsp_flute_transfer.py](/Users/gon/Documents/AI_Generated_Apps/music-remix-app/ddsp_flute_transfer.py)
- [ddsp_setup.py](/Users/gon/Documents/AI_Generated_Apps/music-remix-app/ddsp_setup.py)
- [HANDOFF_NEXT_AI.md](/Users/gon/Documents/AI_Generated_Apps/music-remix-app/HANDOFF_NEXT_AI.md)
- [NEXT_AI_PROMPT.md](/Users/gon/Documents/AI_Generated_Apps/music-remix-app/NEXT_AI_PROMPT.md)
