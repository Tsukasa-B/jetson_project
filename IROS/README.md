# IROS/ — IROS投稿時（2026-02〜03）の成果物アーカイブ

**読み取り専用**。ここのファイル名は当時のまま変えていない（過去の解析スクリプト・
図の再現性を保つため）。新しい実験結果は `data/` に出力すること（ルート README 参照）。

v3再編（2026-09）で `deploy_results/modelA〜D/` 等のサブディレクトリはこの直下へ
フラット化されたが、**フォルダ名の文字自体（modelA→legacyA 等、DR/noDR）は変えていない**。
中身のファイル名・内容も無変更。

## ★ 旧モデル名 ⇔ 論文名の対応表

IROS時のリポジトリでは `models/modelA〜D.onnx` の A/B/C/D が
論文の A〜E と**一致していなかった**。ここのフォルダ名・ファイル名は旧名のままなので、
読むときは必ずこの表を通すこと。

| 旧名（このフォルダ内の表記） | obs次元 | 構造 | lookahead | **論文名** | 現在の場所 |
|---|---|---|---|---|---|
| modelA | 60 | LSTM | 1.0s | **C** | `models/IROS/modelC.onnx` |
| modelB | 35 | LSTM | 0.5s | **B** | `models/IROS/modelB.onnx` |
| modelC | 35 | **MLP** | 0.5s | **D** | `models/IROS/modelD.onnx` |
| modelD | 15 | LSTM | 0.1s | **A** | `models/IROS/modelA.onnx` |

つまり **`deploy_legacyC/` は論文の D（記憶なしMLP）の実機データ**であり、
**`deploy_legacyA/` は論文の C（lookahead 1.0s）** である。取り違え注意。

論文名の定義は sim側リポジトリ `analysis/harness/tb_curves.py::MODEL_LABELS` および
`analysis/eval/run_eval_matrix.py::MODEL_ENV_OVERRIDES` が正。

## 中身（v3再編後のフォルダ名）

| ディレクトリ | 内容 | 旧パス（v2以前） |
|---|---|---|
| `legacy_code/` | IROS時のデプロイスクリプト（`run_rl_deploy_midi.py` ほか）。参照用 | `legacy/` |
| `deploy_legacyA/`〜`deploy_legacyD/` | ポリシー実機ラン結果（旧modelA〜D、上表参照） | `deploy_results/modelA〜D/` |
| `deploy_DRfolder/` / `deploy_noDRfolder/` | ドメインランダム化の有無比較（★下記の既知の問題を参照） | `deploy_results/DR/` / `noDR/` |
| `verification/` | 通信・センサの検証ログ（`check_v2_*.csv`） | `logs_verification/` |
| `measured/` | 指令信号CSV再生時の実測データ（`data_exp*_*.csv`） | 同じ（未移動） |
| `figures_paper/` | 論文Fig.4〜7（旧`analysis_results/figures_iros/` + 旧`figures/`のFig.7を統合） | `analysis_results/figures_iros/`, `figures/` |
| `figures_analysis/` | その他の解析図（png） | `analysis_results/` 直下 |
| `sysid_20260111/` | システム同定の生ログ一式 | `collect_data_raw/raw_20260111/` |
| `sysid_misc/` | システム同定の雑多な生ログ・図 | `collect_data_raw/` 直下 |

`IROS/models_pt/`（旧 `models/IROS/pt_archive/`）に当時の `.pt` チェックポイントがある
（**ファイル名は未変更**。学習ログとの対応を保つため）。デプロイに使うのは ONNX 版のみ。

## 既知の要確認点（未修正・判断保留）

- **`deploy_DRfolder/` と `deploy_noDRfolder/` はフォルダ名とファイル名のラベルが
  逆転している可能性がある。** 例えば `deploy_DRfolder/` 配下に
  `..._noDR_..._1771224404.csv` という名前のファイルが、`deploy_noDRfolder/` 配下に
  `..._DR_..._1771224952.csv` という名前のファイルが入っている
  （single/double系のタイミング）。どちらのラベルが正しいかは判断していない。
  フォルダ名は元のまま（`DR`/`noDR`）保持してあるので、使う前に個々のファイル名の
  `_DR_` / `_noDR_` 表記と突き合わせて確認すること。
- **`deploy_legacyA/`（旧 `deploy_results/modelA/`）に別モデルのファイルが1つ混入している。**
  `deploy_test_single4_bpm60_modelB_DR_1499_03-1_lookahead5_1771319228.csv` という
  `modelB` 名のファイルが `deploy_legacyA/` に入っている。フォルダ単位で移動しただけで
  中身の並べ替えはしていないので、集計時はファイル名側のモデル表記を優先すること。
