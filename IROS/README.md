# IROS/ — IROS投稿時（2026-02〜03）の成果物アーカイブ

**読み取り専用**。ここのファイル名は当時のまま変えていない（過去の解析スクリプト・
図の再現性を保つため）。新しい実験結果は `results/` に出力すること。

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

つまり **`deploy_results/modelC/` は論文の D（記憶なしMLP）の実機データ**であり、
**`deploy_results/modelA/` は論文の C（lookahead 1.0s）** である。取り違え注意。

論文名の定義は sim側リポジトリ `analysis/harness/tb_curves.py::MODEL_LABELS` および
`analysis/eval/run_eval_matrix.py::MODEL_ENV_OVERRIDES` が正。

## 中身

| ディレクトリ | 内容 |
|---|---|
| `legacy/` | IROS時のデプロイスクリプト（`run_rl_deploy_midi.py` ほか）。参照用 |
| `deploy_results/` | ポリシー実機ラン結果。`DR/` `noDR/` はドメインランダム化の有無比較 |
| `logs_verification/` | 通信・センサの検証ログ（`check_v2_*.csv`） |
| `measured/` | 指令信号CSV再生時の実測データ（`data_exp*_*.csv`） |
| `analysis_results/` | 解析図。`figures_iros/` が論文Fig.4〜6 |
| `figures/` | Fig.7 (Sim2Real timing error) |
| `collect_data_raw/` | 学習データ収集の生ログ |

`models/IROS/pt_archive/` に当時の `.pt` チェックポイントがある（**ファイル名は未変更**。
学習ログとの対応を保つため）。デプロイに使うのは ONNX 版のみ。
