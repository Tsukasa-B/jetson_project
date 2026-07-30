#!/usr/bin/env bash
# =============================================================================
# jetson_project ディレクトリ再編 v2  (2026-07-30)
#
#  - モデル名を論文(RA-L)の A〜E に一致させる
#  - IROS時の成果物を IROS/ 配下にアーカイブ
#  - 実行コードを src/ と tools/ に整理
#
# 注意: モデル名は A→C→D→A の3すくみで入れ替わるため、必ず一時名を経由すること。
#       リポジトリのルートで実行する。git 管理下であることが前提。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[migrate] repo root = $(pwd)"

if [ -n "$(git status --porcelain)" ]; then
  echo "[migrate] ERROR: 作業ツリーに未コミットの変更があります。先にcommitかstashしてください。" >&2
  exit 1
fi

mv_git() { # 存在する場合のみ git mv
  if [ -e "$1" ]; then git mv -k "$1" "$2"; else echo "  (skip: $1 が無い)"; fi
}

# -----------------------------------------------------------------------------
# 0. 掃除
# -----------------------------------------------------------------------------
echo "[migrate] 0. __pycache__ を削除"
find . -name '__pycache__' -type d -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
git rm -r --cached --ignore-unmatch -q __pycache__ deploy/__pycache__ 2>/dev/null || true

# -----------------------------------------------------------------------------
# 1. ディレクトリ作成
# -----------------------------------------------------------------------------
echo "[migrate] 1. 新ディレクトリを作成"
mkdir -p src tools/serial_test models/IROS/pt_archive models/RAL \
         test_signals results/RAL analysis \
         IROS/legacy IROS/deploy_results IROS/measured IROS/figures IROS/collect_data_raw

# -----------------------------------------------------------------------------
# 2. ★モデルのリネーム（3すくみ。必ず一時名を経由）
#      旧 modelD (obs15,  LSTM lh0.1) -> 論文 A
#      旧 modelB (obs35,  LSTM lh0.5) -> 論文 B  (唯一そのまま)
#      旧 modelA (obs60,  LSTM lh1.0) -> 論文 C
#      旧 modelC (obs35,  MLP  lh0.5) -> 論文 D
# -----------------------------------------------------------------------------
echo "[migrate] 2. モデルを論文名にリネーム"
git mv models/modelA.onnx models/__tmp_C.onnx
git mv models/modelB.onnx models/__tmp_B.onnx
git mv models/modelC.onnx models/__tmp_D.onnx
git mv models/modelD.onnx models/__tmp_A.onnx

git mv models/__tmp_A.onnx models/IROS/modelA.onnx
git mv models/__tmp_B.onnx models/IROS/modelB.onnx
git mv models/__tmp_C.onnx models/IROS/modelC.onnx
git mv models/__tmp_D.onnx models/IROS/modelD.onnx

echo "[migrate] 2b. .pt はアーカイブへ（名前は変えない＝学習ログとの対応を保つ）"
for f in models/*.pt; do [ -e "$f" ] && git mv "$f" models/IROS/pt_archive/; done

# -----------------------------------------------------------------------------
# 3. 実行コードを src/ へ
# -----------------------------------------------------------------------------
echo "[migrate] 3. src/ を構成"
mv_git midi_rhythm_generator.py            src/midi_rhythm_generator.py
mv_git run_rl_deploy_midi.py               IROS/legacy/run_rl_deploy_midi.py
mv_git run_rl_deploy_midi_mlp.py           IROS/legacy/run_rl_deploy_midi_mlp.py
mv_git deploy/run_rl_deploy.py             IROS/legacy/run_rl_deploy.py
mv_git deploy/midi_rhythm_generator.py     IROS/legacy/midi_rhythm_generator_dup.py
rmdir deploy 2>/dev/null || true

# -----------------------------------------------------------------------------
# 4. 補助ツールを tools/ へ
# -----------------------------------------------------------------------------
echo "[migrate] 4. tools/ を構成"
mv_git IROS/run_iros_validation.py                 tools/run_signal_playback.py
mv_git IROS/verification/generate_test_signals.py  tools/generate_test_signals.py
rmdir IROS/verification 2>/dev/null || true

mv_git collect_data/collect_real_data.py    tools/collect_real_data.py
mv_git collect_data/collect_data_custom.py  tools/collect_data_custom.py
mv_git collect_data/verify_sensors.py       tools/verify_sensors.py
mv_git collect_data/check_data.py           tools/check_data.py
mv_git collect_data/serial_test.py          tools/serial_test/serial_test_collect.py
mv_git collect_data/README_collect_data.md  tools/README_collect_data.md

for f in serial_test/*.py; do [ -e "$f" ] && git mv "$f" "tools/serial_test/$(basename "$f")"; done
rmdir serial_test 2>/dev/null || true

mv_git test_signals/hello_jetson.py         tools/serial_test/hello_jetson.py
mv_git test_signals/test_receiver.py        tools/serial_test/test_receiver.py
mv_git test_signals/inference_test.py       tools/inference_test.py
mv_git test_signals/jetson_inference.py     tools/jetson_inference.py
mv_git test_signals/run_dummy_inference.py  tools/run_dummy_inference.py
mv_git test_signals/repair_csv_timestamps.py tools/repair_csv_timestamps.py

# -----------------------------------------------------------------------------
# 5. 解析スクリプトを analysis/ へ
# -----------------------------------------------------------------------------
echo "[migrate] 5. analysis/ を構成"
mv_git create_fig7.py                       analysis/create_fig7.py
mv_git collect_data/analysis_pneumatic.py   analysis/analysis_pneumatic.py

# -----------------------------------------------------------------------------
# 6. 入力信号CSVは test_signals/ に一本化（IROS/test_signals の exp*.csv が正）
# -----------------------------------------------------------------------------
echo "[migrate] 6. 指令信号CSVを test_signals/ に集約、実測データは IROS/measured/ へ"
for f in IROS/test_signals/exp*.csv; do
  [ -e "$f" ] || continue
  b=$(basename "$f")
  if [ -e "test_signals/$b" ]; then git rm -q "test_signals/$b"; fi
  git mv "$f" "test_signals/$b"
done
for f in IROS/test_signals/data_*.csv; do [ -e "$f" ] && git mv "$f" IROS/measured/; done
rmdir IROS/test_signals 2>/dev/null || true
for f in test_signals/*.csv; do
  b=$(basename "$f")
  case "$b" in exp*) ;; *) git mv "$f" IROS/measured/ ;; esac
done

# -----------------------------------------------------------------------------
# 7. IROS時の成果物をアーカイブ（ファイル名は変えない＝過去の解析の再現性を保つ）
# -----------------------------------------------------------------------------
echo "[migrate] 7. 成果物を IROS/ にアーカイブ"
for d in deploy_results/*/; do
  [ -d "$d" ] || continue
  git mv "$d" "IROS/deploy_results/$(basename "$d")"
done
rmdir deploy_results 2>/dev/null || true

mv_git logs_verification                    IROS/logs_verification
mv_git analysis_results                     IROS/analysis_results
mv_git Fig7_Sim2Real_Timing_Error_Colored.svg IROS/figures/Fig7_Sim2Real_Timing_Error_Colored.svg
mv_git data_exp1_async_1770964289.csv       IROS/measured/data_exp1_async_1770964289.csv

mv_git "collect_data/dataてきとーてきとー"   IROS/collect_data_raw/raw_20260111
mv_git collect_data/data_characteristics_exp1_async_1771481712.csv IROS/collect_data_raw/
mv_git collect_data/dynamics_heatmap_recalc.png IROS/collect_data_raw/
rmdir collect_data 2>/dev/null || true

echo
echo "[migrate] 完了。 git status で差分を確認してください。"
echo "[migrate] このあと models/manifest.yaml と src/*.py を配置し、"
echo "[migrate]   python3 tools/check_models.py"
echo "[migrate] を実行して全モデルの整合性チェックを通すこと。"
