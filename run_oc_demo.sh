#!/bin/bash
# OC デモGUI 起動スクリプト（jetson_project のルートに置く）
#
#   ./run_oc_demo.sh --mock       実機なしのシミュレーション（練習・画面確認用）
#   ./run_oc_demo.sh --verify     実シリアルに繋ぐが圧力0だけ送る（ロボットは動かない）
#   ./run_oc_demo.sh              実機モード（/dev/ttyUSB0, models/IROS/modelB.onnx）
#   OC_MODEL=IROS/modelC.onnx ./run_oc_demo.sh    モデルを変える
#
# コンテナは --network host で起動しているので -p は不要。
# ブラウザから  http://localhost:8080/  （別PCからは http://<JetsonのIP>:8080/）

set -euo pipefail
cd "$(dirname "$0")"

MODEL="${OC_MODEL:-IROS/modelB.onnx}"
PORT_NAME="${OC_SERIAL_PORT:-/dev/ttyUSB0}"
HTTP_PORT="${OC_HTTP_PORT:-8080}"

ARGS=(--midi-dir midi --models-dir models --http-port "$HTTP_PORT")

if [[ "${1:-}" == "--mock" ]]; then
  echo "=== モックモード（実機に繋ぎません）==="
  ARGS+=(--mock)
else
  VERIFY=""
  if [[ "${1:-}" == "--verify" || "${1:-}" == "--no-drive" ]]; then
    VERIFY="--no-drive"
  fi
  if [[ ! -e "$PORT_NAME" ]]; then
    echo "!! $PORT_NAME がありません。ホスト側で以下を実行しましたか？"
    echo "   sudo modprobe ftdi_sio"
    echo "   echo 0584 b050 | sudo tee /sys/bus/usb-serial/drivers/ftdi_sio/new_id"
    echo "   （実機なしで画面だけ見るなら ./run_oc_demo.sh --mock）"
    exit 1
  fi
  ARGS+=(--port-name "$PORT_NAME" --model "$MODEL")
  if [[ -n "$VERIFY" ]]; then
    ARGS+=("$VERIFY")
    echo "=== 駆動なし検証モード  port=$PORT_NAME  model=$MODEL ==="
    echo "    圧力は常に0を送ります。ロボットは動きません。"
  else
    echo "=== 実機モード  port=$PORT_NAME  model=$MODEL ==="
    echo "    ロボットが動きます。周囲の安全を確認してください。"
  fi
fi

echo "ブラウザで http://localhost:$HTTP_PORT/ を開いてください"
exec python3 -m oc_demo.server "${ARGS[@]}"
