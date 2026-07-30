# jetson_project — Porcaro Robot 実機デプロイ

Jetson Orin Nano から MicroLabBox 経由で PAM駆動ドラミングロボットを動かし、
学習済みポリシー（RL）を実機で走らせるためのリポジトリ。

```
Jetson (Docker) ──serial 230400bps, BigEndian, header FFFF──> MicroLabBox (Simulink)
   送信 50Hz : double x3 = 圧力指令 [DF, F, G] (MPa)            → 電空レギュレータ → PAM → ロボット
   受信 200Hz: double x7 = [圧力3, 手首角, グリップ角, flag, force_N]
```

> ROS2 は使っていない。Docker イメージが `my_ros2_pytorch_container` なだけで、
> 制御は素の Python（pyserial + threading）。

---

## ディレクトリ構成

```
src/            実行コード（デプロイ本体）
  deploy_policy.py          ポリシー実行。全モデル共通。manifest駆動
  model_registry.py         manifest読み込み・ONNX整合性検証・framestack・LSTM隠れ状態
  microlabbox.py            シリアル送受信（+ 実機なしドライラン用の疑似デバイス）
  midi_rhythm_generator.py  MIDI → 目標力軌道（sim側と等価）

models/
  manifest.yaml   ★モデル定義の唯一の真実（論文名 A〜E ⇔ ファイル ⇔ パラメータ）
  IROS/           IROS投稿時にデプロイした学習済みモデル（論文名にリネーム済み）
  RAL/            RA-L用にexportするモデル置き場 → models/RAL/README.md

songs/          入力MIDI（test_*, gmd_*）
test_signals/   ポリシー無しの指令信号CSV（exp1〜8）
tools/          データ収集・信号生成・疎通確認・検証スクリプト
analysis/       図の生成・解析
results/        ★これからの実機ログ出力先（results/<group>/model<X>/）
IROS/           IROS時の成果物アーカイブ（読み取り専用）→ IROS/README.md
```

---

## モデル名 ⇔ 論文名（重要）

論文（RA-L）の A〜E に**一致させてある**。IROS時のリポジトリでは名前がずれていたので注意。

| キー | 構造 | lookahead | frame_stack | obs次元 | 旧ファイル名 |
|---|---|---|---|---|---|
| `A` | LSTM | 0.1s | 1 | 15 | modelD.onnx |
| `B` | LSTM | 0.5s | 1 | 35 | modelB.onnx |
| `C` | LSTM | 1.0s | 1 | 60 | modelA.onnx |
| `D` | MLP  | 0.5s | 1 | 35 | modelC.onnx |
| `E` | MLP  | 0.5s | 5 | 175 | （IROS時に存在せず） |

`obs次元 = (10 + lookahead/0.02) × frame_stack`

---

## 起動

```bash
# 0. 時刻合わせ（Jetsonは時計が狂う。ログ名がunixtimeなので先にやる）
sudo date -s "$(wget -qSO- --max-redirect=0 google.com 2>&1 | grep Date: | cut -d' ' -f5-8)Z"

# 1. FTDI(RATOCケーブル)認識のおまじない ※ホスト側
sudo modprobe ftdi_sio
echo 0584 b050 | sudo tee /sys/bus/usb-serial/drivers/ftdi_sio/new_id

# 2. コンテナ起動（ロボット接続時）
sudo docker run --runtime nvidia -it --rm --network host \
    --device /dev/ttyUSB0 \
    -v /home/okuilab/jetson-containers/data/jetson_project:/data/jetson_project \
    -w /data/jetson_project \
    my_ros2_pytorch_container:latest bash

# ロボットを繋がずコード開発だけする場合は --device 行を外す
```

---

## 使い方

```bash
# 登録済みモデルの一覧と整合性
python3 src/deploy_policy.py --list

# 実機なしのドライラン（制御ループ・obs構築・推論を通しで確認）
python3 src/deploy_policy.py --model IROS/B --midi songs/test_single4_bpm60.mid --mock

# 実機で走らせる
python3 src/deploy_policy.py --model RAL/E --midi songs/gmd_02_mid_bpm105.mid --trial 1

# 駆動せずに目標軌道だけ確認
python3 src/deploy_policy.py --model IROS/B --midi songs/test_single4_bpm60.mid --verify

# ポリシー無しで指令信号CSVを再生（sim-real同定用）
python3 tools/run_signal_playback.py exp2_step_response.csv
```

出力は `results/<group>/model<X>/deploy_<曲>_<group>-<X>_trial<NN>_<unixtime>.csv` と、
同名の `.json`（モデル・trial番号・パケット受信率・git rev などの実行条件）。

### モデルを追加・差し替えたら必ず

```bash
python3 tools/check_models.py    # manifest と ONNX の整合性（全モデル）
python3 tools/parity_test.py     # 観測構築が IROS時のコードとビット一致するか + framestack順序
```

`manifest.yaml` の宣言と ONNX の実体が食い違う場合は**起動時にエラーで停止する**。
モデルを変えてもスクリプトを書き換える必要はない。

---

## データ収集（学習用・同定用）

```bash
# 学習データ収集（別プロトコル: baud 115200 / 受信6要素）
sudo python3 tools/collect_real_data.py --mode train        # 3〜5分
sudo python3 tools/collect_real_data.py --mode step         # 30秒（ステップ応答）
sudo python3 tools/collect_real_data.py --mode hysteresis   # 1分（ゆっくり5往復）
```

キー操作: `s` 開始/再開 ／ `p` 一時停止（脱力。コンプレッサ圧が落ちたら押す。segment_id が +1 される）
／ `q` 終了＆CSV保存。

---

## 実装上の注意（sim との差分）

- **角速度クリップ**: 実機は `qd = clip(Δq/0.02, ±20 rad/s)`。sim側にこのクリップは無い
  （実機のエンコーダノイズ対策）。パリティを厳密に取りたい場合はここを揃えること。
- **prev_action**: 実機・sim ともに **clip後**の action を次の観測に入れる（一致済み）。
- **frame stack 初期値**: sim の `_reset_idx` と同じくゼロ埋め。並びは `[最古 … 最新]`。
- **ログの time 列**: 受信が完全に200Hzである前提の再構成時刻（`arange(N)/200`）。
  実測受信時刻は `t_recv_rel` 列にある。±30ms 判定を出す前に必ず両者を比較し、
  `.json` の `packet_yield` が 1.0 近いことを確認する。
- **LSTMの隠れ状態**は実行開始時にゼロ初期化（1曲1プロセスが前提）。
- **ボーレートが用途で違う**: デプロイ 230400 ／ `tools/collect_real_data.py` 115200 ／
  `tools/serial_test/` 115200。MicroLabBox側 Simulink の設定と対で切り替わる。
