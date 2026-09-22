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

## ディレクトリ構成（v3再編、2026-09）

全ファイルのパスはルートから3階層以内（`dir/dir/file` まで）。

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
tools/          データ収集・信号生成・疎通確認・検証スクリプト（serial_test/ 含む）
analysis/       図の生成・解析

data/           ★研究の実機ログ（これからのRUN）。1セッション1フォルダ、中はフラット
  ral_YYYYMMDD/      デプロイ実行の出力先（下記「出力先の既定値」参照）
  ral_YYYYMMDD_b/    同日に複数セッションがある場合の2つ目以降
  ral_quarantine/    力センサ死亡が疑われ集計から除外したラン（IROS期モデルで実施したもの含む）

out/            集計CSV・図（analysis/ の出力）
  ral/               summary_*.csv, strikes_*.csv など

IROS/           ★IROS投稿時点の凍結アーカイブ（読み取り専用）→ IROS/README.md
  deploy_legacyA〜D/, deploy_DRfolder/, deploy_noDRfolder/  実機デプロイ結果（旧フォルダ名）
  sysid_20260111/, sysid_misc/    システム同定の生ログ
  verification/      通信・センサ検証ログ
  measured/           指令信号CSV再生時の実測データ
  figures_paper/, figures_analysis/  論文図・解析図
  legacy_code/        当時のデプロイスクリプト（参照用）
  models_pt/          当時の.ptチェックポイント（models/IROS/*.onnx がデプロイ用実体）

oc_demo/        オープンキャンパスデモ（コード不変）
  docs/, dist/        旧ルート直下にあった設計メモ・配布物
midi/           OCデモ用MIDI（`oc_demo/tools/make_demo_midi.py` の出力先。動かさない）
run_oc_demo.sh  OCデモ起動スクリプト（Jetsonで直接叩く。動かさない）
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

出力先（既定）は `data/<group小文字>_<実行開始日 YYYYMMDD>/`（例: `data/ral_20260922/`）。
同日に複数セッションを走らせて既存フォルダと衝突する場合は、手で `_b` `_c` ... を付けて
退避してから次のセッションを始めること（`data/ral_20260731` 〜 `_c` の前例を参照）。
`--out` で明示的に指定すれば既定値は使わない。
ファイル名は `deploy_<曲>_<group>-<X>_trial<NN>_<unixtime>.csv` と、
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

---

## Jetson側での反映手順（v3再編の取り込み）

1. **pull前に、Jetson上の未コミットのログ（`results/` 以下など、v2時代の場所も含む）を
   commit & push しておく。** 再編PRをmainにマージ後、Jetson側で `git pull` すると
   `results/` は消えて `data/` に置き換わる。ローカルにしか無いログがあると
   `git mv` の履歴と衝突・消失する可能性がある。
2. `git pull` 後、モデル一覧が正しく引けるか確認:
   ```bash
   python3 src/deploy_policy.py --list
   ```
3. `--mock` で出力先が新しい `data/<group>_<今日>/` に出ることを確認:
   ```bash
   python3 src/deploy_policy.py --model RAL/E --midi songs/test_single8_bpm120.mid --mock
   ```
4. `--resume` を使うバッチが再開できるか確認（Windows開発機では onnxruntime/torch/pyserial/mido
   が無くこの部分は未検証。**Jetson側で必ず確認すること**）:
   ```bash
   python3 tools/run_experiment_batch.py --plan tools/experiment_plan.yaml --dry_run --resume
   ```
5. `run_oc_demo.sh` / `oc_demo/` は位置・中身とも変更していないので、そのまま動くはず。
   念のため `./run_oc_demo.sh` が `bad interpreter` エラーになる場合は `bash run_oc_demo.sh`
   で実行する（リポジトリ全体にCRLFが混入しているファイルがあり、`run_oc_demo.sh` 自体は
   今回のスコープ外として中身を変更していないため）。
