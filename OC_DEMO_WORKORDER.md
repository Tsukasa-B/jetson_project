# OCデモGUI（oc_demo）実装指示書

作成: 2026-07-30 / 改訂: 依存ライブラリをゼロにした版。
クラウド側で実装・モック検証済み、Jetson実機での結線は未実施。

来場者が **曲を選んで START を押すだけ** でロボットが演奏し、
**MIDI譜面と実際の打点タイミングが同じ画面に流れる** ブラウザUIです。

---

## 0. まず結論：Jetson側でやること

1. `./run_oc_demo.sh --mock` で画面が出ることを確認（実機不要・**pip install も不要**）
2. **§2.5 の手順で段階的に**実機へ上げる（受信診断 → モデル確認 → 駆動なし通し → 実駆動）
3. その途中で **§3 の「結線点3つ」を潰す**

§3 を飛ばすと「動いてはいるが観測ベクトルが学習時とズレている」状態になり、
**静かに間違います**。実機に触る前に必ず確認してください。

### 追加インストールは不要です

Jetsonのコンテナは pip の索引が jetson-containers 専用のもの
（`jetson.webredirect.org/jp6/cu126`）に固定されていて外部PyPIに出られないため、
このデモは **Python標準ライブラリ + 既存の numpy / onnxruntime / pyserial だけ** で
動くように書いてあります。

| やりたいこと | 使ったもの |
|---|---|
| HTTPサーバ | `http.server`（標準） |
| テレメトリ配信 | SSE / `EventSource`（標準。WebSocketは使いません） |
| MIDI読み込み | `oc_demo/midiparse.py`（自前。mido不要） |
| `models/manifest.yaml` | PyYAMLがあれば読む。無くてもモデル名だけで動く |

`pip install -r oc_demo/requirements.txt` は**実行しないでください**（中身は説明文だけです）。

---

## 1. 配置

`jetson_project` のルート直下に展開します。既存ファイルは1つも書き換えません。

```
jetson_project/
├── run_rl_deploy_midi.py      ← 既存。触らない
├── midi_rhythm_generator.py   ← 既存。oc_demo から参照する
├── models/                    ← 既存
├── midi/                      ← MIDI置き場（無ければ作る）
│   └── labels.json            ← 表示名（任意）
├── run_oc_demo.sh             ← 追加
└── oc_demo/                   ← 追加
    ├── __init__.py
    ├── adapter.py    ★ 既存コードとの結線点。最初に読む
    ├── check_serial.py 受信の健康診断（送信なし）
    ├── check_models.py ONNXの入出力確認
    ├── midiparse.py    標準MIDIファイルの最小パーサ（mido の代わり）
    ├── score.py        MIDI → 譜面 + 目標力軌道
    ├── plant.py        シリアルI/O（実機）/ 簡易シミュレータ（モック）
    ├── policy.py       ONNX推論 / モデル無し用スクリプト動作
    ├── runner.py       50Hz制御ループ + テレメトリ
    ├── server.py       http.server + SSE
    ├── requirements.txt（依存なしの説明のみ）
    └── static/index.html   画面（1ファイル）
```

## 2. 起動モードは3段階

| モード | コマンド | 何が起きるか |
|---|---|---|
| モック | `./run_oc_demo.sh --mock` | シリアルに繋がない。画面だけ |
| **駆動なし検証** | `./run_oc_demo.sh --verify` | 実シリアルに繋いで実センサを読み、ONNX推論もするが、**送る圧力は常に0**。ロボットは動かない |
| 実機 | `./run_oc_demo.sh` | 本番。ロボットが動く |

ブラウザで `http://localhost:8080/`。別PC/タブレットからは `http://<JetsonのIP>:8080/`。
コンテナは `--network host` なので `-p` の追加は不要です。
複数のブラウザから同時に開いて構いません（全画面が同じ演奏をミラーします）。
画面右上のバッジに「デモ / 実機・駆動なし検証 / 実機モード」が出るので、
いまどれで動いているかは必ずそこで確認してください。

会場では Chromium を全画面で:
```bash
chromium-browser --kiosk --app=http://localhost:8080/
```

---

## 2.5 実機立ち上げ手順（この順番で）

いきなり `./run_oc_demo.sh` を打たないでください。下の順に上げると、
問題が起きたときに原因が1つに絞れます。

### Step 0. デバイスを見えるようにする（ホスト側）
```bash
sudo modprobe ftdi_sio
echo 0584 b050 | sudo tee /sys/bus/usb-serial/drivers/ftdi_sio/new_id
ls -l /dev/ttyUSB0
sudo date -s "..."     # Jetsonは時計が狂う。ログ名がunixtimeなので先に合わせる
```

### Step 1. 受信の健康診断（送信なし・ロボットは動かない）
MicroLabBox の Simulink を動かした状態で、**スティックが打面に触れていない**ことを
確認してから:
```bash
python3 -m oc_demo.check_serial --seconds 10
```
見るのは2つだけです。

- **受信レートが 200Hz 近くあるか**。低いとログ時刻の前提が崩れ、±30ms判定が嘘になります
- **force_N の中央値が 0 付近か**。ここが `-20` 付近なら §3-3 の未解決事項が再発しています。
  **先に潰してください**。この数字がズレたまま進むと、以降の判定が全部意味を失います

### Step 2. モデルの入出力を確認（ロボットは動かない）
```bash
ls models/ models/IROS/
python3 -m oc_demo.check_models
```
- 使う予定のモデルの **obs次元 / lookahead / RNN有無** が意図どおりか
- LSTMモデルで「RNN入力はあるのに出力に状態が返っていない」と警告が出たら要注意。
  そのままだと毎ステップ h/c がゼロ、つまり**記憶なしで動いてしまいます**（静かに間違う）。
  `run_rl_deploy_midi.py` のRNN分岐と入出力名の対応を確認してください

### Step 3. 駆動なしで通し（ロボットは動かない）
```bash
./run_oc_demo.sh --verify
```
ブラウザで1曲スタートし、次を確認します。

- 右上バッジが「実機・駆動なし検証」になっている
- 画面左下に `hardware / onnx · obs 35次元 · 力ゼロ点 x.xxxN` が出る
- 関節角と力センサが**実機の値**として動いている（手で軽く動かすと反応する）
- 圧力バーは「出したかった値」が動くが、ロボットは静止したまま
- `curl -s localhost:8080/api/health` で `adapter.target_force` が `fallback:` **でない**（§3-1）
- 最後まで例外なく `finished` になる

### Step 4. obsのパリティ確認（§3-2）
ここまで通ってから、`tools/parity_test.py` と同じ要領で
`run_rl_deploy_midi.py` 経路と `oc_demo` 経路の obs が一致することを確認します。

### Step 5. 実駆動
```bash
./run_oc_demo.sh
```
最初は**遅いテンポの1曲**から。開始前に:

- 周囲に人・物がないか
- 物理の非常停止（元栓/電源）に手が届く位置にいるか
- 画面の「ストップ（緊急停止）」が押せる状態か → **まず押して効くことを確認**

問題なければテンポと曲を広げ、破綻するBPMを把握して `--bpm-max` を絞ります。

---

## 3. ★結線点3つ（実機前に必ず）

### 3-1. 目標力軌道が既存実装と一致しているか

`oc_demo/adapter.py::build_target_force()` は `midi_rhythm_generator` から
以下の名前の関数を順に探します。

```
generate_target_force_trajectory / generate_target_force /
midi_to_target_force / build_target_force
```

**どれとも一致しなければ `score.py` のフォールバック実装（raised-cosine パルス）が
使われます。** これは sim の学習時と別物です。

確認方法:
```bash
curl -s localhost:8080/api/health | python3 -m json.tool
# adapter.target_force が "fallback:..." なら未結線
```
実関数名を確認して `adapter.py` の候補リストに追加するか、直接呼び出しに書き換えてください。
引数が `(path, bpm=..., n_steps=..., dt=...)` と違う場合も同様です。

### 3-2. 観測ベクトルが `run_rl_deploy_midi.py` と一致しているか

`adapter.py::build_obs()` は調査メモ §3 の仕様で実装してあります。

```
obs = [q_wrist, q_grip, qd_wrist, qd_grip, prev_action(3),
       sin(phase), cos(phase), bpm/180, lookahead(L)]     L = obs_dim - 10
```

- `prev_action` は **clip後** の値を入れています（実機の既存実装と同じ）
- `qd` は 角度差分/0.02 を ±20 rad/s clip
- `lookahead` は `target_force` の最大値で正規化（0〜1）
- `LOOKAHEAD_STEPS` は **ONNXの入力次元から自動導出**（ハードコードしていません）

`tools/parity_test.py` と同じ要領で、同一のセンサ系列を
`run_rl_deploy_midi.py` の経路と `oc_demo` の経路に通し、obs がビット一致することを
確認してください。ズレていれば `adapter.build_obs` を直します。

### 3-3. 力センサのゼロ点

`jetson_refactor_v2_20260730.md` の未解決事項です（無負荷で `force_N ≈ -20.0`、
IROS時は -0.1〜-0.3）。**±30ms判定に直結します。**

oc_demo は演奏開始直前に50フレームを読んで中央値をゼロ点として引きます
（`RunConfig.zero_force_samples`）。画面左下と `/api/health` に実測値が出ます。

- これは **その瞬間が無負荷である** ことが前提です。スティックが打面に触れた状態で
  STARTすると、そのぶんが丸ごとオフセットになります
- ゼロ点が -20 N 付近のままなら、原因（配線/アンプ/Simulink側スケール）を
  先に潰してください。デモの数字が全部意味を失います

---

## 4. このデモが**やらない**こと

- **論文E（frame stacking, obs 175次元）は非対応**です。読み込もうとすると
  `policy.py` が明示的に例外を投げます（黙って誤った obs を作らないため）。
  対応するには 直近5フレーム × 35次元 のリングバッファを `policy.py` に実装し、
  並びを `[最古 … 最新]`（最新が末尾）にします
- LSTM の h0/c0 は演奏ごとにゼロ初期化します（1曲=1セッション運用）
- ログの CSV 保存はしていません。必要なら `runner.Runner.samples` を落としてください

## 5. 安全まわり

| 項目 | 実装 |
|---|---|
| 緊急停止 | 画面に常時「ストップ（緊急停止）」。押すと制御ループが抜けて圧力0を2回送信 |
| 異常力 | `force_limit_N=60.0` を超えると自動停止し画面にエラー表示 |
| 終了時 | 正常終了・例外・停止のいずれでも `plant.close()` で圧力0 |
| ブラウザ切断 | 演奏は止めません（画面が落ちてもロボットは曲を弾き切る）。**物理の非常停止は別途会場に用意してください** |
| BPM範囲 | `--bpm-min/--bpm-max`（既定 60〜180）。学習範囲外に出さないよう当日設定を絞ること |

来場者に開放するのは「曲選択」と「テンポ」と「START」だけです。
モデル選択はUIに出していません（既定モデルは `run_oc_demo.sh` の `OC_MODEL`）。

## 6. OC当日の準備

1. `midi/` に流す曲（.mid）を置く。1曲15〜25秒程度が回転が良い
2. `midi/labels.json` に来場者向けの日本語名を書く
   ```json
   {"01_yonuchi": "四分打ち（かんたん）", "03_rock": "ロックビート"}
   ```
3. MIDIのピッチは GM ドラムマップで表示名が付きます（38=スネア, 42=ハイハット…）
4. Jetsonの時刻合わせ（`sudo date -s ...`）を忘れずに
5. 開場前に `--mock` で一度通し、そのあと実機で全曲を1回ずつ通す

## 7. 画面の見方（来場者への説明用）

- **青いバー** = 楽譜。右から流れてきて、白い縦線（いま）に来た瞬間が叩くタイミング
- **青い山** = そのとき出したい力
- **黄色い線** = 力センサが実際に測った力
- **緑/赤の点** = 実際の打点。緑は楽譜から ±30ms 以内、赤は外れ
- **右下の%** = ±30ms 以内に入った打点の割合

「人間のドラマーの許容ズレが±30ms程度」という枕を振ると伝わりやすいです。

## 8. 動作確認済みの範囲（クラウド側・モック）

- 5曲 × BPM 90/120/160 で通し再生、打点検出・±30ms判定が動作
- ブラウザ2枚同時接続でテレメトリが欠落せず、譜面も自動でミラーされる
- 1920×1080 / 1600×900 / 1280×800 でレイアウト崩れなし
- 演奏中のブラウザリロード、演奏終了後の連続スタート、STOP
- `midiparse.py` を mido と突き合わせて一致確認（複数トラック / テンポ変化 /
  format 0 / ランニングステータス / note_on velocity=0 による note_off）

**未確認（Jetson側でお願いします）**: 実シリアル通信、ONNX実モデルでの推論、
Jetson上での描画性能、力センサゼロ点。

## 9. うまくいかないとき

| 症状 | 見るところ |
|---|---|
| 実機で1つも受信しない | `python3 -m oc_demo.check_serial`。baudは本番230400（collect_dataの115200と別） |
| ONNXが読めない / 次元が合わない | `python3 -m oc_demo.check_models`。frame stacking付き(論文E)は未対応 |
| ロボットが動かない（実機のはず） | 右上バッジが「駆動なし検証」になっていないか。`--verify` を外す |
| `No such file or directory: oc_demo/requirements.txt` | 展開先が違う。`/data/jetson_project` 直下に `oc_demo/` があるか確認。なおインストール自体が不要 |
| ブラウザが真っ白 | `curl -s localhost:8080/api/health` が返るか。返るならブラウザのキャッシュ、返らないならサーバのログ |
| 曲が1つも出ない | `midi/` に `.mid` があるか。壊れたファイルは起動ログに `[score] skip ...` が出る |
| 演奏は動くがグラフが止まる | SSE が切れている。ブラウザのDevTools → Network → `events` が `pending` のままか確認。EventSourceは自動再接続する |
| `力センサが xx N に達したため安全停止` | `force_limit_N`（既定60N）を超えた。ゼロ点ズレか実際の異常。§3-3 へ |
| MIDIのテンポが変に読まれる | `python3 -c "from oc_demo.midiparse import parse_midi; print(parse_midi('midi/xxx.mid')[1])"` で確認 |
