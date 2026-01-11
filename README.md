時刻直し
```
sudo date -s "$(wget -qSO- --max-redirect=0 google.com 2>&1 | grep Date: | cut -d' ' -f5-8)Z"
```

USBドライバの認識コマンド
```
# RATOCケーブルを認識させるおまじない
sudo modprobe ftdi_sio
echo 0584 b050 | sudo tee /sys/bus/usb-serial/drivers/ftdi_sio/new_id
```

ros2コンテナ起動プロンプト
```
sudo docker run --runtime nvidia -it --rm --network host \
    --device /dev/ttyUSB0 \
    -v /ssd/jetson_project:/workspace \
    -w /workspace \
    my_ros2_pytorch_container:latest bash
```

collect_real_data.py
目的	コマンド (コピペ用)	収集時間の目安
★AI学習データ収集


(メイン作業)

sudo python3 collect_real_data_v3.py --mode train	
合計 3〜5分


(休憩込みで10分程度)

Sim比較用


(ステップ応答)

sudo python3 collect_real_data_v3.py --mode step	30秒
モデル作成用


(ヒステリシス)

sudo python3 collect_real_data_v3.py --mode hysteresis	
1分


(ゆっくり5往復)

s キー：スタート / 再開 (Start)
動作: ロボットが動き出し、データの記録が始まります。

いつ押す？:

最初の開始時。

休憩（Pause）から復帰するとき。

画面表示: >>> START: TRAIN (Seg X) <<<

p キー：一時停止 / 休憩 (Pause)
動作: ロボットが脱力（圧力0）し、データ記録が止まります。

いつ押す？:

コンプレッサーの圧力が落ちてきた時（重要！）

ロボットの動きが怪しい時。

ちょっと休憩したい時。

裏側の処理: segment_id（データの区切り番号）が +1 されます。これにより、後でデータ解析するときに「ここで一旦切れたんだな」と自動判別できます。

画面表示: ||| PAUSE ||| Seg X Done.

q キー：終了＆保存 (Quit)
動作: プログラムを終了し、CSVファイルを保存します。

いつ押す？:

十分なデータが集まった時。

実験を完全にやめるとき。

画面表示: [SAVE] Saving XXX samples to real_data_mode_xxxx.csv...