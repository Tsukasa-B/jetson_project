"""シリアル受信の健康診断。圧力指令は一切送らない（ロボットは動かない）。

    python3 -m oc_demo.check_serial              # 5秒
    python3 -m oc_demo.check_serial --seconds 20

見るところ:
  - 受信レート … 200Hz 近辺か。低い/欠落があるとログの `time = arange*0.005` 前提が崩れ、
                  ±30ms判定がそのぶん嘘になる
  - force_N     … **無負荷でこれが 0 付近か**。-20 N 付近ならゼロ点異常（既知の未解決事項）
  - 角度        … 静止しているのに動いていないか（ノイズ量の把握）
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import adapter
from .plant import SerialPlant

FIELDS = ["圧力DF", "圧力F", "圧力G", "手首deg", "グリップdeg", "flag", "force_N"]


def main() -> None:
    ap = argparse.ArgumentParser(description="シリアル受信の確認（送信はしません）")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=adapter.SERIAL_BAUD)
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    print(f"[check_serial] {args.port} @ {args.baud} を {args.seconds:.0f} 秒読みます")
    print("[check_serial] 圧力指令は送りません。ロボットは動きません。")
    print("[check_serial] ★ スティックが打面に触れていない（無負荷）状態にしてください\n")

    plant = SerialPlant(args.port, args.baud)
    plant.open()
    try:
        time.sleep(0.3)  # 受信スレッドの立ち上がり
        plant.clear_buffer_for_sync()

        rows = []
        t0 = time.time()
        last_stamp = None
        while time.time() - t0 < args.seconds:
            fr = plant.read()
            if fr.stamp and fr.stamp != last_stamp:
                rows.append([*fr.meas_p, fr.wrist_deg, fr.grip_deg, fr.flag, fr.force_N])
                last_stamp = fr.stamp
            time.sleep(0.001)

        elapsed = time.time() - t0
        stats = plant.stats
    finally:
        plant.close()

    recv = stats.get("recv", 0)
    bad = stats.get("bad", 0)
    rate = recv / elapsed if elapsed else 0.0

    print("=" * 62)
    print(f"受信パケット : {recv} 個 / {elapsed:.2f} 秒 = {rate:.1f} Hz  （期待 200 Hz）")
    print(f"不正パケット : {bad}")
    if recv == 0:
        print("\n!! 1つも受信できていません。")
        print("   - MicroLabBox 側の Simulink は動いていますか")
        print("   - baud は 230400 ですか（collect_data は 115200 なので別物です）")
        print("   - ホスト側で modprobe ftdi_sio と new_id を実行しましたか")
        return
    if rate < 180:
        print(f"\n!! 受信レートが低いです（{rate:.0f} Hz）。取りこぼしがあると")
        print("   ログ時刻の 200Hz 前提が崩れ、±30ms判定の信頼性が落ちます。")

    a = np.array(rows, dtype=float)
    print("-" * 62)
    print(f"{'項目':<12}{'中央値':>10}{'最小':>10}{'最大':>10}{'標準偏差':>10}")
    for i, name in enumerate(FIELDS):
        col = a[:, i]
        print(f"{name:<12}{np.median(col):>10.3f}{col.min():>10.3f}"
              f"{col.max():>10.3f}{col.std():>10.3f}")
    print("=" * 62)

    fz = float(np.median(a[:, 6]))
    if abs(fz) > 2.0:
        print(f"\n!! 無負荷のはずの force_N が {fz:.2f} N です（正常時は -0.3〜0 程度）。")
        print("   このままでも oc_demo は開始時に自動でゼロ点を引きますが、")
        print("   ズレの原因（配線 / アンプ / Simulink側スケール）を先に潰してください。")
        print("   ±30ms判定はこの値に直結します。")
    else:
        print(f"\nOK: 力センサのゼロ点は {fz:.3f} N。妥当な範囲です。")


if __name__ == "__main__":
    main()
