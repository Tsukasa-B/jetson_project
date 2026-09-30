"""
patch_deploy_latency.py — src/deploy_policy.py に遅れ計測用の変更を入れる（1回だけ実行）

入れる変更:
  1) --usb_latency {1,16} 引数
       USB変換（FTDI）の溜め込み時間を実行ごとに明示的に設定する。
       pyserial の set_low_latency_mode(True/False) は ftdi_sio では
       latency_timer を 1 ms / 16 ms に切り替える。再起動で戻る問題を避け、
       条件をログのJSONに必ず残すため、sysfs 手書きではなくここで設定する。
  2) 制御ループの記録に 2 列を追加
       obs_t_recv : 方策が使ったセンサパケットの受信時刻（perf_counter）
       t_send     : 圧力指令を送った直後の時刻（perf_counter）
     → t_send - obs_t_recv が「観測の古さ＋方策の計算時間」
  3) JSON に usb_latency と実際の latency_timer 値を記録

使い方:
  python tools/patch_deploy_latency.py        # 変更を入れる
  python src/deploy_policy.py --model RAL/E_seed2 --midi songs/test_double_bpm160.mid \
      --usb_latency 16 --no-input
"""
import os
import re
import sys

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "deploy_policy.py")
src = open(PATH, encoding="utf-8").read()

if "--usb_latency" in src:
    print("既に適用済みです。何もしません。")
    sys.exit(0)

edits = [
    # 1) 引数
    ('    p.add_argument("--list", action="store_true",',
     '    p.add_argument("--usb_latency", type=int, choices=[1, 16], default=None,\n'
     '                   help="USB変換の溜め込み時間[ms]。1 または 16。省略時は変更しない")\n'
     '    p.add_argument("--list", action="store_true",'),
    # 1') シリアルを開いた直後に設定
    ('            self.ser = open_serial(port, baud)\n',
     '            self.ser = open_serial(port, baud)\n'
     '            if args.usb_latency is not None:\n'
     '                self.ser.set_low_latency_mode(args.usb_latency == 1)\n'
     '            print(f"[Link] latency_timer = {read_latency_timer(port)}")\n'),
    # 2) 送信直後の時刻
    ('                    send_pressure(self.ser, *cmd)\n',
     '                    send_pressure(self.ser, *cmd)\n'
     '                t_send = time.perf_counter()\n'),
    # 2') ログに2列追加
    ('                    "cmd_DF": float(cmd[0]), "cmd_F": float(cmd[1]), "cmd_G": float(cmd[2]),\n',
     '                    "cmd_DF": float(cmd[0]), "cmd_F": float(cmd[1]), "cmd_G": float(cmd[2]),\n'
     '                    "obs_t_recv": float(sensor["t_recv"]), "t_send": t_send,\n'),
    # 3) JSON
    ('            "control_dt": self.dt, "p_max": self.spec.p_max,\n',
     '            "control_dt": self.dt, "p_max": self.spec.p_max,\n'
     '            "usb_latency_arg": a.usb_latency,\n'
     '            "latency_timer": read_latency_timer(a.port or "/dev/ttyUSB0"),\n'),
    # 補助関数（git_rev の直前に置く）
    ('def git_rev() -> str:\n',
     'def read_latency_timer(port: str):\n'
     '    """sysfs の latency_timer を読む。読めなければ None。"""\n'
     '    name = os.path.basename(os.path.realpath(port))\n'
     '    try:\n'
     '        with open(f"/sys/bus/usb-serial/devices/{name}/latency_timer") as f:\n'
     '            return int(f.read().strip())\n'
     '    except Exception:  # noqa: BLE001\n'
     '        return None\n'
     '\n\n'
     'def git_rev() -> str:\n'),
]

for old, new in edits:
    n = src.count(old)
    if n != 1:
        sys.exit(f"[中止] 置換対象が {n} 箇所見つかりました（1箇所のはず）:\n{old}")
    src = src.replace(old, new)

open(PATH, "w", encoding="utf-8").write(src)
print("適用しました:", os.path.relpath(PATH))
