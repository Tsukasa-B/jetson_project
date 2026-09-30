"""
patch_rx_race.py — src/microlabbox.py の受信スレッドの競合を直す（1回だけ実行）

症状（2026/9/30, latency_timer=1 のとき）:
  [Rx Error] device reports readiness to read but returned no data
  受信 400 / 期待 5976
原因:
  開始の瞬間に、メインスレッドの clear_for_sync() が reset_input_buffer() を呼ぶ。
  受信スレッドが「データあり」と判定してから read するまでの間にバッファが消されると、
  pyserial は read 0 バイトを例外にし、受信スレッドはそこで止まる。
  以後、制御ループは開始前の古いセンサ値を使い続ける（ログには開始前2秒分=約400個だけ残る）。
  latency_timer=1 だとデータが細かく頻繁に届くので、この競合がほぼ毎回起きる。
修正:
  1) バッファの消去は受信スレッド自身が行う（メインスレッドは依頼フラグを立てるだけ）
  2) 読み取り0バイトの例外が出ても受信を止めずに続ける（回数は数えて残す）
"""
import os
import sys

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "microlabbox.py")
s = open(P, encoding="utf-8").read()
if "_flush_req" in s:
    print("既に適用済みです。")
    sys.exit(0)

edits = [
    ("        self._clear_flag = False\n        self._lock = threading.Lock()\n",
     "        self._clear_flag = False\n        self._flush_req = False\n        self.n_rx_retry = 0\n"
     "        self._lock = threading.Lock()\n"),
    ("        while self.running:\n            try:\n                n = self.ser.in_waiting\n",
     "        while self.running:\n            try:\n"
     "                if self._flush_req:\n"
     "                    self.ser.reset_input_buffer()\n"
     "                    buf = b\"\"\n"
     "                    self._flush_req = False\n"
     "                n = self.ser.in_waiting\n"),
    ("            except Exception as e:  # noqa: BLE001\n                print(f\"[Rx Error] {e}\")\n"
     "                self.running = False\n",
     "            except Exception as e:  # noqa: BLE001\n"
     "                if \"returned no data\" in str(e):\n"
     "                    self.n_rx_retry += 1     # 競合による空読み。止めずに続ける\n"
     "                    continue\n"
     "                print(f\"[Rx Error] {e}\")\n"
     "                self.running = False\n"),
    ("        self.ser.reset_input_buffer()\n        with self._lock:\n            self._clear_flag = True\n",
     "        with self._lock:\n            self._flush_req = True\n            self._clear_flag = True\n"),
]
for old, new in edits:
    if s.count(old) != 1:
        sys.exit(f"[中止] 置換対象が {s.count(old)} 箇所（1箇所のはず）:\n{old}")
    s = s.replace(old, new)
open(P, "w", encoding="utf-8").write(s)
print("適用しました:", os.path.relpath(P))
