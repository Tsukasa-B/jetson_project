"""
patch_playback_timing.py — tools/run_signal_playback.py の送信タイミングを
「前回から20 ms」ではなく「開始から (idx+1)×20 ms」に変える（1回だけ実行）。

背景: 毎回 sleep の行き過ぎ（約0.5 ms）が積み上がり、再生が約3%遅れていた。
      2026/9/29 に判明（ログ長が信号長の 1.028〜1.036 倍）。
"""
import os, sys
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_signal_playback.py")
s = open(P, encoding="utf-8").read()
if "t0_abs" in s:
    print("既に適用済みです。"); sys.exit(0)
a = "            for idx, row in self.cmd_df.iterrows():\n"
b = "                while (time.perf_counter() - loop_start) < CONTROL_DT:\n"
for x in (a, b):
    if s.count(x) != 1:
        sys.exit(f"[中止] 置換対象が見つからない/複数あります:\n{x}")
s = s.replace(a, "            t0_abs = time.perf_counter()\n" + a)
s = s.replace(b, "                while time.perf_counter() < t0_abs + (idx + 1) * CONTROL_DT:\n")
open(P, "w", encoding="utf-8").write(s)
print("適用しました:", os.path.relpath(P))
