"""
patch_time_gen.py — ログのセンサ時刻を「実際の発生時刻」にする（1回だけ実行）

背景（2026/9/30 判明）:
  旧ログの time 列は「開始後に受信した順番 × 5 ms」。開始の瞬間に USB 変換チップ内に
  溜まっていた開始前のパケットも t=0 から数えてしまうため、latency_timer=16 では
  センサ側の時刻全体が約 40 ms 後ろにずれていた（エコーと受信時刻の2通りで確認）。
  このずれで、打撃が実際より約 30 ms 遅れて見え、成功率も低く出ていた。

変更（src/deploy_policy.py）:
  - 受信時刻 t_recv の下側包絡（最も遅延が小さいパケット）に直線を当て、
    シリアル転送時間 2.5 ms を引いたものを各パケットの発生時刻とする
  - time = 発生時刻 − 制御ループ開始時刻（旧定義は time_recon 列に残す）
  - 指令列（目標力を含む）は、この time に対して貼り付ける
  - JSON に time_basis="gen_envelope" と t_start を記録
変更（analysis/strike_metrics.py）:
  - time_recon 列があるログ（＝このパッチ以降）は time をそのまま使う
    （rate_corrected などの補正はかけない。旧ログの扱いは変えない）
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def patch(path, edits, marker):
    s = open(path, encoding="utf-8").read()
    if marker in s:
        print("既に適用済み:", os.path.relpath(path)); return
    for old, new in edits:
        n = s.count(old)
        if n != 1:
            sys.exit(f"[中止] {os.path.relpath(path)}: 置換対象が {n} 箇所（1箇所のはず）:\n{old}")
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)
    print("適用しました:", os.path.relpath(path))


patch(os.path.join(ROOT, "src", "deploy_policy.py"), [
    ("        t_start = time.perf_counter()\n",
     "        t_start = time.perf_counter()\n        self.t_start = t_start\n"),
    ('        df["time"] = np.arange(len(df)) / SENSOR_RATE_HZ   # 再構成時刻（旧来と同じ定義）\n',
     '        df["time_recon"] = np.arange(len(df)) / SENSOR_RATE_HZ   # 旧定義（受信順×5ms）\n'
     '        # 実際の発生時刻: 受信時刻の下側包絡に直線を当て、シリアル転送時間を引く\n'
     '        k = np.arange(len(df))\n'
     '        tr = df["t_recv"].to_numpy(float)\n'
     '        coef = np.polyfit(k, tr, 1)\n'
     '        env = np.percentile(tr - np.polyval(coef, k), 2)\n'
     '        xfer = 58 * 10 / float(a.baud or 230400)\n'
     '        df["time"] = np.polyval(coef, k) + env - xfer - getattr(self, "t_start", tr[0])\n'),
    ('            "control_dt": self.dt, "p_max": self.spec.p_max,\n',
     '            "control_dt": self.dt, "p_max": self.spec.p_max,\n'
     '            "time_basis": "gen_envelope", "t_start": getattr(self, "t_start", None),\n'),
], marker="time_recon")

patch(os.path.join(ROOT, "analysis", "strike_metrics.py"), [
    ('    df = df.copy()\n    if mode == "recon" or "t_recv_rel" not in df.columns:\n',
     '    df = df.copy()\n'
     '    if "time_recon" in df.columns:   # 9/30以降のログ: time は発生時刻（補正済み）\n'
     '        return df\n'
     '    if mode == "recon" or "t_recv_rel" not in df.columns:\n'),
], marker="time_recon")
