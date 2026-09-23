"""
generate_volume_coupling_signals.py

目的
----
「指令圧が一定でも、関節が動けばPAMの体積が変わり、実測圧が動くのか」
を判定するための加振信号を生成する。治具は一切使わない。ソフトのみ。

なぜ既存の test_signals では駄目か
----------------------------------
  exp1_static_hysteresis : 構造は正しい（F固定・DF正弦）が 0.05 Hz。
      同定済みの一次遅れ τ≈88 ms → コーナー周波数 1/(2πτ) ≈ 1.8 Hz。
      1.8 Hz より下では圧力源が体積変化を追従して打ち消すので、
      体積結合があっても実測圧には出てこない。
  exp3_frequency_sweep   : DF と F を逆位相で振るため「保持側」が存在せず、
      指令由来の圧力変化と体積由来の圧力変化を分離できない。

したがって「片側を一定に保ったまま、1.8 Hz より上で関節を振る」信号が要る。

安全制約（PLAフレーム保護）
--------------------------
  * 関節を機械的に固定しない（治具は使用禁止）。
  * 拮抗側は常に 0.10 MPa 以上（エンドストップへの突き当たりを防ぐ）。
  * 保持圧は 0.35 MPa 以下。実機ログ105runで min(cmd_DF,cmd_F) の p90 が
    0.41 MPa なので、この範囲は既に日常的に経験している荷重の内側。
  * 1バースト 6 s 以下・休止 10 s。実機ログでの
    「両筋 0.30 MPa 以上が連続した最長時間」は 2.59 s なので、
    持続共収縮はそこから大きく離さない。
  * 指令圧の上限 0.45 MPa（通常運転の 0.60 MPa より低い）。
  * 指令の変化率は最大でも 2π·f·A = 2π·8·0.15 ≈ 7.5 MPa/s。
    通常運転のバンバン指令は 30 MPa/s なので 1/4 以下。

出力
----
  tools/test_signals/exc_volume_coupling.csv
  列は run_signal_playback.py が読む形式:
      time, cmd_pressure_DF, cmd_pressure_F, cmd_pressure_G

実行
----
  python tools/generate_volume_coupling_signals.py
  python tools/run_signal_playback.py exc_volume_coupling
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DT = 0.02                 # 50 Hz（run_signal_playback.py の CONTROL_DT と一致させる）

# ---- 安全定数（ここを緩めないこと）----------------------------------------
P_ABS_MAX = 0.45          # MPa 指令の絶対上限
P_HOLD_MAX = 0.35         # MPa 保持側の上限
P_ANTAG_MIN = 0.10        # MPa 拮抗側の下限
BURST_MAX_S = 6.0         # s  1バーストの最大長
REST_S = 10.0             # s  バースト間の休止
RAMP_S = 0.5              # s  正弦振幅の立上り／立下り
SETTLE_S = 1.0            # s  DC成立を待つ時間
P_GRIP = 0.30             # MPa スティック保持（一定）

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_signals")


def _taper(n: int, n_ramp: int) -> np.ndarray:
    """レイズドコサインの窓。バースト端での段差をなくす。"""
    w = np.ones(n)
    n_ramp = min(n_ramp, n // 2)
    if n_ramp <= 0:
        return w
    r = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_ramp) / n_ramp))
    w[:n_ramp] = r
    w[-n_ramp:] = r[::-1]
    return w


class Sequence:
    def __init__(self):
        self.df: list[float] = []
        self.f: list[float] = []
        self.g: list[float] = []
        self.tag: list[str] = []

    def _push(self, df, f, g, tag):
        self.df.extend(np.atleast_1d(df).tolist())
        self.f.extend(np.atleast_1d(f).tolist())
        self.g.extend(np.atleast_1d(g).tolist())
        self.tag.extend([tag] * len(np.atleast_1d(df)))

    def rest(self, sec: float = REST_S):
        n = int(round(sec / DT))
        z = np.full(n, P_ANTAG_MIN)          # 完全排気はせず最低圧を残す
        self._push(z, z, np.full(n, P_GRIP), "rest")

    def burst(self, *, exc: str, hold: str, p_center: float, amp: float,
              freq: float, dur: float, p_hold: float, tag: str):
        """exc 側を正弦加振、hold 側を一定に保つ。"""
        assert exc in ("DF", "F", "G") and hold in ("DF", "F", "G") and exc != hold
        assert dur <= BURST_MAX_S, f"burst {dur}s > {BURST_MAX_S}s"
        assert p_hold <= P_HOLD_MAX, f"hold {p_hold} > {P_HOLD_MAX}"
        assert p_center + amp <= P_ABS_MAX + 1e-9
        assert p_center - amp >= P_ANTAG_MIN - 1e-9
        assert p_hold >= P_ANTAG_MIN

        # 1) DC を確立（ここで関節は静的平衡へ）
        n_s = int(round(SETTLE_S / DT))
        ramp = np.linspace(P_ANTAG_MIN, 1.0, n_s)
        dc_exc = P_ANTAG_MIN + (p_center - P_ANTAG_MIN) * np.linspace(0, 1, n_s)
        dc_hold = P_ANTAG_MIN + (p_hold - P_ANTAG_MIN) * np.linspace(0, 1, n_s)
        self._assign(exc, hold, dc_exc, dc_hold, f"{tag}:settle")
        del ramp

        # 2) 加振
        n = int(round(dur / DT))
        t = np.arange(n) * DT
        w = _taper(n, int(round(RAMP_S / DT)))
        sig = p_center + amp * w * np.sin(2.0 * np.pi * freq * t)
        self._assign(exc, hold, sig, np.full(n, p_hold), tag)

        # 3) DC のまま少し保持（残響を見る）
        n_e = int(round(0.5 / DT))
        self._assign(exc, hold, np.full(n_e, p_center), np.full(n_e, p_hold),
                     f"{tag}:tail")

    def _assign(self, exc, hold, v_exc, v_hold, tag):
        n = len(v_exc)
        vals = {"DF": np.full(n, P_ANTAG_MIN),
                "F": np.full(n, P_ANTAG_MIN),
                "G": np.full(n, P_GRIP)}
        vals[exc] = np.asarray(v_exc, float)
        vals[hold] = np.asarray(v_hold, float)
        self._push(vals["DF"], vals["F"], vals["G"], tag)

    def to_frame(self) -> pd.DataFrame:
        df = np.clip(np.array(self.df), 0.0, P_ABS_MAX)
        f = np.clip(np.array(self.f), 0.0, P_ABS_MAX)
        g = np.clip(np.array(self.g), 0.0, P_ABS_MAX)
        t = np.arange(len(df)) * DT
        return pd.DataFrame({"time": t,
                             "cmd_pressure_DF": df,
                             "cmd_pressure_F": f,
                             "cmd_pressure_G": g,
                             "segment": self.tag})


def build() -> pd.DataFrame:
    s = Sequence()
    s.rest(3.0)

    # --- Part A: 周波数依存性 -------------------------------------------
    # 体積結合が本物なら、圧力源が追従できなくなる 1.8 Hz 以上で
    # リプル振幅が周波数とともに増え、やがて飽和する。
    # 圧力経路の同定誤差なら、周波数依存はこの形にならない。
    for fr in (1.0, 2.0, 3.0, 5.0, 8.0):
        s.burst(exc="DF", hold="F", p_center=0.30, amp=0.15, freq=fr,
                dur=BURST_MAX_S, p_hold=0.25, tag=f"A_f{fr:g}")
        s.rest()

    # --- Part B: 保持圧依存性 -------------------------------------------
    # 理想気体なら dP/dθ ∝ P。保持圧を変えて比例するかを見る。
    # （0.25 MPa / 3 Hz は Part A と共通なので再取得しない）
    for ph in (0.15, 0.35):
        s.burst(exc="DF", hold="F", p_center=0.30, amp=0.15, freq=3.0,
                dur=BURST_MAX_S, p_hold=ph, tag=f"B_hold{ph:g}")
        s.rest()

    # --- Part C: 振幅依存性（たるみの判別）------------------------------
    # 体積結合なら リプル ∝ 関節角振幅（原点を通る直線）。
    # ワイヤのたるみなら ある角度までリプルが出ず、しきい値を境に立ち上がる。
    for am in (0.05, 0.10):
        s.burst(exc="DF", hold="F", p_center=0.30, amp=am, freq=3.0,
                dur=BURST_MAX_S, p_hold=0.25, tag=f"C_amp{am:g}")
        s.rest()

    # --- Part D: 役割を入れ替えた対照 -----------------------------------
    # DF を保持し F を加振。Aと符号が反転するはず（拮抗対なので）。
    # 反転しなければ、見ているのは体積結合ではなく計測系のクロストーク。
    s.burst(exc="F", hold="DF", p_center=0.30, amp=0.15, freq=3.0,
            dur=BURST_MAX_S, p_hold=0.25, tag="D_swap_f3")
    s.rest(5.0)
    return s.to_frame()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build()

    # --- 安全チェック（書き出し前に必ず通す）---------------------------
    for c in ("cmd_pressure_DF", "cmd_pressure_F", "cmd_pressure_G"):
        assert df[c].max() <= P_ABS_MAX + 1e-9, f"{c} exceeds {P_ABS_MAX}"
        assert df[c].min() >= -1e-9
    wrist_pair_min = np.minimum(df["cmd_pressure_DF"], df["cmd_pressure_F"])
    assert wrist_pair_min.max() <= P_HOLD_MAX + 1e-9
    rate = np.abs(np.diff(df["cmd_pressure_DF"])) / DT
    assert rate.max() < 10.0, f"dP/dt {rate.max():.1f} MPa/s too high"

    path = os.path.join(OUT_DIR, "exc_volume_coupling.csv")
    # segment 列は実行側が読まないので落とす（参照用に別ファイルへ）
    df[["time", "cmd_pressure_DF", "cmd_pressure_F",
        "cmd_pressure_G"]].to_csv(path, index=False)
    df.to_csv(path.replace(".csv", "_annotated.csv"), index=False)

    print(f"Generated: {path}")
    print(f"  duration      : {df['time'].iloc[-1] + DT:.1f} s "
          f"({len(df)} steps @ {1 / DT:.0f} Hz)")
    print(f"  bursts        : {df['segment'].str.match(r'^[A-D]_').sum() * DT:.1f} s of excitation")
    print(f"  max cmd       : DF {df['cmd_pressure_DF'].max():.2f} / "
          f"F {df['cmd_pressure_F'].max():.2f} / G {df['cmd_pressure_G'].max():.2f} MPa")
    print(f"  max co-contr  : {wrist_pair_min.max():.2f} MPa "
          f"(実機ログ105runのp90 = 0.41 MPa)")
    print(f"  max |dP/dt|   : {rate.max():.2f} MPa/s (通常運転 30 MPa/s)")


if __name__ == "__main__":
    main()
