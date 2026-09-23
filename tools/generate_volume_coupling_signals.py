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
  <run_signal_playback.py が探す test_signals/>/exc_volume_coupling.csv
  列は run_signal_playback.py が読む形式:
      time, cmd_pressure_DF, cmd_pressure_F, cmd_pressure_G

  出力先はリポジトリの構成に合わせて自動で決める。v3再編前は
  tools/test_signals/、再編後は <repo root>/test_signals/ が探索先なので、
  存在する方に書く（どちらも無ければ repo root 側を作る）。
  --out で明示指定も可。

実行
----
  python tools/generate_volume_coupling_signals.py
  python tools/run_signal_playback.py exc_volume_coupling
"""

from __future__ import annotations

import argparse
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
DOWN_S = 0.4              # s  バースト終端から休止圧へ戻すランプ
RATE_LIMIT = 9.0          # MPa/s 指令変化率の上限（アサート用）
P_GRIP = 0.30             # MPa スティック保持（一定）

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)


def resolve_out_dir() -> str:
    """run_signal_playback.py が探す test_signals/ を見つける。

    v3再編で _resolve_path() の基準が script_dir から REPO_ROOT に変わったため、
    どちらの構成でも正しい場所に書けるように存在確認で決める。
    """
    for cand in (os.path.join(_REPO_ROOT, "test_signals"),
                 os.path.join(_HERE, "test_signals")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(_REPO_ROOT, "test_signals")


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
        """休止。直前の値から休止圧まで DOWN_S 秒かけて降ろしてから保持する。

        ここを段差にすると 0.30 -> 0.10 MPa が 1ステップ(20ms)で落ちて
        15 MPa/s になり、指令変化率の上限に貼り付く。ランプで逃がす。
        """
        n_d = int(round(DOWN_S / DT))
        if self.df and n_d > 0:
            for arr, tgt in ((self.df, P_ANTAG_MIN), (self.f, P_ANTAG_MIN),
                             (self.g, P_GRIP)):
                start = arr[-1]
                arr.extend(np.linspace(start, tgt, n_d + 1)[1:].tolist())
            self.tag.extend(["rest:down"] * n_d)
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


def build(repeats: int = 1) -> pd.DataFrame:
    s = Sequence()
    s.rest(3.0)
    for _ in range(max(1, repeats)):
        _one_pass(s)
    return s.to_frame()


def _one_pass(s: "Sequence") -> None:

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

    # --- Part C: 動作点依存性（たるみの判別）----------------------------
    # たるみは「関節角がある値より小さいとワイヤが張らない」という現象なので、
    # 振幅ではなく動作点（平均関節角）に依存する。
    # 振幅を固定したまま中心圧だけ動かせば、SNR を一定に保ったまま
    # 平均関節角だけを変えられる。
    #   dP/dθ が動作点によらず一定      -> たるみは無い（体積結合で説明できる）
    #   ある動作点から下で dP/dθ が落ちる -> そこがたるみのしきい角
    # （振幅を小さくする設計では、たるみが有る場合ほど信号が消えて
    #   ノイズ床と区別できなくなる。判定したい当の条件で測れないので使わない。）
    for pc in (0.20, 0.28, 0.35):
        s.burst(exc="DF", hold="F", p_center=pc, amp=0.10, freq=3.0,
                dur=BURST_MAX_S, p_hold=0.25, tag=f"C_center{pc:g}")
        s.rest()

    # 線形性の確認（体積結合なら リプル ∝ 関節角振幅）。
    # Part A の 3 Hz（振幅0.15）と C_center0.28（振幅0.10）の2点で足りる。

    # --- Part D: 役割を入れ替えた対照 -----------------------------------
    # DF を保持し F を加振。Aと符号が反転するはず（拮抗対なので）。
    # 反転しなければ、見ているのは体積結合ではなく計測系のクロストーク。
    s.burst(exc="F", hold="DF", p_center=0.30, amp=0.15, freq=3.0,
            dur=BURST_MAX_S, p_hold=0.25, tag="D_swap_f3")
    s.rest(5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="出力ディレクトリ（既定は自動判定）")
    ap.add_argument("--repeats", type=int, default=2,
                    help="全バーストを何回繰り返すか。既定2。合成ログでの"
                         "試算では1回だと高周波側が SNR<3 に落ちる。"
                         "2回で誤差が 1/sqrt(2) になる（所要 約6.5分）")
    args = ap.parse_args()

    out_dir = args.out or resolve_out_dir()
    os.makedirs(out_dir, exist_ok=True)
    df = build(args.repeats)

    # --- 安全チェック（書き出し前に必ず通す）---------------------------
    for c in ("cmd_pressure_DF", "cmd_pressure_F", "cmd_pressure_G"):
        assert df[c].max() <= P_ABS_MAX + 1e-9, f"{c} exceeds {P_ABS_MAX}"
        assert df[c].min() >= -1e-9
    wrist_pair_min = np.minimum(df["cmd_pressure_DF"], df["cmd_pressure_F"])
    assert wrist_pair_min.max() <= P_HOLD_MAX + 1e-9
    rate = max(np.abs(np.diff(df[c])).max() / DT
               for c in ("cmd_pressure_DF", "cmd_pressure_F"))
    assert rate < RATE_LIMIT, f"dP/dt {rate:.2f} MPa/s >= {RATE_LIMIT}"

    path = os.path.join(out_dir, "exc_volume_coupling.csv")
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
    print(f"  max |dP/dt|   : {rate:.2f} MPa/s "
          f"(上限 {RATE_LIMIT:.1f} / 通常運転 30 MPa/s)")


if __name__ == "__main__":
    main()
