"""
analyze_volume_coupling.py

run_signal_playback.py が exc_volume_coupling.csv で取ったログを読み、
各バーストについて「関節角1度あたり何kPa 実測圧が動いたか」(dP/dθ) を
ロックイン検波で推定する。

ロックイン検波を使う理由
------------------------
加振周波数 f は既知なので、その周波数成分だけを取り出せばよい。
N サンプル平均すると振幅推定の誤差は σ·sqrt(2/N) まで落ちる。
実機ログの静定区間から見積もった圧力ノイズは σ ≈ 3 kPa (DF) / 12 kPa (F)。
1バースト 5 s × 200 Hz = 1000 サンプルなら誤差 0.1〜0.5 kPa。
期待される信号は数 kPa〜20 kPa なので、SNR は十分。

判定
----
  dP/dθ が 1.8 Hz 以下でほぼ 0、それ以上で立ち上がって頭打ち
      → 体積結合が実在する。前向きモデルに体積項を入れる根拠になる。
  dP/dθ が周波数に依らずほぼ 0（< 0.2 kPa/deg）
      → 体積結合ではない。残差の原因は圧力経路（時定数・むだ時間・ヒステリシス）側。
  Part C の振幅掃引で 原点を通らない（小振幅で 0、しきい値から立ち上がる）
      → ワイヤのたるみ。Heaviside 項のしきい値を実測で決められる。
  Part D で符号が反転しない
      → 見ているのは体積結合ではなく計測系のクロストーク。配線を疑う。

使い方
------
  python tools/analyze_volume_coupling.py <log.csv> \
      --annot tools/test_signals/exc_volume_coupling_annotated.csv
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

SENSOR_RATE_HZ = 200.0
EDGE_SKIP_S = 0.6          # テーパ区間を捨てる


def lockin(x: np.ndarray, t: np.ndarray, f: float) -> tuple[float, float]:
    """x の周波数 f 成分の振幅と位相[deg]。"""
    c = np.cos(2 * np.pi * f * t)
    s = np.sin(2 * np.pi * f * t)
    x = x - x.mean()
    a = 2.0 * np.mean(x * c)
    b = 2.0 * np.mean(x * s)
    return float(np.hypot(a, b)), float(np.degrees(np.arctan2(a, b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--annot", required=True)
    args = ap.parse_args()

    log = pd.read_csv(args.log)
    ann = pd.read_csv(args.annot)

    # ログ側の時間軸。run_signal_playback.py は 200 Hz で再構成している。
    if "time" not in log.columns:
        log["time"] = np.arange(len(log)) / SENSOR_RATE_HZ

    rows = []
    for tag, g in ann.groupby("segment", sort=False):
        m = re.match(r"^([A-D])_", str(tag))
        if not m:
            continue
        t0, t1 = g["time"].iloc[0], g["time"].iloc[-1]
        sel = log[(log["time"] >= t0 + EDGE_SKIP_S) & (log["time"] <= t1 - EDGE_SKIP_S)]
        if len(sel) < 100:
            continue

        part = m.group(1)
        exc, hold = ("F", "DF") if part == "D" else ("DF", "F")
        # 加振周波数は指令から直接推定（tag に頼らない）
        cmd = g[f"cmd_pressure_{exc}"].values
        cmd = cmd - cmd.mean()
        n = len(cmd)
        sp = np.fft.rfft(cmd * np.hanning(n))
        freq = np.fft.rfftfreq(n, 1.0 / 50.0)[np.argmax(np.abs(sp[1:])) + 1]

        t = sel["time"].values
        ang = sel["wrist_angle_deg"].values
        p_hold = sel[f"meas_pres_{hold}"].values * 1000.0     # kPa
        p_exc = sel[f"meas_pres_{exc}"].values * 1000.0

        a_ang, ph_ang = lockin(ang, t, freq)
        a_hold, ph_hold = lockin(p_hold, t, freq)
        a_exc, _ = lockin(p_exc, t, freq)

        # ノイズ床: 加振周波数の 1.37 倍（非調和）での応答を代用
        nz, _ = lockin(p_hold, t, freq * 1.37)

        rows.append(dict(
            segment=tag, part=part, freq_Hz=round(float(freq), 2),
            hold_ch=hold,
            ang_amp_deg=round(a_ang, 2),
            hold_ripple_kPa=round(a_hold, 2),
            noise_kPa=round(nz, 2),
            dPdtheta_kPa_per_deg=round(a_hold / a_ang, 3) if a_ang > 0.5 else np.nan,
            phase_lag_deg=round(((ph_hold - ph_ang + 180) % 360) - 180, 1),
            exc_ripple_kPa=round(a_exc, 1),
            snr=round(a_hold / nz, 1) if nz > 0 else np.nan,
        ))

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(out.to_string(index=False))

    a = out[out["part"] == "A"].sort_values("freq_Hz")
    if len(a) >= 3:
        lo = a[a["freq_Hz"] <= 1.8]["dPdtheta_kPa_per_deg"].mean()
        hi = a[a["freq_Hz"] >= 3.0]["dPdtheta_kPa_per_deg"].mean()
        print(f"\n[Part A] dP/dtheta  <=1.8Hz: {lo:.3f}   >=3Hz: {hi:.3f} kPa/deg")
        if np.isfinite(hi) and hi > 0.5 and hi > 2 * max(lo, 1e-6):
            print("  -> 高周波で立ち上がっている。体積結合は実在する。")
        elif np.isfinite(hi) and hi < 0.2:
            print("  -> 周波数によらず小さい。体積結合では残差を説明できない。")
        else:
            print("  -> 判定保留。SNR とノイズ床を確認すること。")

    c = out[out["part"] == "C"]
    if len(c) >= 2:
        print("\n[Part C] 振幅依存性（原点を通るか＝たるみの有無）")
        print(c[["segment", "ang_amp_deg", "hold_ripple_kPa"]].to_string(index=False))

    d = out[out["part"] == "D"]
    if len(d) and len(a):
        a3 = a[np.isclose(a["freq_Hz"], 3.0, atol=0.3)]
        if len(a3):
            print(f"\n[Part D] 位相 A(3Hz) {a3['phase_lag_deg'].iloc[0]:+.0f} deg "
                  f"vs D {d['phase_lag_deg'].iloc[0]:+.0f} deg "
                  "（拮抗対なら約180度ずれるはず）")

    out.to_csv("volume_coupling_result.csv", index=False)
    print("\nwrote volume_coupling_result.csv")


if __name__ == "__main__":
    main()
