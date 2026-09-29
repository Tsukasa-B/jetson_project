"""
check_forward_model_swap.py

実機ログの圧力指令 cmd_* から、シミュレータと同じ前向き圧力モデルで実測圧を予測し、
実測 meas_pres_* との一致度 (r, NRMSE) を
  (1) ログのまま
  (2) meas_pres_DF と meas_pres_F を入れ替えて
の2通りで出す。

背景: 2026/9/29 の静的ステップ試験で、RA-L期以降の生ログは
meas_pres_DF と meas_pres_F が入れ替わっていることが分かった
（cmd_DF を上げると meas_pres_F が上がる）。
「前向きモデルが手首の拮抗対で外れる」という結果が、この入れ替わりの
見かけでないかを確認する。

前向きモデルは porcaro_2026 の
  common/actions/pneumatic.py（TAU_TABLE_2D_DATA, DEAD_TABLE_2D_DATA）
  common/actions/pam.py（PneumaticModel.step: 方向反転時の始点圧ラッチ,
                         2D表の双線形補間, 分数遅延, 一次遅れ）
を numpy で写したもの。tau_scale = 1（DRなし）。
ログの 200 Hz（dt = 5 ms）でそのまま回す。

使い方:
  python analysis/check_forward_model_swap.py data/ral_20260803
  python analysis/check_forward_model_swap.py data/ral_20260803 --out out/jfps/fwd_swap.csv
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

# ---- porcaro_2026 common/actions/pneumatic.py と同じ値 ----------------------
TAU_2D = np.array([
    [0.0010, 0.0950, 0.0850, 0.0800, 0.0700, 0.0950, 0.1000],
    [0.0850, 0.0010, 0.0750, 0.0950, 0.0900, 0.0800, 0.1050],
    [0.0800, 0.0550, 0.0010, 0.0500, 0.1050, 0.0800, 0.1000],
    [0.0550, 0.0950, 0.0550, 0.0010, 0.0350, 0.0750, 0.0700],
    [0.0600, 0.0850, 0.1150, 0.0950, 0.0010, 0.1550, 0.1400],
    [0.0600, 0.0950, 0.0550, 0.1000, 0.1150, 0.0010, 0.1750],
    [0.0600, 0.0850, 0.0700, 0.0850, 0.0900, 0.1600, 0.0010],
])
DEAD_2D = np.array([
    [0.0010, 0.0050, 0.0100, 0.0150, 0.0600, 0.0550, 0.0550],
    [0.0100, 0.0010, 0.0450, 0.0100, 0.0300, 0.0650, 0.0500],
    [0.0150, 0.0450, 0.0010, 0.0750, 0.0100, 0.0750, 0.0600],
    [0.0550, 0.0800, 0.0850, 0.0010, 0.1050, 0.0850, 0.0650],
    [0.0500, 0.0650, 0.0200, 0.0400, 0.0010, 0.0050, 0.0250],
    [0.0450, 0.0700, 0.0700, 0.0300, 0.0300, 0.0010, 0.0350],
    [0.0700, 0.0750, 0.0750, 0.0350, 0.0300, 0.0050, 0.0010],
])
P_AXIS = np.linspace(0.0, 0.6, 7)
PMAX = 0.6
DEADBAND = 1.0e-4
L_MAX = 0.20
DT = 0.005           # ログは 200 Hz
CH = ("DF", "F", "G")


def interp2d(z: np.ndarray, x: float, y: float) -> float:
    """pneumatic.interp2d_bilinear と同じ（z[y, x]、軸は両方 P_AXIS）。"""
    g = P_AXIS
    dx = g[1] - g[0]
    x = min(max(x, g[0]), g[-1])
    y = min(max(y, g[0]), g[-1])
    xf = (x - g[0]) / dx
    yf = (y - g[0]) / dx
    x0 = int(min(max(np.floor(xf), 0), len(g) - 2))
    y0 = int(min(max(np.floor(yf), 0), len(g) - 2))
    wx, wy = xf - x0, yf - y0
    r0 = z[y0, x0] * (1 - wx) + z[y0, x0 + 1] * wx
    r1 = z[y0 + 1, x0] * (1 - wx) + z[y0 + 1, x0 + 1] * wx
    return r0 * (1 - wy) + r1 * wy


def simulate(cmd: np.ndarray) -> np.ndarray:
    """1チャンネル分。pam.PneumaticModel.step を1サンプルずつ回す。"""
    n = len(cmd)
    K = int(np.ceil(L_MAX / DT) + 5)
    buf = np.zeros(K)
    wp = 0
    p = 0.0
    prev = 0.0
    latch = 0.0
    last_dir = 0
    out = np.empty(n)
    for i in range(n):
        u = min(max(float(cmd[i]), 0.0), PMAX)
        diff = u - prev
        d = 1 if diff > DEADBAND else (-1 if diff < -DEADBAND else 0)
        if d != 0 and d != last_dir:
            latch = p
            last_dir = d
        prev = u
        tau = max(interp2d(TAU_2D, u, latch), 1e-4)
        L = interp2d(DEAD_2D, u, latch)

        # FractionalDelay.step
        D = min(max(L / DT, 0.0), K - 2.0)
        M = int(np.floor(D))
        mu = D - M
        buf[wp] = u
        delayed = (1 - mu) * buf[(wp - M) % K] + mu * buf[(wp - M - 1) % K]
        wp = (wp + 1) % K

        # first_order_lag
        if tau <= 1e-6:
            p = delayed
        else:
            p = p + DT / (tau + DT) * (delayed - p)
        out[i] = p
    return out


def score(pred: np.ndarray, meas: np.ndarray) -> tuple[float, float]:
    if np.std(pred) < 1e-9 or np.std(meas) < 1e-9:
        return np.nan, np.nan
    r = float(np.corrcoef(pred, meas)[0, 1])
    nrmse = float(np.sqrt(np.mean((pred - meas) ** 2)) / (np.ptp(meas) + 1e-12))
    return r, nrmse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="fwd_model_swap_check.csv")
    args = ap.parse_args()

    files = []
    for d in args.dirs:
        files += sorted(f for f in glob.glob(os.path.join(d, "*.csv"))
                        if "summary" not in os.path.basename(f))

    rows = []
    for f in files:
        df = pd.read_csv(f)
        need = {f"cmd_{c}" for c in CH} | {f"meas_pres_{c}" for c in CH}
        if not need.issubset(df.columns):
            continue
        pred = {c: simulate(df[f"cmd_{c}"].values) for c in CH}
        meas_raw = {c: df[f"meas_pres_{c}"].values for c in CH}
        meas_swp = {"DF": meas_raw["F"], "F": meas_raw["DF"], "G": meas_raw["G"]}
        row = {"file": os.path.basename(f)}
        for tag, meas in (("raw", meas_raw), ("swap", meas_swp)):
            for c in CH:
                r, e = score(pred[c], meas[c])
                row[f"r_{c}_{tag}"] = r
                row[f"nrmse_{c}_{tag}"] = e
        rows.append(row)
        print(f"  {len(rows):3d} {row['file'][:60]}", flush=True)

    res = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    res.to_csv(args.out, index=False)

    print(f"\nN = {len(res)} runs   (NRMSE = RMSE / 実測のレンジ)")
    print(f"{'':6s}{'r (ログのまま)':>16s}{'r (DF/F入替)':>16s}"
          f"{'NRMSE (まま)':>16s}{'NRMSE (入替)':>16s}")
    for c in CH:
        print(f"{c:6s}"
              f"{res[f'r_{c}_raw'].median():16.3f}{res[f'r_{c}_swap'].median():16.3f}"
              f"{res[f'nrmse_{c}_raw'].median():16.1%}{res[f'nrmse_{c}_swap'].median():16.1%}")
    print("（各ランの値の中央値。ランごとの値は", args.out, "）")


if __name__ == "__main__":
    main()
