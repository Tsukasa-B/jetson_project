"""
compare_pressure_models.py — 圧力モデルの構造比較

  T : シミュレータのテーブル型（τ(Pcmd,Pstart), L(Pcmd,Pstart), 方向反転ラッチ）
  S : 対称な一次遅れ（τ一定）＋むだ時間
  O : 流量制限付きモデル（小信号は一次遅れ、充填は供給圧側・排気は大気側の
      オリフィス流量式 ISO 6358 型で速度の上限が決まる）

S と O は ① エコー付きステップ（通信を含まない）だけで同定し、
② 2月の検証データ（ステップ・チャープ・準静的）、③ 打撃中の105ラン で検証する。
②③には通信などの一定遅れ δ を、モデルごとに同じ自由度で1つだけ許す。
"""
import glob, sys, json
import numpy as np, pandas as pd
from numba import njit
from scipy.optimize import least_squares

DT = 0.005
PA = 0.1013
sys.path.insert(0, "analysis")
from check_forward_model_swap import TAU_2D, DEAD_2D  # noqa: E402

G = np.linspace(0, 0.6, 7)


@njit(cache=True)
def _i2(z, x, y):
    x = min(max(x, 0.0), 0.6); y = min(max(y, 0.0), 0.6)
    xf = x / 0.1; yf = y / 0.1
    x0 = int(min(max(np.floor(xf), 0), 5)); y0 = int(min(max(np.floor(yf), 0), 5))
    wx = xf - x0; wy = yf - y0
    r0 = z[y0, x0] * (1 - wx) + z[y0, x0 + 1] * wx
    r1 = z[y0 + 1, x0] * (1 - wx) + z[y0 + 1, x0 + 1] * wx
    return r0 * (1 - wy) + r1 * wy


@njit(cache=True)
def sim_T(cmd, TAU, DEAD):
    n = len(cmd); K = 45; buf = np.zeros(K); wp = 0
    p = 0.0; prev = 0.0; latch = 0.0; last = 0; out = np.empty(n)
    for i in range(n):
        u = min(max(cmd[i], 0.0), 0.6); df = u - prev
        d = 1 if df > 1e-4 else (-1 if df < -1e-4 else 0)
        if d != 0 and d != last:
            latch = p; last = d
        prev = u
        tau = max(_i2(TAU, u, latch), 1e-4); L = _i2(DEAD, u, latch)
        D = min(max(L / DT, 0.0), K - 2.0); M = int(np.floor(D)); mu = D - M
        buf[wp] = u
        dl = (1 - mu) * buf[(wp - M) % K] + mu * buf[(wp - M - 1) % K]
        wp = (wp + 1) % K
        p = p + DT / (tau + DT) * (dl - p)
        out[i] = p
    return out


@njit(cache=True)
def _delay(cmd, L):
    n = len(cmd); out = np.empty(n); D = L / DT; M = int(np.floor(D)); mu = D - M
    for i in range(n):
        a = cmd[i - M] if i - M >= 0 else cmd[0]
        b = cmd[i - M - 1] if i - M - 1 >= 0 else cmd[0]
        out[i] = (1 - mu) * a + mu * b
    return out


@njit(cache=True)
def sim_S(cmd, tau, L):
    u = _delay(cmd, L); p = 0.0; out = np.empty(len(u)); a = DT / (tau + DT)
    for i in range(len(u)):
        p += a * (u[i] - p); out[i] = p
    return out


@njit(cache=True)
def _g(pu, pd, b):
    r = pd / pu
    if r <= b:
        return pu
    x = (r - b) / (1 - b)
    return pu * np.sqrt(max(1 - x * x, 0.0))


@njit(cache=True)
def sim_O(cmd, tau, L, cin, cout, ps, b):
    u = _delay(cmd, L); p = 0.0; out = np.empty(len(u))
    for i in range(len(u)):
        q = (u[i] - p) / tau
        pab = p + PA
        if q > 0:
            lim = cin * _g(ps + PA, pab, b) if ps + PA > pab else 0.0
            q = min(q, lim)
        else:
            lim = cout * _g(pab, PA, b) if pab > PA else 0.0
            q = max(q, -lim)
        p += q * DT; out[i] = p
    return out


def run(model, th, cmd):
    if model == "T":
        return sim_T(cmd, TAU_2D, DEAD_2D)
    if model == "S":
        return sim_S(cmd, *th)
    return sim_O(cmd, *th)


def shift(p, s):
    return p if s == 0 else np.concatenate([np.full(s, p[0]), p[:-s]])


def metr(p, m):
    return (float(np.corrcoef(p, m)[0, 1]),
            float(np.sqrt(np.mean((p - m) ** 2)) / (np.ptp(m) + 1e-12)))


# ------------------------------------------------------------------ data
def load_echo():
    out = []
    for f in ["test_signals/data_echo_grid_1790668422.csv",
              "test_signals/data_echo_steps_1790667361.csv",
              "test_signals/data_echo_steps_1790666997.csv"]:
        d = pd.read_csv(f)
        u = d.flag.values.astype(float)
        i0 = np.argmax(u > 0.05)          # 最初の指令が届く前（flag=0）は捨てる
        out.append((f.split("/")[-1], u[i0:], d.meas_pres_F.values[i0:]))   # 物理DF
    return out


def load_feb():
    out = []
    for e, k in [("exp1_static_hysteresis", 1.027), ("exp2_step_response", 1.032),
                 ("exp3_frequency_sweep", 1.030)]:
        d = pd.read_csv(glob.glob(f"IROS/measured/data_{e}_*.csv")[0])
        s = pd.read_csv(f"test_signals/{e}.csv")
        tc = np.arange(0, s.time.iloc[-1] + 0.02, DT)
        u = np.interp(np.floor(tc / 0.02) * 0.02, s.time, s.cmd_pressure_DF)
        m = np.interp(tc, d.time.values / k, d.meas_pres_DF.values)
        out.append((e, u, m))
    return out


def load_ral():
    out = []
    for f in sorted(glob.glob("data/ral_20260803/*.csv")):
        if "summary" in f:
            continue
        d = pd.read_csv(f)
        meas = {"DF": d.meas_pres_F.values, "F": d.meas_pres_DF.values, "G": d.meas_pres_G.values}
        for c in ("DF", "F", "G"):
            out.append((c, d[f"cmd_{c}"].values.astype(float), meas[c]))
    return out


# ------------------------------------------------------------------ fit on ①
echo = load_echo()


def resid(th, model):
    r = []
    for _, u, m in echo:
        p = run(model, th, u)
        r.append((p - m)[::4])
    return np.concatenate(r)


fits = {}
fits["S"] = least_squares(resid, [0.08, 0.04], args=("S",),
                          bounds=([0.01, 0.0], [0.3, 0.1])).x
fits["O"] = least_squares(resid, [0.05, 0.04, 3.0, 3.0, 0.6, 0.3], args=("O",),
                          bounds=([0.005, 0.0, 0.1, 0.1, 0.45, 0.05],
                                  [0.3, 0.1, 50, 50, 0.9, 0.6])).x
fits["T"] = np.array([])
print("① で同定したパラメータ")
print("  S: tau=%.3f s, L=%.3f s" % tuple(fits["S"]))
print("  O: tau=%.3f s, L=%.3f s, c_in=%.2f, c_out=%.2f /s, Ps=%.3f MPa, b=%.2f" % tuple(fits["O"]))


def evaluate(name, data, allow_shift, per_item=False):
    rows = {}
    for mdl in ("T", "S", "O"):
        preds = [(run(mdl, fits[mdl], u), m, tag) for tag, u, m in data]
        best_s = 0
        if allow_shift:
            errs = []
            for s in range(0, 17):
                errs.append(np.mean([np.mean((shift(p, s) - m) ** 2) for p, m, _ in preds]))
            best_s = int(np.argmin(errs))
        sc = [metr(shift(p, best_s), m) for p, m, _ in preds]
        rows[mdl] = (best_s * 5, sc, [t for _, _, t in preds])
    print(f"\n== {name} ==")
    if per_item:
        tags = rows["T"][2]
        for i, t in enumerate(tags):
            line = "  ".join(f"{m}: r={rows[m][1][i][0]:.3f} N={rows[m][1][i][1]:.1%}" for m in "TSO")
            print(f"  {t:36s} {line}")
        print("  追加遅れ δ: " + "  ".join(f"{m}={rows[m][0]}ms" for m in "TSO"))
    else:
        tags = np.array(rows["T"][2])
        for ch in ("DF", "F", "G"):
            k = tags == ch
            line = "  ".join(
                f"{m}: r={np.median(np.array(rows[m][1])[k,0]):.3f} N={np.median(np.array(rows[m][1])[k,1]):.1%}"
                for m in "TSO")
            print(f"  {ch:3s} {line}")
        print("  追加遅れ δ: " + "  ".join(f"{m}={rows[m][0]}ms" for m in "TSO"))


evaluate("① エコー付きステップ（同定に使ったデータ・通信なし）", echo, False, per_item=True)
evaluate("② 2月の検証データ（時間伸び補正済み）", load_feb(), True, per_item=True)
evaluate("③ 打撃中の105ラン（中央値）", load_ral(), True)
import os
os.makedirs("out/jfps", exist_ok=True)
json.dump({k: list(map(float, v)) for k, v in fits.items()}, open("out/jfps/model_fits.json", "w"))
