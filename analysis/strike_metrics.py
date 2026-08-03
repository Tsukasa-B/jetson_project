"""
strike_metrics.py — 実機ログから ±30ms成功率を出す（sim と同一定義）

sim側の正準実装 `porcaro_2026/analysis/harness/strike_extract.py::extract_strikes`
を実機CSV向けに移植したもの。判定ロジック・パラメータは**一切変えていない**:

  T16th        = 15 / bpm  [s]
  目標打点     = target_force のピーク
                 (height = min_strike_frac * target_ref, distance = 0.3*T16th)
  実打点       = 各目標時刻の ±1.5*T16th 窓内での force のargmax
  success      = peak_force >= 0.25*target_ref かつ |timing_err| <= 30ms

列名の対応:
  sim  time_s / target_force / force_z
  実機 time   / target_force / force_N

★実機固有の注意（simと違う点、ここだけ実装で吸収している）:
  実機CSVはセンサ200Hz行に50Hzの指令列を merge_asof で貼っているため、
  target_force が1制御ステップにつき4サンプルの階段状になる。そのまま
  find_peaks に掛けると平坦ピークの中央が返り、目標時刻が数msズレる。
  そこで **目標時刻は指令の変化点だけを取り出した50Hz系列から** 求め、
  実打点の探索は200Hzの生信号で行う。

  力センサのゼロ点補正: --force_offset で指定。既定は 0.0（無補正）。
  無負荷時の force_N が0から大きくずれている場合、閾値判定
  (peak_force >= 0.25*20N = 5N) が壊れるので必ず確認すること。

Usage:
  # 1ファイル
  python3 analysis/strike_metrics.py results/RAL/modelE_seed1/deploy_xxx.csv

  # ディレクトリ配下を再帰的に集計してサマリCSVを出す
  python3 analysis/strike_metrics.py results/RAL --summary results/RAL/summary.csv

  # ゼロ点補正をかける
  python3 analysis/strike_metrics.py results/RAL --force_offset -20.0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# --- sim と同一の定数（strike_extract.py の既定値） ---
TOL_MS = 30.0
MIN_STRIKE_FRAC = 0.25
TARGET_REF_DEFAULT = 20.0


def extract_strikes_real(
    df: pd.DataFrame,
    bpm: float,
    target_ref: float = TARGET_REF_DEFAULT,
    tol_ms: float = TOL_MS,
    min_strike_frac: float = MIN_STRIKE_FRAC,
    force_col: str = "force_N",
    time_col: str = "time",
    force_offset: float = 0.0,
    force_threshold: float | None = None,
) -> pd.DataFrame:
    """force_threshold: 打撃と認める最小の力[N]。None なら sim準拠の
    min_strike_frac * target_ref (= 0.25 x 20 = 5 N)。
    ★1N など変更する場合、sim側 strike_extract.py も同じ値に揃えないと比較が壊れる。"""
    """実機CSVから打点ごとの結果を返す。sim の extract_strikes と同一判定。"""
    need = {time_col, "target_force", force_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"列が足りません: {missing}")

    t = df[time_col].to_numpy(float)
    tgt = df["target_force"].to_numpy(float)
    f = df[force_col].to_numpy(float) - force_offset

    ok = np.isfinite(t) & np.isfinite(tgt) & np.isfinite(f)
    t, tgt, f = t[ok], tgt[ok], f[ok]
    if len(t) < 2:
        raise ValueError("サンプルが少なすぎます")

    if bpm <= 0:
        raise ValueError(f"不正なbpm={bpm}")
    t16 = 15.0 / bpm
    thr = force_threshold if force_threshold is not None else min_strike_frac * target_ref

    # --- 目標時刻: 指令の変化点だけを取った50Hz系列から求める ---
    # (200Hzに引き伸ばされた階段の平坦ピーク中央化を避けるため)
    chg = np.r_[True, np.diff(tgt) != 0.0]
    t_cmd, tgt_cmd = t[chg], tgt[chg]
    dt_cmd = float(np.median(np.diff(t_cmd))) if len(t_cmd) > 1 else 0.02
    if dt_cmd <= 0:
        dt_cmd = 0.02
    distance = max(1, int(round(0.3 * t16 / dt_cmd)))
    tgt_peak_idx, _ = find_peaks(tgt_cmd, height=min_strike_frac * target_ref, distance=distance)
    target_times = t_cmd[tgt_peak_idx]

    # --- 実打点: 200Hzの生信号で ±1.5*T16th のargmax ---
    half_win = 1.5 * t16
    rows = []
    for i, tt in enumerate(target_times):
        mask = (t >= tt - half_win) & (t <= tt + half_win)
        if not mask.any():
            rows.append(dict(strike_idx=i, target_time=float(tt), peak_time=np.nan,
                             timing_err_ms=np.nan, peak_force=0.0, success=False))
            continue
        lt, lf = t[mask], f[mask]
        j = int(np.argmax(lf))
        err_ms = (float(lt[j]) - float(tt)) * 1000.0
        pf = float(lf[j])
        rows.append(dict(
            strike_idx=i, target_time=float(tt), peak_time=float(lt[j]),
            timing_err_ms=err_ms, peak_force=pf,
            success=bool(pf >= thr and abs(err_ms) <= tol_ms),
        ))
    return pd.DataFrame(rows, columns=["strike_idx", "target_time", "peak_time",
                                       "timing_err_ms", "peak_force", "success"])


def apply_time_source(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """センサ行のタイムスタンプをどう作るか。

    recon          : time = index/200（IROS時からの既定。200Hz完全という仮定）
    rate_corrected : reconの等間隔性は保ったまま、全体の長さを t_recv_rel の
                     実測経過時間に一致させる。MicroLabBox/Jetson間のクロック
                     レート差（曲の後半ほど効く系統ドリフト）を除去する。★推奨
    recv           : t_recv_rel をそのまま使う。USBのバースト受信でジッタが
                     乗るが、制御ループと同じ時計なのでドリフトは無い。
    """
    df = df.copy()
    if mode == "recon" or "t_recv_rel" not in df.columns:
        return df
    n = len(df)
    if mode == "recv":
        df["time"] = df["t_recv_rel"].to_numpy(float)
    elif mode == "rate_corrected":
        span = float(df["t_recv_rel"].iloc[-1] - df["t_recv_rel"].iloc[0])
        if n > 1 and span > 0:
            df["time"] = np.arange(n) * (span / (n - 1))
    else:
        raise ValueError(f"未知の time_source: {mode}")
    return df


def summarize_file(csv_path: str, force_offset: float = 0.0,
                   time_source: str = "recon", include_mock: bool = False,
                   include_suspect: bool = False, force_threshold: float | None = None,
                   strike_sink: list | None = None) -> dict | None:
    """1ラン分のCSV+JSONを読んで、1行のサマリdictを返す。"""
    json_path = os.path.splitext(csv_path)[0] + ".json"
    meta = {}
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as fh:
            meta = json.load(fh)

    # ★ mock実行 / verify実行は実験データではないので既定で除外
    if not include_mock and (meta.get("mock") or meta.get("verify")):
        return None

    df = pd.read_csv(csv_path)

    # ★ データ品質ガード: 力センサが死んでいたランを弾く。
    #   アンプ非通電時は force_N が大きな負の定数付近に張り付き、レンジも出ない。
    #   これを混ぜると成功率が静かに0%側へ引っ張られるので、既定で除外する。
    if "force_N" in df.columns and len(df) > 10:
        f_med = float(df["force_N"].median())
        f_rng = float(df["force_N"].max() - df["force_N"].min())
        if f_med < -5.0 or f_rng < 10.0:
            if not include_suspect:
                print(f"[SUSPECT] {os.path.basename(csv_path)}: "
                      f"force_N 中央値 {f_med:.1f}N / レンジ {f_rng:.1f}N "
                      f"→ センサ異常の疑いで除外（--include_suspect で強制集計）")
                return None

    df = apply_time_source(df, time_source)
    if "target_force" not in df.columns:
        return None  # verify実行など、指令列が無いもの

    bpm = float(meta.get("bpm", 0) or 0)
    if bpm <= 0:  # JSONが無い旧ログはファイル名からbpmを拾う
        import re
        m = re.search(r"bpm(\d+)", os.path.basename(csv_path))
        if not m:
            return None
        bpm = float(m.group(1))

    target_ref = float(meta.get("target_force", TARGET_REF_DEFAULT))
    st = extract_strikes_real(df, bpm=bpm, target_ref=target_ref, force_offset=force_offset,
                              force_threshold=force_threshold)
    if st.empty:
        return None

    thr = force_threshold if force_threshold is not None else MIN_STRIKE_FRAC * target_ref
    hit = st[st["peak_force"] >= thr]

    if strike_sink is not None:
        s2 = st.copy()
        s2["model_key"] = meta.get("model_key", "")
        s2["model"] = str(meta.get("model_key", "")).split("/")[-1].split("_")[0]
        s2["seed"] = meta.get("manifest_extra", {}).get("seed", "")
        s2["midi"] = os.path.basename(meta.get("midi", ""))
        s2["trial"] = meta.get("trial", "")
        s2["force_threshold"] = thr
        strike_sink.append(s2)

    # 誤差が「一定オフセット」か「時間とともに増えるドリフト」かを切り分ける
    drift_ms_per_s = np.nan
    valid = st.dropna(subset=["timing_err_ms"])
    if len(valid) >= 3:
        drift_ms_per_s = float(np.polyfit(valid["target_time"], valid["timing_err_ms"], 1)[0])

    return {
        "file": os.path.relpath(csv_path),
        "model_key": meta.get("model_key", ""),
        "midi": os.path.basename(meta.get("midi", "")),
        "bpm": bpm,
        "trial": meta.get("trial", ""),
        "n_target_strikes": len(st),
        "success_rate": float(st["success"].mean()),
        "abs_err_ms_mean": float(st.loc[hit.index, "timing_err_ms"].abs().mean()) if len(hit) else np.nan,
        "err_ms_mean": float(st.loc[hit.index, "timing_err_ms"].mean()) if len(hit) else np.nan,
        "err_ms_std": float(st.loc[hit.index, "timing_err_ms"].std()) if len(hit) else np.nan,
        "peak_force_mean": float(st["peak_force"].mean()),
        "miss_rate_force": float((st["peak_force"] < thr).mean()),
        "drift_ms_per_s": drift_ms_per_s,
        "time_source": time_source,
        "packet_yield": meta.get("packet_yield", np.nan),
        "force_offset_applied": force_offset,
    }


def main():
    ap = argparse.ArgumentParser(description="実機ログの±30ms成功率（sim同一定義）")
    ap.add_argument("path", help="CSVファイル、またはディレクトリ（再帰探索）")
    ap.add_argument("--force_offset", type=float, default=0.0,
                    help="force_N のゼロ点補正 [N]。無負荷時の平均値を指定する")
    ap.add_argument("--summary", type=str, default=None, help="サマリCSVの出力先")
    ap.add_argument("--detail", action="store_true", help="打点ごとの明細も表示")
    ap.add_argument("--time_source", default="rate_corrected",
                    choices=["recon", "rate_corrected", "recv"],
                    help="センサ時刻の作り方（既定 rate_corrected）")
    ap.add_argument("--compare_time_sources", action="store_true",
                    help="3つの時刻定義で成功率を並べ、時計依存性を確認する")
    ap.add_argument("--include_mock", action="store_true", help="mock/verify実行も含める")
    ap.add_argument("--force_threshold", type=float, default=None,
                    help="打撃と認める最小の力[N]。既定は sim準拠の 5N (=0.25x20N)")
    ap.add_argument("--dump_strikes", type=str, default=None,
                    help="打点ごとの明細CSVを保存（タイミング誤差分布の図に使う）")
    ap.add_argument("--include_suspect", action="store_true",
                    help="力センサ異常の疑いがあるランも集計に含める")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "**", "*.csv"), recursive=True))
        files = [f for f in files if "summary" not in os.path.basename(f)]
    else:
        files = [args.path]

    rows = []
    sink = [] if args.dump_strikes else None
    for f in files:
        try:
            r = summarize_file(f, args.force_offset, args.time_source, args.include_mock,
                               args.include_suspect, args.force_threshold, sink)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {os.path.basename(f)}: {e}")
            continue
        if r is None:
            continue
        rows.append(r)
        print(f"{r['model_key']:>16s} {r['midi']:<26s} tr{r['trial']} | "
              f"打点{r['n_target_strikes']:3d} | ±30ms {r['success_rate']*100:5.1f}% | "
              f"err {r['err_ms_mean']:+6.1f}ms |err| {r['abs_err_ms_mean']:5.1f}ms | "
              f"drift {r['drift_ms_per_s']:+5.2f}ms/s | 力 {r['peak_force_mean']:5.1f}N")
        if args.compare_time_sources:
            for ts in ("recon", "rate_corrected", "recv"):
                rr = summarize_file(f, args.force_offset, ts, args.include_mock,
                                    args.include_suspect)
                if rr:
                    print(f"      [{ts:>14s}] ±30ms {rr['success_rate']*100:5.1f}% | "
                          f"err {rr['err_ms_mean']:+6.1f}ms | drift {rr['drift_ms_per_s']:+5.2f}ms/s")
        if args.detail:
            st_df = extract_strikes_real(pd.read_csv(f),
                                         bpm=r["bpm"], force_offset=args.force_offset)
            print(st_df.to_string(index=False))

    if not rows:
        print("集計対象が見つかりませんでした。")
        return 1

    summary = pd.DataFrame(rows)
    print(f"\n=== {len(summary)} ラン ===")
    if summary["model_key"].nunique() > 1:
        agg = summary.groupby("model_key").agg(
            runs=("success_rate", "size"),
            success_rate=("success_rate", "mean"),
            abs_err_ms=("abs_err_ms_mean", "mean"),
        )
        print(agg.to_string())

    if args.dump_strikes and sink:
        allst = pd.concat(sink, ignore_index=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_strikes)) or ".", exist_ok=True)
        allst.to_csv(args.dump_strikes, index=False)
        print(f"[saved] {args.dump_strikes}  ({len(allst)} 打点)")

    if args.summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary)) or ".", exist_ok=True)
        summary.to_csv(args.summary, index=False)
        print(f"\n[saved] {args.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
