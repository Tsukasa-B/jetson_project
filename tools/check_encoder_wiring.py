"""
check_encoder_wiring.py — 手首/ハンド関節エンコーダの配線が入れ替わっていないか、
                          IROS時のデータ（配線が正しかった時期）と比較して判定する。

実機に触らずログだけで判定できる。物理確認ができないときに使う。

原理:
  IROS/deploy_results/ の90ファイル（配線が正しかった時期）では、
  2つの角度チャンネルの統計的な指紋がはっきり分かれている:

      wrist_angle_deg : 平均 33.96 ± 11.09 deg,  範囲 64.3 deg,  力との相関 0.13
      grip_angle_deg  : 平均  7.41 ±  4.53 deg,  範囲 42.6 deg,  力との相関 0.03

  平均値で 90 ファイル中 88 が wrist > grip。手首は打撃を駆動するので
  可動域が広く、力センサとの相関も 4 倍強い。
  検査対象でこの大小関係が反転していれば、配線が逆になっている。

Usage:
  python3 tools/check_encoder_wiring.py results/RAL_session1
  python3 tools/check_encoder_wiring.py results/RAL_session2 results/RAL_session3
  python3 tools/check_encoder_wiring.py results/RAL_session1 --ref IROS/deploy_results
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

NEED = {"wrist_angle_deg", "grip_angle_deg"}


def signature(directory: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)):
        if "summary" in os.path.basename(f):
            continue
        try:
            d = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        if not NEED.issubset(d.columns) or len(d) < 100:
            continue
        w = d["wrist_angle_deg"].to_numpy(float)
        g = d["grip_angle_deg"].to_numpy(float)
        if not (np.isfinite(w).all() and np.isfinite(g).all()):
            continue
        # 動いていないログ（verify等）は判定に使えない
        if (w.max() - w.min()) < 5 and (g.max() - g.min()) < 5:
            continue
        row = dict(file=os.path.basename(f),
                   w_mean=w.mean(), g_mean=g.mean(),
                   w_rng=w.max() - w.min(), g_rng=g.max() - g.min())
        if "force_N" in d.columns:
            F = d["force_N"].to_numpy(float)
            if np.std(F) > 1e-9:
                row["w_corrF"] = abs(np.corrcoef(w, F)[0, 1])
                row["g_corrF"] = abs(np.corrcoef(g, F)[0, 1])
        rows.append(row)
    return pd.DataFrame(rows)


def report(name: str, s: pd.DataFrame) -> dict:
    n = len(s)
    print(f"\n--- {name}  (n={n} ファイル) ---")
    if n == 0:
        print("  角度列を含むCSVが見つかりません")
        return {}
    out = {}
    for key, lab in [("mean", "平均値"), ("rng", "範囲")]:
        a, b = s[f"w_{key}"], s[f"g_{key}"]
        frac = float((a > b).mean())
        out[key] = frac
        print(f"  {lab:<6s}: wrist {a.mean():7.2f}±{a.std():5.2f}   "
              f"grip {b.mean():7.2f}±{b.std():5.2f}   "
              f"wrist>grip の割合 {frac*100:5.1f}%")
    if "w_corrF" in s.columns:
        a, b = s["w_corrF"].dropna(), s["g_corrF"].dropna()
        if len(a):
            frac = float((s["w_corrF"] > s["g_corrF"]).mean())
            out["corrF"] = frac
            print(f"  力相関: wrist {a.mean():7.3f}        grip {b.mean():7.3f}"
                  f"        wrist>grip の割合 {frac*100:5.1f}%")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+", help="検査するディレクトリ")
    ap.add_argument("--ref", default="IROS/deploy_results",
                    help="基準（配線が正しかった時期のデータ）")
    args = ap.parse_args()

    ref_sig = signature(args.ref)
    ref = report(f"[基準] {args.ref}", ref_sig)
    if not ref:
        print("基準データが読めません。--ref を確認してください。")
        return 1

    print("\n" + "=" * 66)
    for t in args.targets:
        tgt = report(f"[検査] {t}", signature(t))
        if not tgt:
            continue
        votes = []
        for k, lab in [("mean", "平均値"), ("rng", "範囲"), ("corrF", "力相関")]:
            if k in ref and k in tgt:
                same = (ref[k] > .5) == (tgt[k] > .5)
                votes.append(same)
                print(f"    {lab:<6s}の大小関係: 基準 {'wrist>grip' if ref[k]>.5 else 'grip>wrist'}"
                      f"  /  検査 {'wrist>grip' if tgt[k]>.5 else 'grip>wrist'}"
                      f"   → {'一致' if same else '★反転'}")
        if not votes:
            continue
        agree = sum(votes)
        print()
        if agree == len(votes):
            print(f"  判定: 配線は基準と同じ ({agree}/{len(votes)} 指標が一致)")
        elif agree == 0:
            print(f"  判定: ★★ 配線が入れ替わっている ({len(votes)}/{len(votes)} 指標が反転) ★★")
            print(f"        → 今後は --swap_encoders を付けて実行すること")
        else:
            print(f"  判定: 判別不能 ({agree}/{len(votes)} 一致)。物理確認が必要")
        print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
