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
  周波数依存の形が決まらない場合は、大きさの上限で判定する。
  しきい値はすべて EXPECTED_DPDTHETA（期待値）から導き、丸い数字は置かない。
      95%上限 < 期待値/2 → 体積結合ではない。残差の原因は圧力経路側
      95%上限 < 期待値   → あっても期待値より小さい。主因とは考えにくい
      それ以上           → 感度不足。判定できない
  Part C の動作点掃引で dP/dθ が動作点によって変わる
      → ワイヤのたるみ。落ちる動作点がしきい角。Heaviside 項を実測で決められる。
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

# 体積結合が実在した場合に期待される大きさ [kPa/deg]。
# 等温・密閉で dP/P = -dV/V、Δθ=20deg -> Δε=3.3%、dV/V≈2Δε、
# P_abs=0.35 MPa から 23 kPa / 20 deg ≈ 1.1。
# 「応答が小さい」を結論として使うときの基準になるので、
# 丸めたしきい値を別に置かず、すべてここから導く。
EXPECTED_DPDTHETA = 1.1


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

    # --repeats 2 で同じ tag が複数回現れるので、名前ではなく
    # 「連続した同一 tag のかたまり」で切る。名前で groupby すると
    # 1回目と2回目の間の休止まで巻き込む。
    seg = ann["segment"].astype(str).values
    blk = np.concatenate(([0], np.cumsum(seg[1:] != seg[:-1])))
    ann = ann.assign(_blk=blk)

    rows = []
    for _, g in ann.groupby("_blk", sort=True):
        tag = str(g["segment"].iloc[0])
        m = re.match(r"^([A-D])_", tag)
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

        # ノイズ床: 加振周波数の非調和倍での応答の中央値。
        # 1点だけだとノイズ床自体がばらつくので複数点の中央値を取る。
        nz = float(np.median([lockin(p_hold, t, freq * k)[0]
                              for k in (0.71, 1.37, 1.61, 2.29, 2.71)]))

        rows.append(dict(
            segment=tag, part=part, freq_Hz=round(float(freq), 2),
            hold_ch=hold,
            ang_mean_deg=round(float(np.mean(ang)), 1),
            ang_amp_deg=round(a_ang, 2),
            hold_ripple_kPa=round(a_hold, 2),
            noise_kPa=round(nz, 2),
            dPdtheta_kPa_per_deg=round(a_hold / a_ang, 3) if a_ang > 0.5 else np.nan,
            # dP/dθ の 1σ。ロックイン振幅の誤差はノイズ床そのものなので
            # そのまま関節角振幅で割る（角度側の誤差は 0.05deg/2.9deg で無視）。
            sigma=round(nz / a_ang, 3) if a_ang > 0.5 else np.nan,
            phase_lag_deg=round(((ph_hold - ph_ang + 180) % 360) - 180, 1),
            exc_ripple_kPa=round(a_exc, 1),
            snr=round(a_hold / nz, 1) if nz > 0 else np.nan,
        ))

    raw = pd.DataFrame(rows)
    pd.set_option("display.width", 170)

    # 同一条件の繰り返しを平均する（--repeats 2 のとき）。
    n_rep = raw.groupby("segment").size().max() if len(raw) else 1
    if n_rep > 1:
        num = raw.select_dtypes(include=[np.number]).columns
        out = (raw.groupby("segment", sort=False)
                  .agg({**{c: "mean" for c in num},
                        "part": "first", "hold_ch": "first"})
                  .reset_index())
        out["noise_kPa"] /= np.sqrt(n_rep)     # n回平均でノイズ床は 1/sqrt(n)
        out["sigma"] /= np.sqrt(n_rep)
        out["snr"] = out["hold_ripple_kPa"] / out["noise_kPa"]
        out = out.round(3)
        print(f"(同一条件 {n_rep} 回を平均)")
    else:
        out = raw
    print(out.to_string(index=False))

    # ---- 判定順序は Part C が先 -----------------------------------------
    # たるみが有ると Part A / B / D の解釈が汚染される。特に Part A は
    # たるみが有っても「体積結合は実在する」と、もっともらしい fc つきで
    # 断言してしまう（合成ログで fc 2.00 -> 2.77 Hz に上振れしつつ
    # 判定レンジ内に収まった）。順序を運用で守らせるのではなく、
    # Part C の結論が出るまで Part A/B/D の判定文を出さない。
    slack = report_part_c(out, n_rep)

    if slack == "slack":
        print("\n" + "!" * 68)
        print("!! Part C が たるみ 陽性。以下 Part A / B / D の判定は信用しないこと。")
        print("!! たるみは全バーストに効くので、dP/dtheta が一様に小さく出て")
        print("!! fc は上振れする。fc をモデル同定に使ってはいけない。")
        print("!! 先にワイヤを張り直して取り直すこと。")
        print("!" * 68)
    elif slack == "unknown":
        print("\n" + "!" * 68)
        print("!! Part C が判定不能。たるみの有無が確定していないので、")
        print("!! 以下の Part A / B / D は暫定値として読むこと。")
        print("!" * 68)

    # "unknown" のときは結論行そのものを書き換える。
    # バナーだけだと、結論行を拾ってメモに書き写したときに
    # "clean" と区別がつかない。
    pre = "【暫定・たるみ未確定】" if slack == "unknown" else ""

    a_all = out[out["part"] == "A"].sort_values("freq_Hz")
    # SNR が 3 に満たない点は使わない。高周波側は関節角振幅が小さくなるので
    # ここで落ちやすい。落ちた場合は --repeats を増やして取り直す。
    a = a_all[a_all["snr"] >= 3.0]
    if len(a) < len(a_all):
        dropped = ", ".join(f"{r.segment}(snr {r.snr:.1f})"
                            for r in a_all[a_all["snr"] < 3.0].itertuples())
        print(f"\n[除外] SNR<3 のため判定から除外: {dropped}")
    if len(a) >= 3:
        lo = a[a["freq_Hz"] <= 1.8]["dPdtheta_kPa_per_deg"].mean()
        hi = a[a["freq_Hz"] >= 3.0]["dPdtheta_kPa_per_deg"].mean()
        print(f"\n[Part A] dP/dtheta  <=1.8Hz: {lo:.3f}   >=3Hz: {hi:.3f} kPa/deg")

        # 一次ハイパス G*f/sqrt(f^2+fc^2) を当てる。
        # 圧力源が体積変化を打ち消せなくなる遮断周波数 fc が
        # 同定済みの 1/(2*pi*tau) ≈ 1.8 Hz 付近に来れば、
        # 「見えているのは体積結合」という解釈と辻褄が合う。
        f = a["freq_Hz"].values.astype(float)
        y = a["dPdtheta_kPa_per_deg"].values.astype(float)
        ok = np.isfinite(y)
        fc_fit = g_fit = np.nan
        if ok.sum() >= 3:
            best = None
            for fc in np.linspace(0.2, 12.0, 400):
                h = f[ok] / np.hypot(f[ok], fc)
                gain = float(np.sum(y[ok] * h) / np.sum(h * h))
                err = float(np.sum((y[ok] - gain * h) ** 2))
                if best is None or err < best[0]:
                    best = (err, fc, gain)
            _, fc_fit, g_fit = best
            print(f"  一次ハイパス当てはめ: fc = {fc_fit:.2f} Hz, "
                  f"漸近値 G = {g_fit:.2f} kPa/deg "
                  f"(同定済み tau=88ms から予想される fc = 1.8 Hz)")

        # しきい値 1.5 倍の根拠: 一次ハイパスが fc=1.8Hz なら
        # |H(1Hz)|=0.49, |H(3Hz)|=0.86, |H(8Hz)|=0.98 なので
        # 低域平均と高域平均の比は 1.7 程度にしかならない。2倍は厳しすぎる。
        if slack == "slack":
            print("  -> 判定しない（Part C が たるみ 陽性のため）。"
                  "上の数値はたるみに汚染されている。")
        elif (np.isfinite(hi) and hi > EXPECTED_DPDTHETA / 2
                and hi > 1.5 * max(lo, 1e-6)):
            print(f"  -> {pre}高周波で立ち上がっている。体積結合は実在する。")
            if np.isfinite(fc_fit) and 0.9 <= fc_fit <= 4.0:
                # 印は結論行にだけ付ける。ここに重ねると冗長になる。
                print("     fc も予想範囲内。圧力源の追従限界という説明と整合する。")
            if slack == "unknown":
                print("     この結論を確定として扱わないこと。"
                      "先に Part C を --repeats を増やして取り直す。")
        else:
            # 形（周波数依存）では言えなかった。大きさの上限なら言える。
            # ここを「判定保留」で終わらせると、点が3つ以上生き残った
            # ときだけ結論が出ない、という逆転が起きる（点が足りない
            # ときは下の分岐が上限で結論を出すのに）。
            print("  周波数依存の形からは言えないので、大きさの上限で判定する。")
            report_bound(a_all, pre, n_rep)
    elif len(a_all) >= 3:
        # 全点が SNR で落ちた ＝ 応答がどこにも出ていない。
        # これは「測れなかった」ではなく「体積結合が無い」かもしれない。
        # 区別できるのは、期待値 1.1 kPa/deg を検出できるだけの感度が
        # あったかどうか。上限を出して判定する。
        # （Part C と同じ話。フィルタで落とすと、出したい結論の片方が
        #   永久に印字されなくなる。）
        print(f"\n[Part A] 判定に使える点が {len(a)}/{len(a_all)} しかない"
              "（SNR<3 が多い）。周波数依存の形は決められないが、"
              "応答の大きさに上限はつけられる。")
        report_bound(a_all, pre, n_rep)

    if len(out[out["part"] == "B"]) and slack == "slack":
        print("\n[Part B] 判定しない（Part C が たるみ 陽性のため）。"
              "dP/dtheta ∝ P の検証はたるみが無い状態でしか成立しない。")

    d = out[out["part"] == "D"]
    if len(d) and len(a):
        a3 = a[np.isclose(a["freq_Hz"], 3.0, atol=0.3)]
        if len(a3):
            print(f"\n[Part D] 位相 A(3Hz) {a3['phase_lag_deg'].iloc[0]:+.0f} deg "
                  f"vs D {d['phase_lag_deg'].iloc[0]:+.0f} deg "
                  "（拮抗対なら約180度ずれるはず）")
            if slack == "slack":
                # たるみが有ると D の動作点でも応答が消えるので、
                # 「反転しない」がクロストークの証拠にならない。
                print("  -> 根拠に使わない（Part C が たるみ 陽性のため）。"
                      "反転しないのがクロストークのせいか たるみ のせいか"
                      "区別できない。")

    out.to_csv("volume_coupling_result.csv", index=False)
    print("\nwrote volume_coupling_result.csv")


def report_bound(a_all: pd.DataFrame, pre: str, n_rep: int) -> None:
    """3 Hz 以上の応答の大きさに上限をつけて判定する。

    形（周波数依存）が決められないとき、これが唯一言えることになる。
    しきい値はすべて EXPECTED_DPDTHETA から導く。丸い数字を置かない。
    """
    hf = a_all[a_all["freq_Hz"] >= 3.0]
    v = hf["dPdtheta_kPa_per_deg"].values.astype(float)
    sg = hf["sigma"].values.astype(float)
    ok = np.isfinite(v) & np.isfinite(sg) & (sg > 0)
    if ok.sum() < 1:
        print("  -> 有効な点が無い。関節が振れているか確認すること。")
        return
    w = 1.0 / sg[ok] ** 2
    vw = float(np.sum(w * v[ok]) / np.sum(w))
    se = float(1.0 / np.sqrt(np.sum(w)))
    ub = vw + 2 * se
    print(f"  3Hz以上の加重平均 {vw:.3f} +/- {se:.3f} kPa/deg "
          f"(95%上限 {ub:.2f})")
    if ub < EXPECTED_DPDTHETA / 2:
        print(f"  -> {pre}体積結合は無い。期待値 {EXPECTED_DPDTHETA} kPa/deg の"
              f"半分未満（上限 {ub:.2f}）に抑えられている。"
              "残差の原因は圧力経路（時定数・むだ時間・ヒステリシス）側。")
    elif ub < EXPECTED_DPDTHETA:
        print(f"  -> {pre}体積結合があっても期待値より小さい"
              f"（上限 {ub:.2f} kPa/deg）。残差の主因とは考えにくい。")
    else:
        need = int(np.ceil(n_rep * (ub / (EXPECTED_DPDTHETA / 2)) ** 2))
        print(f"  -> 感度不足で判定できない（上限 {ub:.2f} kPa/deg は"
              f"期待値 {EXPECTED_DPDTHETA} より大きい）。"
              f"--repeats {need} 以上か、圧力センサのノイズを先に減らすこと。")


def report_part_c(out: pd.DataFrame, n_rep: int) -> str:
    """Part C を判定し "clean" / "slack" / "unknown" を返す。"""
    c = out[out["part"] == "C"].sort_values("ang_mean_deg")
    if len(c) >= 2:
        print("\n[Part C] 動作点依存性（たるみの有無）＝ 最初に読むべきブロック")
        print(c[["segment", "ang_mean_deg", "ang_amp_deg",
                 "hold_ripple_kPa", "dPdtheta_kPa_per_deg", "sigma", "snr"]]
              .to_string(index=False))

        # 「何%ばらついたら たるみ とみなすか」を固定値で決めない。
        # 各点には測定誤差 sigma があるので、ばらつきが誤差だけで説明できるか
        # を chi2 で見る。定数モデル（＝たるみ無し）に対して
        #   chi2 = sum((v_i - v_w)^2 / sigma_i^2),  dof = n-1
        # 誤差だけで説明できるなら chi2/dof ~ 1 になる。
        # Part A と違い、ここでは SNR の低い点を捨てない。
        # たるみが有るなら小角側の応答は「ゼロ」になるのが予測であって、
        # SNR<3 はノイズではなく信号そのもの。捨てると陽性が判定不能になる。
        # （Part A では低SNR点は単なるノイズで、物理はゼロでない漸近値を
        #   予測しているので、除外が正しい。）
        v = c["dPdtheta_kPa_per_deg"].values.astype(float)
        sg = c["sigma"].values.astype(float)
        good = np.isfinite(v) & np.isfinite(sg) & (sg > 0)
        if good.sum() >= 2:
            w = 1.0 / sg[good] ** 2
            vw = float(np.sum(w * v[good]) / np.sum(w))
            chi2 = float(np.sum(((v[good] - vw) / sg[good]) ** 2))
            dof = int(good.sum() - 1)
            red = chi2 / dof
            spread = (np.max(v[good]) - np.min(v[good])) / vw
            print(f"  加重平均 {vw:.2f} kPa/deg, 実測ばらつき {spread:.0%}, "
                  f"chi2/dof = {red:.2f} (dof={dof})")
            if red < 2.0:
                print("  -> ばらつきは測定誤差だけで説明できる。たるみの証拠なし。")
                return "clean"
            if red > 4.0:
                lo = c[good].sort_values("ang_mean_deg").iloc[0]
                print(f"  -> 誤差では説明できないばらつき。"
                      f"最小は平均 {lo['ang_mean_deg']:.0f} deg 付近。"
                      "たるみのしきい角の候補。")
                d = c[good].sort_values("ang_mean_deg")
                if np.all(np.diff(d["dPdtheta_kPa_per_deg"].values) > 0):
                    r0 = d.iloc[0]
                    if abs(r0["dPdtheta_kPa_per_deg"]) < 2 * r0["sigma"]:
                        print(f"     最小角 {r0['ang_mean_deg']:.0f} deg の応答は"
                              f"ゼロと区別できない ({r0['dPdtheta_kPa_per_deg']:.2f}"
                              f" +/- {r0['sigma']:.2f})。たるみの典型的な形。")
                    else:
                        print("     関節角に対して単調増加。たるみと整合する。")
                else:
                    print("     ただし関節角に対して単調増加ではない。"
                          "たるみなら小角側から単調に増えるはずなので、"
                          "外れ値や計測系の異常も疑うこと。")
                return "slack"
            need = int(np.ceil(n_rep * (red / 2.0) ** 2))
            print(f"  -> 判定不能（誤差とも有意差とも言えない）。"
                  f"--repeats {need} 以上で取り直すこと。")
            return "unknown"
        print("  -> 有効な点が足りない。判定不可。")
    return "unknown"


if __name__ == "__main__":
    main()
