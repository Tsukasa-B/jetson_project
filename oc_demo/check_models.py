"""models/ 以下の .onnx の入出力を表示し、oc_demo の解釈と突き合わせる。

    python3 -m oc_demo.check_models
    python3 -m oc_demo.check_models --models-dir models

ここで出る obs次元 / lookahead / RNN判定 が、run_rl_deploy_midi.py の想定と
一致しているかを目視で確認してください。特に LSTM モデルで
「RNN入力はあるのに出力に状態が返っていない」場合は **静かに間違います**
（毎ステップ h/c がゼロのまま = 記憶なしで動いてしまう）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .policy import OnnxPolicy, discover_models


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNXモデルの入出力を確認する")
    ap.add_argument("--models-dir", default="models")
    args = ap.parse_args()

    models = discover_models(args.models_dir)
    if not models:
        print(f"[check_models] {args.models_dir} に .onnx がありません")
        return

    for m in models:
        print("=" * 66)
        print(f"{m['id']}   （表示名 {m['label']}）")
        try:
            pol = OnnxPolicy(m["path"])
        except Exception as exc:  # noqa: BLE001
            print(f"  !! 読み込めません: {exc.__class__.__name__}: {exc}")
            continue

        print(f"  実行プロバイダ : {pol.providers}")
        print("  入力:")
        for name, spec in pol.inputs.items():
            mark = " ← obs" if name == pol.obs_name else " ← RNN状態" if pol.is_rnn else ""
            print(f"    {name:<24} {spec.shape}{mark}")
        outs = pol.sess.get_outputs()
        print("  出力:")
        for o in outs:
            print(f"    {o.name:<24} {o.shape}")

        print(f"  → obs次元 {pol.obs_dim} / lookahead {pol.lookahead_steps} step "
              f"({pol.lookahead_steps * 0.02:.2f} 秒) / RNN {'あり' if pol.is_rnn else 'なし'}")

        # 1回だけ推論して形を確認
        try:
            a = pol.act(np.zeros(pol.obs_dim, dtype=np.float32), 0)
            print(f"  ゼロobsでの推論 OK  action={np.round(a, 4)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! 推論できません: {exc}")
            continue

        if pol.is_rnn:
            if len(outs) < 1 + len(pol.rnn_inputs):
                print("  !! RNN入力があるのに出力に状態が返っていません。")
                print("     このままだと毎ステップ h/c がゼロ、つまり『記憶なし』で動きます。")
                print("     run_rl_deploy_midi.py のRNN分岐と入出力の対応を確認してください。")
            else:
                nz = any(np.any(v != 0) for v in pol._rnn_state.values())
                print(f"  RNN状態の更新: {'されています' if nz else '値がゼロのままです（要確認）'}")

    print("=" * 66)


if __name__ == "__main__":
    main()
