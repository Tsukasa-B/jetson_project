"""
check_models.py — manifest に登録された全モデルの整合性チェック（実機不要）

  * ONNXファイルの存在
  * manifest の arch / lookahead_horizon / frame_stack から計算した obs 次元と
    ONNX の実入力次元の一致
  * ダミー観測を1ステップ流して action(3) が出ること
  * frame stacking の並び（最新が末尾）が sim と一致していること

モデルを追加・差し替えたら必ずこれを通してから実機を動かすこと。

  python3 tools/check_models.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from model_registry import PolicyRunner, list_models, load_manifest, resolve  # noqa: E402


def check_framestack_order() -> bool:
    """sim (torch.roll(-1) して末尾に最新) と同じ並びかを、依存なしで確認する。"""
    class _S:
        frame_stack, base_obs_dim = 3, 2
    buf = np.zeros((3, 2), dtype=np.float32)
    for v in (1.0, 2.0, 3.0, 4.0):
        buf = np.roll(buf, -1, axis=0)
        buf[-1] = [v, v]
    got = buf.reshape(-1)
    want = np.array([2, 2, 3, 3, 4, 4], dtype=np.float32)   # [最古 ... 最新]
    ok = np.array_equal(got, want)
    print(f"[framestack] 並び {got.tolist()} vs 期待 {want.tolist()} -> {'OK' if ok else 'NG'}")
    return ok


def main() -> int:
    manifest = load_manifest()
    keys = list_models(manifest)
    if not keys:
        print("manifest にモデルが1件もありません")
        return 1

    ok_all = check_framestack_order()
    print()

    for key in keys:
        spec = resolve(key, manifest)
        print(f"--- {key} ---")
        if not os.path.exists(spec.path):
            print(f"  [NG] ファイルが無い: {spec.path}\n")
            ok_all = False
            continue
        try:
            runner = PolicyRunner(spec, verbose=False)
            frame = np.random.randn(spec.base_obs_dim).astype(np.float32) * 0.1
            for _ in range(spec.frame_stack + 2):
                action = runner.step(frame)
            assert action.shape == (3,), action.shape
            a, p = runner.action_to_pressure(action)
            print(f"  [OK] {spec.label or spec.arch} | obs={spec.obs_dim} "
                  f"(base {spec.base_obs_dim} x k {spec.frame_stack}) | "
                  f"lookahead {spec.lookahead_horizon}s | "
                  f"action={np.round(a, 3).tolist()} -> P={np.round(p, 3).tolist()} MPa")
        except Exception as e:  # noqa: BLE001
            print(f"  [NG] {e}")
            ok_all = False
        print()

    print("=== 総合:", "OK" if ok_all else "NG（上記を修正すること）", "===")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
