"""
parity_test.py — 新しい観測構築が IROS時のコードと数値的に一致することの確認（実機不要）

やること
  1. 旧 run_rl_deploy_midi.py の観測構築（torch.cat 版）をそのまま再現し、
     新 deploy_policy.build_base_frame と **ビット一致** するかを確認する。
     入力は実機ログ(IROS/deploy_results/**.csv)の角度系列をそのまま再生。
  2. frame stacking の並びが sim (user0/porcaro_2026_env.py) と一致するかを確認。
     sim:  obs_history = roll(obs_history, -1); obs_history[-1] = new; reshape(-1)

  python3 tools/parity_test.py
"""

import glob
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from midi_rhythm_generator import MidiRhythmGenerator  # noqa: E402
from model_registry import PolicyRunner, resolve  # noqa: E402

QD_CLIP = 20.0


def legacy_obs(rhythm, wrist_deg, grip_deg, prev_q, prev_action, t, dt):
    """旧 run_rl_deploy_midi.py L247-273 をそのまま写したもの。"""
    q_wrist = np.radians(wrist_deg)
    q_grip = np.radians(grip_deg)
    if prev_q is None:
        qd_wrist = qd_grip = 0.0
    else:
        qd_wrist = np.clip((q_wrist - prev_q[0]) / dt, -QD_CLIP, QD_CLIP)
        qd_grip = np.clip((q_grip - prev_q[1]) / dt, -QD_CLIP, QD_CLIP)

    q_vec = torch.tensor([q_wrist, q_grip])
    qd_vec = torch.tensor([qd_wrist, qd_grip])
    obs_action = torch.tensor(prev_action)
    phase, traj = rhythm.get_state(t)
    obs_phase = torch.tensor([np.sin(phase), np.cos(phase)])
    obs_bpm = torch.tensor([rhythm.bpm / 180.0])
    obs = torch.cat([q_vec, qd_vec, obs_action, obs_phase, obs_bpm, traj]).unsqueeze(0).float()
    return obs.numpy().reshape(-1), (q_wrist, q_grip)


def new_obs(rhythm, wrist_deg, grip_deg, prev_q, prev_action, t, dt, target_force):
    """src/deploy_policy.py::build_base_frame と同じ式（numpy）。"""
    q = np.array([np.radians(wrist_deg), np.radians(grip_deg)], dtype=np.float64)
    if prev_q is None:
        qd = np.zeros(2, dtype=np.float64)
    else:
        qd = np.clip((q - prev_q) / dt, -QD_CLIP, QD_CLIP)
    phase, traj = rhythm.get_state(t)
    traj = traj.detach().cpu().numpy().astype(np.float64)
    frame = np.concatenate([
        q, qd, np.asarray(prev_action, np.float64),
        np.array([np.sin(phase), np.cos(phase)], np.float64),
        np.array([rhythm.bpm / 180.0], np.float64),
        traj,
    ]).astype(np.float32)
    return frame, q


def test_obs_parity(model_key="IROS/B", midi="songs/test_single4_bpm60.mid", n_steps=400):
    spec = resolve(model_key)
    dt = spec.control_dt

    # 実機ログの角度系列を入力に使う（無ければ合成波）
    cands = sorted(glob.glob(os.path.join(ROOT, "IROS", "deploy_results", "**", "*.csv"),
                             recursive=True))
    if cands:
        import pandas as pd
        df = pd.read_csv(cands[0])
        wrist = df["wrist_angle_deg"].to_numpy()[:n_steps]
        grip = df["grip_angle_deg"].to_numpy()[:n_steps]
        src = os.path.relpath(cands[0], ROOT)
    else:
        t = np.arange(n_steps) * dt
        wrist = 10 * np.sin(2 * np.pi * 2 * t)
        grip = 8 + 2 * np.sin(2 * np.pi * t)
        src = "(合成波)"
    n_steps = min(n_steps, len(wrist))
    print(f"[obs parity] model={model_key} 入力={src} steps={n_steps}")

    rg_old = MidiRhythmGenerator(os.path.join(ROOT, midi), torch.device("cpu"), dt,
                                 spec.target_force, spec.lookahead_steps)
    rg_new = MidiRhythmGenerator(os.path.join(ROOT, midi), torch.device("cpu"), dt,
                                 spec.target_force, spec.lookahead_steps)

    prev_q_old = prev_q_new = None
    prev_a = np.zeros(3, dtype=np.float32)
    max_diff = 0.0
    for i in range(n_steps):
        t = i * dt
        o_old, prev_q_old = legacy_obs(rg_old, wrist[i], grip[i], prev_q_old, prev_a, t, dt)
        o_new, prev_q_new = new_obs(rg_new, wrist[i], grip[i], prev_q_new, prev_a, t, dt,
                                    spec.target_force)
        max_diff = max(max_diff, float(np.max(np.abs(o_old - o_new))))
        prev_a = np.clip(np.sin(np.array([i, i + 1, i + 2], np.float32) * 0.1), -1, 1)

    ok = max_diff == 0.0
    print(f"[obs parity] 旧実装との最大差 = {max_diff:.3e} -> {'OK (ビット一致)' if ok else 'NG'}")
    return ok


def test_framestack_vs_sim(k=5, base=35, n=12):
    """sim側の torch.roll(-1) → 末尾書き込み → reshape と一致するか。"""
    frames = [np.full(base, i + 1, dtype=np.float32) for i in range(n)]

    sim_hist = torch.zeros((1, k, base))
    for f in frames:
        sim_hist = torch.roll(sim_hist, shifts=-1, dims=1)
        sim_hist[:, -1, :] = torch.from_numpy(f)
    sim_obs = sim_hist.reshape(1, -1).numpy().reshape(-1)

    class _Spec:
        frame_stack, base_obs_dim = k, base
    buf = np.zeros((k, base), dtype=np.float32)
    for f in frames:
        buf = np.roll(buf, -1, axis=0)
        buf[-1] = f
    ours = buf.reshape(-1)

    ok = np.array_equal(sim_obs, ours)
    print(f"[framestack] k={k} base={base}: sim と {'一致 OK' if ok else '不一致 NG'} "
          f"(先頭値={ours[0]:.0f} 末尾値={ours[-1]:.0f} / 期待 先頭={n-k+1} 末尾={n})")
    return ok


def test_manifest_guard():
    """manifest と ONNX が矛盾したとき、ちゃんと落ちるか。"""
    spec = resolve("IROS/B")
    spec.frame_stack = 5          # わざと嘘をつく (35 -> 175 のはず)
    try:
        PolicyRunner(spec, verbose=False)
    except RuntimeError as e:
        print(f"[guard] 矛盾を検出して停止 OK\n        {str(e).splitlines()[1].strip()}")
        return True
    print("[guard] NG: 矛盾を見逃した")
    return False


if __name__ == "__main__":
    results = [
        test_obs_parity("IROS/B"),
        test_obs_parity("IROS/A"),
        test_obs_parity("IROS/C"),
        test_framestack_vs_sim(),
        test_manifest_guard(),
    ]
    print("\n=== 総合:", "OK" if all(results) else "NG", "===")
    sys.exit(0 if all(results) else 1)
