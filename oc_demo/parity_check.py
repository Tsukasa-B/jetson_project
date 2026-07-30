"""oc_demo/parity_check.py — oc_demo経路が src/deploy_policy.py 経路と
完全一致(差 0.0)することを確認する。tools/parity_test.py の oc_demo 版。

検証する2点:
  1. build_obs: 同一入力(q, qd, prev_action, t, bpm, lookahead)から
     src側(model_registry.PolicyRunner.build_obs, frame_stack=1)と
     oc_demo側(adapter.build_obs)が同じ配列を作るか。
  2. build_target_force: 同一MIDI+bpmから
     src側(MidiRhythmGenerator.trajectory_buffer)と
     oc_demo側(adapter.build_target_force / midi_rhythm_port)が
     同じ目標力軌道[N]を作るか。

★このスクリプト自体は「実物(src側)」をground truthとして使うため torch/mido
  が要る。oc_demo本体(adapter.py 等)はこれらに依存していない
  （見つからなくても oc_demo/midi_rhythm_port.py のポート版で動く）。

  python3 -m oc_demo.parity_check
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402  ground truth側でのみ使う

from midi_rhythm_generator import MidiRhythmGenerator  # noqa: E402
from model_registry import resolve  # noqa: E402

from . import adapter  # noqa: E402
from . import midi_rhythm_port as mrp  # noqa: E402
from .score import load_midi  # noqa: E402

QD_CLIP = adapter.QD_CLIP


# ---------------------------------------------------------------------
# 1. build_obs パリティ
# ---------------------------------------------------------------------
def _real_frame(rhythm, wrist_deg, grip_deg, prev_q, prev_action, t, dt):
    """src/deploy_policy.py::Deployer.build_base_frame + model_registry.build_obs
    (frame_stack=1 のときは1フレームがそのままobs) の再現。ground truth。"""
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


def _oc_demo_frame(rhythm, wrist_deg, grip_deg, prev_q, prev_action, t, dt):
    """oc_demo/runner.py が adapter.build_obs を呼ぶのと同じ手順。"""
    q_wrist = np.radians(wrist_deg)
    q_grip = np.radians(grip_deg)
    if prev_q is None:
        qd_wrist = qd_grip = 0.0
    else:
        qd_wrist = (q_wrist - prev_q[0]) / dt
        qd_grip = (q_grip - prev_q[1]) / dt
    phase, traj = rhythm.get_state(t)  # ground truthの正規化済みlookaheadをそのまま使う
    traj = traj.detach().cpu().numpy().astype(np.float64)
    frame = adapter.build_obs(q_wrist, q_grip, qd_wrist, qd_grip,
                              prev_action, t, rhythm.bpm, traj)
    return frame, np.array([q_wrist, q_grip], dtype=np.float64)


def check_build_obs(model_key: str, midi: str, n_steps: int = 400) -> bool:
    spec = resolve(model_key)
    dt = spec.control_dt

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
    print(f"[build_obs] model={model_key} 入力={src} steps={n_steps}")

    rg_real = MidiRhythmGenerator(os.path.join(ROOT, midi), torch.device("cpu"), dt,
                                  spec.target_force, spec.lookahead_steps)
    rg_oc = MidiRhythmGenerator(os.path.join(ROOT, midi), torch.device("cpu"), dt,
                                spec.target_force, spec.lookahead_steps)

    prev_q_real = prev_q_oc = None
    prev_a = np.zeros(3, dtype=np.float32)
    max_diff = 0.0
    for i in range(n_steps):
        t = i * dt
        o_real, prev_q_real = _real_frame(rg_real, wrist[i], grip[i], prev_q_real, prev_a, t, dt)
        o_oc, prev_q_oc = _oc_demo_frame(rg_oc, wrist[i], grip[i], prev_q_oc, prev_a, t, dt)
        max_diff = max(max_diff, float(np.max(np.abs(o_real - o_oc))))
        prev_a = np.clip(np.sin(np.array([i, i + 1, i + 2], np.float32) * 0.1), -1, 1)

    ok = max_diff == 0.0
    print(f"[build_obs] {model_key} 最大差 = {max_diff:.3e} -> {'OK (ビット一致)' if ok else 'NG'}")
    return ok


# ---------------------------------------------------------------------
# 2. build_target_force パリティ
# ---------------------------------------------------------------------
def check_target_force(midi: str, bpm) -> bool:
    dt = adapter.CONTROL_DT
    tf = adapter.TARGET_FORCE_N
    path = os.path.join(ROOT, midi)

    rg = MidiRhythmGenerator(path, torch.device("cpu"), dt, tf,
                             lookahead_steps=25, override_bpm=bpm)
    real = rg.trajectory_buffer.detach().cpu().numpy().astype(np.float64)

    port = mrp.build_target_force_trajectory(path, bpm, dt=dt, target_force=tf)

    if len(real) != len(port):
        print(f"[target_force] {midi} bpm={bpm}: 長さ不一致 real={len(real)} port={len(port)} -> NG")
        return False
    max_diff = float(np.max(np.abs(real - port)))
    ok = max_diff == 0.0
    print(f"[target_force] {midi} bpm={bpm}: 最大差 = {max_diff:.3e} "
          f"(len={len(real)}) -> {'OK (ビット一致)' if ok else 'NG'}")

    # adapter.build_target_force 経由(Score object + STATUS配線)でも確認
    score = load_midi(path)
    via_adapter = adapter.build_target_force(score, bpm or score.nominal_bpm, len(real), dt)
    max_diff2 = float(np.max(np.abs(real - via_adapter)))
    ok2 = max_diff2 == 0.0
    tag = adapter.STATUS.get("target_force", "?")
    print(f"[target_force] adapter.build_target_force 経由: 最大差 = {max_diff2:.3e} "
          f"STATUS={tag} -> {'OK' if ok2 else 'NG'}")
    return ok and ok2


# ---------------------------------------------------------------------
if __name__ == "__main__":
    results = [
        check_build_obs("IROS/B", "songs/test_single4_bpm60.mid"),
        check_build_obs("IROS/A", "songs/test_single4_bpm60.mid"),
        check_build_obs("IROS/C", "songs/test_single4_bpm60.mid"),
        check_target_force("songs/test_single4_bpm60.mid", None),
        check_target_force("songs/test_double_bpm120.mid", 150.0),
        check_target_force("songs/gmd_03_high_bpm138.mid", 90.0),
    ]
    print("\n=== 総合:", "OK" if all(results) else "NG", "===")
    sys.exit(0 if all(results) else 1)
