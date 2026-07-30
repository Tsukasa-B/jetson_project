"""既存 jetson_project コードとの結線点。

★ Jetson側で最初に確認すべきファイルはここ。
ここ以外は jetson_project の中身に依存しないように書いてある。

方針:
  - 既存モジュール（midi_rhythm_generator など）が import できればそれを使う
  - できなければ oc_demo 内のフォールバック実装を使い、起動ログに警告を出す

「静かに間違う」のを避けるため、どちらを使っているかは必ず /api/health に出す。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from . import midi_rhythm_port as _mrp

# jetson_project のルート（oc_demo/ の1つ上）と src/ を import path に足す。
# リファクタで実装コードが src/ 配下に移動したため、src/ も足さないと
# 「既存モジュールが import できればそれを使う」の判定自体が常に失敗する。
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
for _p in (_ROOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------
# 1. 定数（run_rl_deploy_midi.py / models/manifest.yaml の defaults と一致させること）
# ---------------------------------------------------------------------
CONTROL_DT = 0.02  # 50Hz
P_MAX = 0.6  # MPa。action -> pressure の上限
TARGET_FORCE_N = 20.0  # models/manifest.yaml defaults.target_force と同じ。lookaheadの正規化に使う
SERIAL_BAUD = 230400
SERIAL_HEADER = b"\xff\xff"
RECV_FIELDS = 7  # [meas_p_DF, meas_p_F, meas_p_G, wrist_deg, grip_deg, flag, force_N]
SEND_FIELDS = 3  # [DF, F, G]
ENDIAN = ">"  # Big Endian
QD_CLIP = 20.0  # rad/s
HIT_WINDOW_MS = 30.0  # ±30ms 判定

# ---------------------------------------------------------------------
# 2. 既存モジュールの取り込み状況
# ---------------------------------------------------------------------
STATUS: dict[str, str] = {}

# src/midi_rhythm_generator.py は torch + mido に依存しているため、pipが使えない
# OCデモ環境では import できるとは限らない（できてもできなくても、実際の計算は
# 常に下の oc_demo/midi_rhythm_port.py の torch/mido 不要ポートを使う。
# ここでの import は「実物が今この環境で読めるか」の診断表示のためだけ）。
try:
    import midi_rhythm_generator as _mrg_probe  # type: ignore  # noqa: F401

    STATUS["midi_rhythm_generator"] = (
        "src/ を診断importできました（この環境には torch/mido があります）。"
        "ただし実際の軌道生成は oc_demo/midi_rhythm_port.py のポート版を使用します"
    )
    del _mrg_probe
except Exception as _exc:  # noqa: BLE001
    STATUS["midi_rhythm_generator"] = (
        f"src/ を診断importできませんでした ({_exc.__class__.__name__})。"
        "oc_demo/midi_rhythm_port.py の torch/mido 不要ポートを使用します（動作に問題なし）"
    )


def build_target_force(score, bpm: float, n_steps: int, dt: float = CONTROL_DT) -> np.ndarray:
    """目標力軌道[N]を返す。

    src/midi_rhythm_generator.py::MidiRhythmGenerator と同じ数式を
    oc_demo/midi_rhythm_port.py（numpy + oc_demo/midiparse.py。torch/mido不要）
    で計算する。oc_demo/parity_check.py が実物とのビット一致を検証している。
    """
    try:
        traj = _mrp.build_target_force_trajectory(
            score.path, bpm, dt=dt, target_force=TARGET_FORCE_N,
        )
        STATUS["target_force"] = f"real:{_mrp.build_target_force_trajectory.__name__}"
        return _fit_len(traj, n_steps)
    except Exception as exc:  # noqa: BLE001
        STATUS["target_force"] = (
            f"fallback:score.target_force_trajectory ({exc.__class__.__name__}: {exc})"
        )
        return _fit_len(score.target_force_trajectory(n_steps=n_steps, dt=dt), n_steps)


def _fit_len(a: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    if a.size == n:
        return a
    if a.size > n:
        return a[:n]
    return np.concatenate([a, np.zeros(n - a.size)])


# ---------------------------------------------------------------------
# 3. 観測ベクトルの構築
# ---------------------------------------------------------------------
def build_obs(
    q_wrist: float,
    q_grip: float,
    qd_wrist: float,
    qd_grip: float,
    prev_action: np.ndarray,
    t: float,
    bpm: float,
    lookahead: np.ndarray,
) -> np.ndarray:
    """obs = [q(2), qd(2), prev_action(3), sin, cos, bpm/180, lookahead(L)]

    jetson_deploy_survey_20260730.md §3 の仕様どおり。
    lookahead は target_force で正規化済み（0〜1）の配列を渡すこと。
    """
    phase = t * (bpm / 60.0) * 2.0 * np.pi
    head = np.array(
        [
            q_wrist,
            q_grip,
            float(np.clip(qd_wrist, -QD_CLIP, QD_CLIP)),
            float(np.clip(qd_grip, -QD_CLIP, QD_CLIP)),
            prev_action[0],
            prev_action[1],
            prev_action[2],
            np.sin(phase),
            np.cos(phase),
            bpm / 180.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([head, np.asarray(lookahead, dtype=np.float32)])


def action_to_pressure(action: np.ndarray) -> np.ndarray:
    """action(-1..1) -> 圧力[MPa]。run_rl_deploy_midi.py と同じ式。"""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return (a + 1.0) / 2.0 * P_MAX


# ---------------------------------------------------------------------
# 4. モデル一覧（models/manifest.yaml があれば読む）
# ---------------------------------------------------------------------
def load_manifest(models_dir: Optional[Path] = None) -> dict:
    """models/manifest.yaml を読む。無ければ空dict。"""
    models_dir = Path(models_dir) if models_dir else _ROOT / "models"
    for candidate in (models_dir / "manifest.yaml", models_dir / "manifest.yml"):
        if candidate.exists():
            try:
                import yaml  # type: ignore

                with open(candidate, "r", encoding="utf-8") as f:
                    STATUS["manifest"] = str(candidate)
                    return yaml.safe_load(f) or {}
            except Exception as exc:  # noqa: BLE001
                STATUS["manifest"] = f"error: {exc}"
                return {}
    STATUS["manifest"] = "not found"
    return {}
