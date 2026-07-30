"""制御ランナー。50Hzの制御ループを別スレッドで回し、テレメトリを溜める。

実機/モックの違いは Plant と Policy の差し替えだけで吸収する。
既存 run_rl_deploy_midi.py と同様に、start_time + step*0.02 の絶対時刻同期。
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import adapter
from .plant import Frame, MockPlant, Plant, SerialPlant
from .policy import OnnxPolicy, Policy, ScriptedPolicy
from .score import Score


@dataclass
class Sample:
    t: float
    cmd_p: np.ndarray
    meas_p: np.ndarray
    wrist_deg: float
    grip_deg: float
    force_N: float
    target_N: float

    def to_list(self) -> list:
        """WebSocketで送る用。桁を落として軽くする。"""
        return [
            round(self.t, 3),
            round(float(self.cmd_p[0]), 4), round(float(self.cmd_p[1]), 4), round(float(self.cmd_p[2]), 4),
            round(float(self.meas_p[0]), 4), round(float(self.meas_p[1]), 4), round(float(self.meas_p[2]), 4),
            round(self.wrist_deg, 2),
            round(self.grip_deg, 2),
            round(self.force_N, 3),
            round(self.target_N, 3),
        ]


@dataclass
class Hit:
    t: float  # 実際に叩いた時刻
    peak_N: float
    note_t: Optional[float]  # 対応する譜面上の時刻
    error_ms: Optional[float]

    def to_dict(self) -> dict:
        return {
            "t": round(self.t, 4),
            "peak": round(self.peak_N, 2),
            "note_t": None if self.note_t is None else round(self.note_t, 4),
            "err_ms": None if self.error_ms is None else round(self.error_ms, 1),
        }


@dataclass
class RunConfig:
    score: Score
    bpm: float
    model_path: Optional[str] = None
    port: str = "/dev/ttyUSB0"
    use_hardware: bool = True
    lead_in: float = 1.5  # カウントイン（曲頭前の待ち）秒
    tail: float = 1.0  # 曲尾の余韻
    drive: bool = True  # False なら圧力0を送り続ける（駆動なし検証モード）
    force_threshold_N: float = 0.8  # 打点判定のしきい値
    force_limit_N: float = 60.0  # これを超えたら安全停止
    zero_force_samples: int = 50  # 起動直後の力センサゼロ点推定に使うフレーム数


def build_run_target(score: Score, bpm: float, lead_in: float, tail: float
                     ) -> tuple[np.ndarray, int]:
    """演奏1回ぶんの目標力軌道（カウントイン込み）と総ステップ数を返す。

    Runner とブラウザ側プレビューで同じものを使うため、ここに切り出してある。
    """
    played = score.rescaled(bpm)
    total_time = lead_in + played.duration + tail
    n_steps = int(math.ceil(total_time / adapter.CONTROL_DT))
    pad_steps = int(lead_in / adapter.CONTROL_DT)
    body = adapter.build_target_force(played, bpm, n_steps - pad_steps)
    target = np.concatenate([np.zeros(pad_steps), body])[:n_steps]
    if target.size < n_steps:
        target = np.concatenate([target, np.zeros(n_steps - target.size)])
    return target, n_steps


class Runner:
    """1回の演奏を管理する。"""

    def __init__(self, cfg: RunConfig, telemetry_max: int = 20000):
        self.cfg = cfg
        self.state = "idle"  # idle|arming|running|finished|stopped|error
        self.message = ""
        self.samples: deque[Sample] = deque(maxlen=telemetry_max)
        self.hits: list[Hit] = []
        self.run_id = 0  # 接続側がカーソルをリセットする判定に使う
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.plant: Optional[Plant] = None
        self.policy: Optional[Policy] = None
        self.started_at: Optional[float] = None
        self.force_zero = 0.0
        self.info: dict = {}

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.state in ("running", "arming"):
            raise RuntimeError("already running")
        self._stop.clear()
        self.samples.clear()
        self.hits.clear()
        self.run_id += 1
        self.state = "arming"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    def _setup(self) -> tuple[Plant, Policy, np.ndarray, int]:
        cfg = self.cfg
        target, n_steps = build_run_target(cfg.score, cfg.bpm, cfg.lead_in, cfg.tail)

        # Plant
        plant: Plant
        if cfg.use_hardware:
            plant = SerialPlant(cfg.port)
        else:
            plant = MockPlant()
        plant.open()

        # Policy
        policy: Policy
        if cfg.model_path:
            policy = OnnxPolicy(cfg.model_path)
        else:
            policy = ScriptedPolicy(target)
        policy.reset()

        self.info = {
            "plant": plant.kind,
            "policy": policy.kind,
            "obs_dim": policy.obs_dim,
            "lookahead_steps": policy.lookahead_steps,
            "is_rnn": policy.is_rnn,
            "n_steps": n_steps,
            "bpm": cfg.bpm,
            "drive": cfg.drive,
            "adapter_status": dict(adapter.STATUS),
        }
        return plant, policy, target, n_steps

    # ------------------------------------------------------------------
    def _run(self) -> None:
        cfg = self.cfg
        plant = policy = None
        try:
            plant, policy, target, n_steps = self._setup()
            self.plant, self.policy = plant, policy

            played = cfg.score.rescaled(cfg.bpm)
            note_times = np.array([n.time + cfg.lead_in for n in played.notes])

            # --- 力センサのゼロ点を実測（無負荷前提） -----------------
            zs = []
            for _ in range(cfg.zero_force_samples):
                zs.append(plant.read().force_N)
                time.sleep(0.005)
            self.force_zero = float(np.median(zs)) if zs else 0.0
            self.info["force_zero"] = round(self.force_zero, 3)

            if isinstance(plant, SerialPlant):
                plant.clear_buffer_for_sync()

            L = max(policy.lookahead_steps, 0)
            prev_action = np.zeros(3)
            prev_q = None
            in_hit = False
            hit_peak = 0.0
            hit_start = 0.0

            self.state = "running"
            start = time.time()
            self.started_at = start

            for step in range(n_steps):
                if self._stop.is_set():
                    self.state = "stopped"
                    break

                # 絶対時刻同期
                target_wall = start + step * adapter.CONTROL_DT
                sleep = target_wall - time.time()
                if sleep > 0:
                    time.sleep(sleep)
                t = step * adapter.CONTROL_DT

                fr: Frame = plant.read()
                force = fr.force_N - self.force_zero

                # 安全: 想定外の力が出たら止める
                if abs(force) > cfg.force_limit_N:
                    self.state = "error"
                    self.message = f"力センサが {force:.1f} N に達したため安全停止しました"
                    break

                q_wrist = math.radians(fr.wrist_deg)
                q_grip = math.radians(fr.grip_deg)
                if prev_q is None:
                    qd_wrist = qd_grip = 0.0
                else:
                    qd_wrist = (q_wrist - prev_q[0]) / adapter.CONTROL_DT
                    qd_grip = (q_grip - prev_q[1]) / adapter.CONTROL_DT
                prev_q = (q_wrist, q_grip)

                la = target[step : step + L]
                if la.size < L:
                    la = np.concatenate([la, np.zeros(L - la.size)])
                # src/deploy_policy.py (MidiRhythmGenerator.get_state) は曲ごとの
                # ピークではなく固定の target_force で正規化している。ここを
                # target.max() で割っていたのは obs のスケールが実物とズレるバグ
                # だった（曲によってピークが変わるため）。
                la = la / adapter.TARGET_FORCE_N

                obs = adapter.build_obs(q_wrist, q_grip, qd_wrist, qd_grip,
                                        prev_action, t, cfg.bpm, la)
                action = policy.act(obs, step)
                action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:3], -1.0, 1.0)
                prev_action = action

                pressures = adapter.action_to_pressure(action)
                # 駆動なし検証モードでは「本来出したい圧力」をログには残しつつ、
                # 実際には0を送る。ロボットは動かないが obs/推論/画面は本番と同じ経路を通る。
                plant.write(pressures if cfg.drive else np.zeros(3))

                # 打点検出（力の立ち上がり→ピーク→戻り）
                if not in_hit and force > cfg.force_threshold_N:
                    in_hit, hit_peak, hit_start = True, force, t
                elif in_hit:
                    if force > hit_peak:
                        hit_peak, hit_start = force, t
                    if force < cfg.force_threshold_N * 0.5:
                        in_hit = False
                        nt, err = self._match_note(hit_start, note_times)
                        with self._lock:
                            self.hits.append(Hit(hit_start, hit_peak, nt, err))

                with self._lock:
                    self.samples.append(
                        Sample(t, pressures, fr.meas_p, fr.wrist_deg, fr.grip_deg,
                               float(force), float(target[step]))
                    )
            else:
                self.state = "finished"

        except Exception as exc:  # noqa: BLE001
            self.state = "error"
            self.message = f"{exc.__class__.__name__}: {exc}"
        finally:
            try:
                if plant is not None:
                    plant.close()
            except Exception:  # noqa: BLE001
                pass
            if self.state == "running":
                self.state = "stopped"

    @staticmethod
    def _match_note(t: float, note_times: np.ndarray) -> tuple[Optional[float], Optional[float]]:
        if note_times.size == 0:
            return None, None
        i = int(np.argmin(np.abs(note_times - t)))
        nt = float(note_times[i])
        err = (t - nt) * 1000.0
        if abs(err) > 250.0:  # 明らかに対応しない打点
            return None, None
        return nt, err

    # ------------------------------------------------------------------
    def drain(self, cursor: int = 0) -> tuple[list, list, int]:
        """cursor 以降のサンプルと、現時点のヒット一覧、次のcursorを返す。

        cursor は呼び出し側（WebSocket接続ごと）が持つ。Runner側で持つと
        ブラウザを2枚開いたときにサンプルが取り合いになって歯抜けになる。
        """
        with self._lock:
            all_samples = list(self.samples)
            hits = [h.to_dict() for h in self.hits]
        cursor = max(0, min(cursor, len(all_samples)))
        new = all_samples[cursor:]
        return [s.to_list() for s in new], hits, len(all_samples)

    def snapshot(self) -> dict:
        with self._lock:
            n_hits = len(self.hits)
            errs = [h.error_ms for h in self.hits if h.error_ms is not None]
        good = sum(1 for e in errs if abs(e) <= adapter.HIT_WINDOW_MS)
        return {
            "state": self.state,
            "message": self.message,
            "song": self.cfg.score.name,
            "bpm": self.cfg.bpm,
            "lead_in": self.cfg.lead_in,
            "elapsed": None if self.started_at is None else round(time.time() - self.started_at, 3),
            "info": self.info,
            "n_hits": n_hits,
            "good": good,
            "mean_abs_err_ms": round(float(np.mean(np.abs(errs))), 1) if errs else None,
            "plant_stats": self.plant.stats if self.plant else {},
        }
