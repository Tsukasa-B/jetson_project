"""MIDI譜面 → 画面表示用データ + 目標力軌道。

役割は2つ:
  1. ブラウザのピアノロールに描くための「音符リスト」を作る
  2. 制御ループが使う「目標力軌道 target_force[t]」を作る

重要: 2 は既存の midi_rhythm_generator.py と *完全に一致* していなければ
ならない。ここでは既存モジュールがあればそれを使い、無い場合のみ
フォールバック実装を使う（adapter.py を参照）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .midiparse import parse_midi


CONTROL_DT = 0.02  # 50Hz。run_rl_deploy_midi.py と揃える


@dataclass
class Note:
    """1打点。"""

    time: float  # 曲頭からの秒
    pitch: int  # MIDIノート番号
    velocity: int  # 1..127
    duration: float  # 秒（表示用。打楽器なので短い）

    def to_dict(self) -> dict:
        return {
            "t": round(self.time, 4),
            "p": self.pitch,
            "v": self.velocity,
            "d": round(self.duration, 4),
        }


@dataclass
class Score:
    """1曲ぶんの譜面。"""

    name: str
    path: str
    notes: List[Note] = field(default_factory=list)
    nominal_bpm: float = 120.0
    duration: float = 0.0
    lanes: List[int] = field(default_factory=list)  # 使われているpitchの昇順

    # ---- 表示用 -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "nominal_bpm": round(self.nominal_bpm, 2),
            "duration": round(self.duration, 3),
            "lanes": self.lanes,
            "n_notes": len(self.notes),
            "notes": [n.to_dict() for n in self.notes],
        }

    # ---- BPM変更 ------------------------------------------------------
    def rescaled(self, bpm: float) -> "Score":
        """再生BPMを変えた譜面を返す。

        nominal_bpm=120 の曲を bpm=150 で鳴らす場合、時間軸を 120/150 倍に縮める。
        """
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        k = self.nominal_bpm / float(bpm)
        notes = [
            Note(time=n.time * k, pitch=n.pitch, velocity=n.velocity, duration=n.duration * k)
            for n in self.notes
        ]
        return Score(
            name=self.name,
            path=self.path,
            notes=notes,
            nominal_bpm=bpm,  # 再生後の実効BPM（obsの bpm/180 にはこちらを使う）
            duration=self.duration * k,
            lanes=list(self.lanes),
        )

    # ---- 目標力軌道 ---------------------------------------------------
    def target_force_trajectory(
        self,
        n_steps: Optional[int] = None,
        peak_force: float = 5.0,
        rise: float = 0.06,
        fall: float = 0.10,
        dt: float = CONTROL_DT,
    ) -> np.ndarray:
        """各制御ステップの目標力[N]の配列を返す（フォールバック実装）。

        各打点に raised-cosine の立ち上がり + 指数減衰を置く。
        ※ 実機では adapter.build_target_force() が既存実装を優先して呼ばれる。
        """
        if n_steps is None:
            n_steps = int(math.ceil((self.duration + fall * 3 + 1.0) / dt))
        traj = np.zeros(n_steps, dtype=np.float64)
        t = np.arange(n_steps) * dt
        for note in self.notes:
            amp = peak_force * (note.velocity / 127.0)
            rel = t - note.time
            # 立ち上がり: -rise <= rel < 0 で 0→1 の raised cosine
            up = (rel >= -rise) & (rel < 0.0)
            traj[up] = np.maximum(traj[up], amp * 0.5 * (1.0 - np.cos(np.pi * (rel[up] + rise) / rise)))
            # 減衰: rel >= 0
            dn = (rel >= 0.0) & (rel < fall * 4.0)
            traj[dn] = np.maximum(traj[dn], amp * np.exp(-rel[dn] / fall))
        return traj


# ---------------------------------------------------------------------
# MIDI読み込み
# ---------------------------------------------------------------------
def load_midi(path: str | Path, default_bpm: float = 120.0) -> Score:
    """MIDIファイルを Score に変換する。

    note_on(velocity>0) を打点として拾う。テンポメタイベントから nominal_bpm を求める。
    パースは標準ライブラリのみの midiparse.py で行う（追加インストール不要）。
    """
    path = Path(path)
    raw, nominal_bpm = parse_midi(path)
    if not nominal_bpm:
        nominal_bpm = default_bpm

    notes = [Note(time=t, pitch=p, velocity=v, duration=d) for (t, p, v, d) in raw]
    lanes = sorted({n.pitch for n in notes})
    duration = max((n.time + n.duration for n in notes), default=0.0)

    return Score(
        name=path.stem,
        path=str(path),
        notes=notes,
        nominal_bpm=float(nominal_bpm),
        duration=float(duration),
        lanes=lanes,
    )


def scan_midi_dir(directory: str | Path) -> List[Score]:
    """ディレクトリ内の .mid/.midi を全部読む。壊れたファイルは飛ばす。"""
    directory = Path(directory)
    out: List[Score] = []
    if not directory.is_dir():
        return out
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in (".mid", ".midi"):
            continue
        try:
            out.append(load_midi(p))
        except Exception as exc:  # noqa: BLE001
            print(f"[score] skip {p.name}: {exc}")
    return out
