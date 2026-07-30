"""src/midi_rhythm_generator.py (MidiRhythmGenerator) の目標力軌道生成を
torch / mido 抜きで再現したもの。

なぜ移植するか:
  OCデモ環境はネットワークが無く pip install できない。
  src/midi_rhythm_generator.py は torch と mido に依存しており、どちらかが
  無い環境では import 自体が失敗する（OCデモの動作環境を選ばないようにする
  ため、この依存を切り離す）。ここでは同じ数式を numpy + oc_demo/midiparse.py
  のみで再実装している。src/ 側は変更しない。

  数式は src/midi_rhythm_generator.py の
  MidiRhythmGenerator.__init__ / _load_and_process_midi と完全一致させること
  （oc_demo/parity_check.py が実物とのビット一致を検査する）。

簡略化の踏襲（実物と同じ）:
  テンポは「ファイル中で最初に見つかった値」を曲全体で一定として使う。
  途中のテンポ変化には対応しない（src側もしていない）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .midiparse import _collect_events

WIDTH_SEC = 0.035  # src/midi_rhythm_generator.py の Super-Gaussian カーネル幅と同じ


def _kernel(dt: float, target_force: float) -> Tuple[np.ndarray, int]:
    """MidiRhythmGenerator.__init__ のカーネル定義をそのまま再現。"""
    sigma = np.float32(WIDTH_SEC / 2.0)
    radius = int(WIDTH_SEC / dt)
    t_vals = np.arange(-radius, radius + 1, dtype=np.float32) * np.float32(dt)
    kernel = np.float32(target_force) * np.exp(-0.5 * (t_vals / sigma) ** 4)
    return kernel.astype(np.float32), radius


def _note_on_times(path: str | Path, override_bpm) -> Tuple[list, float, float]:
    """MidiRhythmGenerator._load_and_process_midi のタイミング計算部分の再現。

    戻り値: (note_on秒のリスト, 曲末尾の累積秒(2.0を足す前), 使用したbpm)
    """
    data = Path(path).read_bytes()
    events, division = _collect_events(data)

    tempo = 500_000
    for _tick, _order, kind, a, _b in events:
        if kind == "tempo":
            tempo = a
            break

    original_bpm = 60_000_000.0 / tempo
    bpm = override_bpm if override_bpm is not None else original_bpm
    time_scale = (original_bpm / bpm) if override_bpm else 1.0

    if division & 0x8000:
        raise ValueError("SMPTEタイムベースのMIDIは未対応です（src側もtpqn前提）")
    tpqn = division or 480
    sec_per_tick = (tempo / 1_000_000.0) / tpqn

    current_time = 0.0
    last_tick = 0
    spikes: list = []
    for tick, _order, kind, a, _b in events:
        current_time += (tick - last_tick) * sec_per_tick * time_scale
        last_tick = tick
        if kind == "on":
            spikes.append(current_time)

    if not spikes:
        raise ValueError("No note_on events found in MIDI file.")

    return spikes, current_time, bpm


def build_target_force_trajectory(
    path: str | Path,
    bpm,
    dt: float = 0.02,
    target_force: float = 20.0,
) -> np.ndarray:
    """目標力軌道[N]を返す（自然長 = MidiRhythmGenerator と同じ total_steps）。

    src/midi_rhythm_generator.py の trajectory_buffer と完全一致する設計。
    get_state() のような正規化(/target_force)はしない。生の[N]値を返す。
    """
    spikes, duration_end, _bpm_used = _note_on_times(path, override_bpm=bpm)
    total_duration = duration_end + 2.0
    total_steps = int(total_duration / dt)

    spike = np.zeros(total_steps, dtype=np.float32)
    for t in spikes:
        idx = int(t / dt)
        if idx < total_steps:
            spike[idx] = 1.0

    kernel, radius = _kernel(dt, target_force)
    # torch.conv1d(padding=radius) の「両側ゼロ埋め + valid畳み込み」を明示的に再現。
    # カーネルは対称なので、相互相関(torch)と畳み込み(numpy)の違いは影響しない。
    padded = np.zeros(total_steps + 2 * radius, dtype=np.float32)
    padded[radius: radius + total_steps] = spike
    traj = np.convolve(padded, kernel, mode="valid").astype(np.float32)
    assert traj.shape[0] == total_steps

    return traj.astype(np.float64)
