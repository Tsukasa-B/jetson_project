"""標準MIDIファイル(SMF)の最小パーサ。

mido を入れられない環境（Jetsonのコンテナはpipの索引が閉じている）向けに、
Python標準ライブラリだけで note_on/note_off とテンポだけを取り出す。

対応: format 0/1、ランニングステータス、テンポ変化、SMPTEタイムベース。
非対応: SysEx の中身、format 2（トラックが順に並ぶ形式。まず使われない）。
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import List, Tuple

DEFAULT_TEMPO_US = 500_000  # 120 BPM


def _read_vlq(data: bytes, p: int) -> Tuple[int, int]:
    """可変長数値（1バイト7bit、最上位bitが継続フラグ）を読む。"""
    value = 0
    for _ in range(4):
        b = data[p]
        p += 1
        value = (value << 7) | (b & 0x7F)
        if not (b & 0x80):
            return value, p
    return value, p


def _collect_events(data: bytes) -> Tuple[list, int]:
    """全トラックのイベントを絶対tickで集めて返す。"""
    if data[0:4] != b"MThd":
        raise ValueError("MThd がありません（標準MIDIファイルではないようです）")
    hlen = int.from_bytes(data[4:8], "big")
    fmt, ntrks, division = struct.unpack(">HHH", data[8:14])
    if fmt not in (0, 1):
        raise ValueError(f"未対応のMIDI format: {fmt}")

    events: list = []  # (tick, order, kind, a, b)   order: tempo=0, note=1
    pos = 8 + hlen
    for _ in range(ntrks):
        if data[pos : pos + 4] != b"MTrk":
            break
        tlen = int.from_bytes(data[pos + 4 : pos + 8], "big")
        p, end = pos + 8, pos + 8 + tlen
        tick = 0
        running = None
        while p < end:
            delta, p = _read_vlq(data, p)
            tick += delta
            if p >= end:
                break
            b0 = data[p]

            if b0 == 0xFF:  # メタイベント
                p += 1
                mtype = data[p]
                p += 1
                length, p = _read_vlq(data, p)
                payload = data[p : p + length]
                p += length
                if mtype == 0x51 and length == 3:
                    events.append((tick, 0, "tempo", int.from_bytes(payload, "big"), 0))
                elif mtype == 0x2F:  # End of Track
                    break

            elif b0 in (0xF0, 0xF7):  # SysEx。中身は読み飛ばす
                p += 1
                length, p = _read_vlq(data, p)
                p += length

            else:  # MIDIチャンネルメッセージ
                if b0 & 0x80:
                    status = b0
                    running = status
                    p += 1
                else:
                    status = running
                    if status is None:
                        raise ValueError("ランニングステータスの解釈に失敗しました")
                hi = status & 0xF0
                if hi in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                    d1, d2 = data[p], data[p + 1]
                    p += 2
                    if hi == 0x90 and d2 > 0:
                        events.append((tick, 1, "on", d1, d2))
                    elif hi == 0x80 or (hi == 0x90 and d2 == 0):
                        events.append((tick, 1, "off", d1, 0))
                elif hi in (0xC0, 0xD0):
                    p += 1
                else:
                    p += 1  # 未知。壊れないよう1バイトだけ進める
        pos = end

    events.sort(key=lambda e: (e[0], e[1]))
    return events, division


def parse_midi(path: str | Path) -> Tuple[List[Tuple[float, int, int, float]], float]:
    """(notes, nominal_bpm) を返す。

    notes は (開始秒, pitch, velocity, 長さ秒) のリスト（開始秒でソート済み）。
    """
    data = Path(path).read_bytes()
    events, division = _collect_events(data)

    # タイムベース
    if division & 0x8000:  # SMPTE
        frames = 256 - (division >> 8)  # 上位バイトは負数(-24,-25,-29,-30)
        ticks_per_frame = division & 0xFF
        sec_per_tick_fixed = 1.0 / (frames * ticks_per_frame)
        tpqn = None
    else:
        tpqn = division or 480
        sec_per_tick_fixed = None

    tempo_us = DEFAULT_TEMPO_US
    first_tempo = None
    sec = 0.0
    last_tick = 0

    open_notes: dict = {}
    notes: List[Tuple[float, int, int, float]] = []

    for tick, _order, kind, a, b in events:
        dticks = tick - last_tick
        if sec_per_tick_fixed is not None:
            sec += dticks * sec_per_tick_fixed
        else:
            sec += dticks * (tempo_us / 1_000_000.0) / tpqn
        last_tick = tick

        if kind == "tempo":
            tempo_us = a or DEFAULT_TEMPO_US
            if first_tempo is None:
                first_tempo = tempo_us
        elif kind == "on":
            open_notes.setdefault(a, []).append((sec, b))
        elif kind == "off":
            stack = open_notes.get(a)
            if stack:
                start, vel = stack.pop(0)
                notes.append((start, a, vel, max(sec - start, 0.05)))

    # note_off が来なかったもの
    for pitch, stack in open_notes.items():
        for start, vel in stack:
            notes.append((start, pitch, vel, 0.1))

    notes.sort(key=lambda n: (n[0], n[1]))
    bpm = 60_000_000.0 / (first_tempo or DEFAULT_TEMPO_US)
    return notes, bpm
