"""OC(オープンキャンパス)デモ用のMIDIを生成する。

mido は使えない環境(ネットワーク無し・pip不可)なので、標準ライブラリだけで
最小限の標準MIDIファイル(SMF format 0)を書き出す自前ライタを使う。

片腕ロボット前提の制約:
  - 同時打ちなし(常に1音ずつ順番に鳴らす)
  - 単一ピッチのみ(打面は1つしかない)

  python3 -m oc_demo.tools.make_demo_midi
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import List, Tuple

TICKS_PER_BEAT = 480
PITCH = 38  # スネア相当(単一ピッチ)
VELOCITY = 100
NOTE_LEN_SEC = 0.05  # 表示上の音符長。ロボット側の動作(打点時刻)には影響しない

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "midi"


# =============================================================================
# 自前SMFライタ(標準ライブラリのみ)
# =============================================================================
def _vlq(n: int) -> bytes:
    """可変長数値エンコード(MIDI標準。1バイト7bit、最上位bitが継続フラグ)。"""
    bs = [n & 0x7F]
    n >>= 7
    while n:
        bs.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(bs))


def write_midi(path: str | Path, onsets_sec: List[float], bpm: float,
              pitch: int = PITCH, velocity: int = VELOCITY,
              note_len_sec: float = NOTE_LEN_SEC) -> None:
    """onsets_sec(曲頭からの秒。順不同で可)から SMF format 0 を書き出す。

    同時打ちは想定していない(片腕ロボット制約)。note_off は次のnote_onより
    前に来るよう note_len_sec を十分短く取ること。
    """
    onsets_sec = sorted(onsets_sec)
    tempo_us = round(60_000_000 / bpm)
    sec_per_tick = (tempo_us / 1_000_000.0) / TICKS_PER_BEAT

    events: List[Tuple[int, int, int]] = []  # (tick, order, kind) kind:0=off が先
    for t in onsets_sec:
        on_tick = round(t / sec_per_tick)
        off_tick = round((t + note_len_sec) / sec_per_tick)
        events.append((on_tick, 1, 1))
        events.append((off_tick, 0, 0))
    events.sort()

    track = bytearray()
    track += _vlq(0) + b"\xff\x51\x03" + tempo_us.to_bytes(3, "big")  # set_tempo

    last_tick = 0
    for tick, _order, kind in events:
        delta = max(tick - last_tick, 0)
        last_tick = tick
        track += _vlq(delta)
        if kind == 1:
            track += bytes([0x90, pitch, velocity])
        else:
            track += bytes([0x80, pitch, 0])
    track += _vlq(0) + b"\xff\x2f\x00"  # end of track

    header = b"MThd" + (6).to_bytes(4, "big") + struct.pack(">HHH", 0, 1, TICKS_PER_BEAT)
    chunk = b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)
    Path(path).write_bytes(header + chunk)


def min_gap_ms(onsets_sec: List[float]) -> float:
    s = sorted(onsets_sec)
    gaps = [b - a for a, b in zip(s, s[1:])]
    return min(gaps) * 1000.0 if gaps else float("nan")


# =============================================================================
# パターン定義
#
# 全パターン: 単一ピッチ・同時打ちなし(片腕ロボット前提)。
# 四分/八分は「現行版は打点過多」との指摘を受け、元のBPMは変えずに
# 1拍おき(=密度半分)にした。三連符・シンコペーション・ダブルストロークは
# 新規。BPMの根拠は各コメント参照。
# =============================================================================
def pattern_quarter(duration_target: float = 20.0) -> Tuple[List[float], float]:
    """四分打ち(一番やさしい)。

    旧 01_yonuchi は BPM100 で毎拍(4分音符ごと)ヒットしており密度過多だった
    指摘に対応し、BPMはそのままに「1拍おき」にして密度を半分にする。
    """
    bpm = 100.0
    beat = 60.0 / bpm
    step = beat * 2  # 1拍おき = 密度半分
    n = int(duration_target / step) + 1
    onsets = [i * step for i in range(n)]
    return onsets, bpm


def pattern_eighth(duration_target: float = 20.0) -> Tuple[List[float], float]:
    """八分打ち。

    旧 02_eighth は BPM110 で8分音符を隙間なく敷き詰めており密度過多だった
    指摘に対応し、BPMはそのままに「1拍ごとに8分2連→次の1拍は休み」にして
    密度を半分にする(8分音符のツブ立ちは残しつつ総打点数は半分)。
    """
    bpm = 110.0
    beat = 60.0 / bpm
    eighth = beat / 2
    cycle = beat * 2  # 2拍で1セット(1拍鳴らす+1拍休み)
    n_cycles = int(duration_target / cycle) + 1
    onsets = []
    for c in range(n_cycles):
        base = c * cycle
        onsets += [base, base + eighth]
    return onsets, bpm


def pattern_triplet(duration_target: float = 20.0) -> Tuple[List[float], float]:
    """三連符(むずかしい)。

    旧 04_triplet と同じ BPM96 を踏襲。1拍を3連符で埋め、次の1拍は休みにして
    (四分/八分と同様の"1拍休み"設計)、リズムの粒立ちを見せつつ総打点数を抑える。
    """
    bpm = 96.0
    beat = 60.0 / bpm
    trip = beat / 3
    cycle = beat * 2  # 1拍3連 + 1拍休み
    n_cycles = int(duration_target / cycle) + 1
    onsets = []
    for c in range(n_cycles):
        base = c * cycle
        onsets += [base, base + trip, base + 2 * trip]
    return onsets, bpm


def pattern_syncopation(duration_target: float = 20.0) -> Tuple[List[float], float]:
    """シンコペーション。

    定番の「トレシロ(3+3+2)」パターン。8分音符8個(=4拍)を 3+3+2 に区切って
    打つ、ラテン系リズムの代表形。単一ピッチでも「食っている」感が
    はっきり分かるためデモ向き。BPMは旧 05_syncopa の108を踏襲。
    ★このパターンの難しさは「速さ」ではなく「拍から見た不規則さ」であり、
      最小打点間隔(下記レポート参照)は四分打ち以外の中で最も長い
      (=物理的には一番ゆっくり)。ロボットにとっての挑戦点は
      「等間隔でない先読みに追従できるか」である。
    """
    bpm = 108.0
    beat = 60.0 / bpm
    eighth = beat / 2
    cycle = eighth * 8  # 4拍 = 8分音符8個ぶん
    n_cycles = int(duration_target / cycle) + 1
    onsets = []
    for c in range(n_cycles):
        base = c * cycle
        onsets += [base, base + 3 * eighth, base + 6 * eighth]  # 3+3+2
    return onsets, bpm


def pattern_double_stroke(duration_target: float = 20.0) -> Tuple[List[float], float]:
    """ダブルストローク(ロボットの限界を見せる用)。

    2打を100msで連続させ、800ms周期で繰り返す。100msは実機ログ
    (IROS/deploy_results/modelB, 200Hz)で実際に安定して打てていた
    最短間隔 81〜84ms(BPM138〜170の8分音符)より少し余裕を持たせつつ、
    oc_demo/runner.py の打点不応期 hit_refractory_s=0.05s(50ms)は
    確実に上回る値として選んだ(不応期未満だと1打として数えられてしまい、
    デモの見せ場である「2連打」がUI上1回にしか見えなくなる)。
    表示用の nominal_bpm は周期(0.8s)から逆算した 75 とする
    (拍として規則的な曲ではないため参考値)。
    """
    pair_gap = 0.100
    cycle = 0.800
    bpm = 60.0 / cycle  # = 75 (表示用)
    n_cycles = int(duration_target / cycle) + 1
    onsets = []
    for c in range(n_cycles):
        base = c * cycle
        onsets += [base, base + pair_gap]
    return onsets, bpm


PATTERNS = {
    "01_quarter": ("四分打ち（かんたん）",
                  "まず基本の四分打ち。ロボットがリズムに合わせて安定して叩けるか見てみよう。",
                  pattern_quarter),
    "02_eighth": ("八分打ち",
                 "四分打ちより速い八分音符のかたまりが1拍おきにやってくる。",
                 pattern_eighth),
    "03_triplet": ("三連符（むずかしい）",
                  "1拍を3等分する三連符。等間隔で叩けるかがポイント。",
                  pattern_triplet),
    "04_syncopation": ("シンコペーション",
                      "拍の裏を突く「食った」リズム。人でも難しい、不規則な間隔への追従を見せる。",
                      pattern_syncopation),
    "05_double_stroke": ("ダブルストローク（ロボットの限界！）",
                        "0.1秒間隔での2連打に挑戦。空気圧アクチュエータの応答速度の限界に近い。",
                        pattern_double_stroke),
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = {}
    rows = []
    for name, (label, note, fn) in PATTERNS.items():
        onsets, bpm = fn()
        dur = max(onsets) if onsets else 0.0
        path = OUT_DIR / f"{name}.mid"
        write_midi(path, onsets, bpm)
        labels[name] = {"label": label, "note": note}
        rows.append((name, bpm, len(onsets), dur, min_gap_ms(onsets)))

    import json
    (OUT_DIR / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'name':<18}{'bpm':>7}{'n_notes':>9}{'dur[s]':>9}{'min_gap[ms]':>13}")
    for name, bpm, n, dur, mg in rows:
        print(f"{name:<18}{bpm:7.1f}{n:9d}{dur:9.2f}{mg:13.1f}")


if __name__ == "__main__":
    main()
