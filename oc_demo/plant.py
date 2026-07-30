"""ロボット側I/O。実機(シリアル)とモックを同じインタフェースで扱う。

Plant:
    open()            接続
    write(p3)         圧力指令[MPa]を送る
    read() -> Frame   最新のセンサ値（非ブロッキング。無ければ直前値）
    close()           圧力0を送って切断

これで制御ループ本体は実機/モックで完全に同じコードになる。
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import adapter


@dataclass
class Frame:
    """MicroLabBox から返ってくる1フレーム。"""

    meas_p: np.ndarray = field(default_factory=lambda: np.zeros(3))  # MPa
    wrist_deg: float = 0.0
    grip_deg: float = 0.0
    flag: float = 0.0
    force_N: float = 0.0
    stamp: float = 0.0


class Plant:
    kind = "base"

    def open(self) -> None: ...
    def write(self, pressures: np.ndarray) -> None: ...
    def read(self) -> Frame: ...
    def close(self) -> None: ...

    @property
    def stats(self) -> dict:
        return {}


# ---------------------------------------------------------------------
# 実機
# ---------------------------------------------------------------------
class SerialPlant(Plant):
    """FTDI経由でMicroLabBoxと通信する。受信は200Hzの別スレッド。"""

    kind = "hardware"
    PACKET_LEN = 2 + 8 * adapter.RECV_FIELDS  # 58 byte

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = adapter.SERIAL_BAUD):
        self.port = port
        self.baud = baud
        self._ser = None
        self._latest = Frame()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._recv_count = 0
        self._bad_count = 0
        self._unpack = struct.Struct(adapter.ENDIAN + "d" * adapter.RECV_FIELDS)
        self._pack = struct.Struct(adapter.ENDIAN + "d" * adapter.SEND_FIELDS)

    def open(self) -> None:
        import serial  # pyserial。実機のみ必要

        self._ser = serial.Serial(self.port, self.baud, timeout=0.05)
        self.clear_buffer_for_sync()
        self._stop.clear()
        self._thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._thread.start()

    def clear_buffer_for_sync(self) -> None:
        if self._ser is not None:
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
        self._recv_count = 0
        self._bad_count = 0

    def _rx_loop(self) -> None:
        buf = bytearray()
        assert self._ser is not None
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(self.PACKET_LEN * 4)
            except Exception:  # noqa: BLE001
                break
            if chunk:
                buf.extend(chunk)
            # ヘッダ同期
            while True:
                idx = buf.find(adapter.SERIAL_HEADER)
                if idx < 0:
                    if len(buf) > 4096:
                        del buf[:-2]
                    break
                if len(buf) - idx < self.PACKET_LEN:
                    if idx:
                        del buf[:idx]
                    break
                payload = bytes(buf[idx + 2 : idx + self.PACKET_LEN])
                del buf[: idx + self.PACKET_LEN]
                try:
                    vals = self._unpack.unpack(payload)
                except struct.error:
                    self._bad_count += 1
                    continue
                fr = Frame(
                    meas_p=np.array(vals[0:3], dtype=np.float64),
                    wrist_deg=float(vals[3]),
                    grip_deg=float(vals[4]),
                    flag=float(vals[5]),
                    force_N=float(vals[6]),
                    stamp=time.time(),
                )
                with self._lock:
                    self._latest = fr
                    self._recv_count += 1

    def write(self, pressures: np.ndarray) -> None:
        if self._ser is None:
            return
        p = np.clip(np.asarray(pressures, dtype=np.float64), 0.0, adapter.P_MAX)
        self._ser.write(adapter.SERIAL_HEADER + self._pack.pack(*p))

    def read(self) -> Frame:
        with self._lock:
            return self._latest

    def close(self) -> None:
        try:
            self.write(np.zeros(3))
            time.sleep(0.02)
            self.write(np.zeros(3))
        except Exception:  # noqa: BLE001
            pass
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._ser is not None:
            try:
                self._ser.close()
            finally:
                self._ser = None

    @property
    def stats(self) -> dict:
        return {"recv": self._recv_count, "bad": self._bad_count, "port": self.port}


# ---------------------------------------------------------------------
# モック
# ---------------------------------------------------------------------
class MockPlant(Plant):
    """実機が無いときの簡易プラント。

    厳密な同定モデルではない。「圧力を入れると手首が振れて、振り下ろしきると
    力センサにスパイクが出る」程度の、画面確認に足りる挙動を作る。
    """

    kind = "mock"

    STRIKE_ANGLE = 18.0  # deg。ここを超えると打面に当たる
    LAG = 0.030  # 圧力の一次遅れ時定数[s]
    WN = 26.0  # 手首の固有角周波数[rad/s]
    ZETA = 0.32  # 減衰比
    SWING_GAIN = 62.0  # (p_F - p_DF)/P_MAX -> 目標角[deg]

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._p_cmd = np.zeros(3)
        self._p_meas = np.zeros(3)
        self._wrist = 0.0  # deg
        self._wrist_vel = 0.0  # deg/s
        self._grip = 12.0
        self._force = 0.0
        self._last_t = None
        self._count = 0
        self._contact = False

    def open(self) -> None:
        self._last_t = time.time()

    def write(self, pressures: np.ndarray) -> None:
        self._p_cmd = np.clip(np.asarray(pressures, dtype=np.float64), 0.0, adapter.P_MAX)

    SUBSTEPS = 4

    def read(self) -> Frame:
        """1回の呼び出しで制御1ステップ(20ms)ぶん進める。

        壁時計ではなく固定dtで積分する。OSのスケジューリング揺らぎで
        「シミュレーションなのにタイミングがブレる」のを避けるため。
        """
        dt_total = adapter.CONTROL_DT
        dt = dt_total / self.SUBSTEPS

        for _ in range(self.SUBSTEPS):
            # 圧力: 一次遅れ
            alpha = dt / (self.LAG + dt)
            self._p_meas += alpha * (self._p_cmd - self._p_meas)

            # 手首: 拮抗する2本の人工筋の差圧で決まる目標角へ、2次系で追従
            theta_cmd = self.SWING_GAIN * (self._p_meas[1] - self._p_meas[0]) / adapter.P_MAX
            acc = ((self.WN ** 2) * (theta_cmd - self._wrist)
                   - 2.0 * self.ZETA * self.WN * self._wrist_vel)
            self._wrist_vel += acc * dt
            self._wrist += self._wrist_vel * dt

            # 打面接触
            self._force *= float(np.exp(-dt / 0.040))
            if self._wrist >= self.STRIKE_ANGLE and self._wrist_vel > 0:
                if not self._contact:
                    self._force = max(self._force, 0.020 * abs(self._wrist_vel))
                    self._contact = True
                self._wrist = self.STRIKE_ANGLE
                self._wrist_vel = -0.30 * self._wrist_vel  # 反発
            elif self._wrist < self.STRIKE_ANGLE - 1.5:
                self._contact = False

            self._grip += (10.0 + 60.0 * self._p_meas[2] - self._grip) * min(dt / 0.08, 1.0)

        meas = self._p_meas + self._rng.normal(0.0, 0.002, 3)
        self._count += 1
        self._last_t = time.time()
        return Frame(
            meas_p=np.maximum(meas, 0.0),
            wrist_deg=float(self._wrist),
            grip_deg=float(self._grip),
            flag=1.0,
            force_N=float(self._force + self._rng.normal(0.0, 0.02)),
            stamp=self._last_t,
        )

    def close(self) -> None:
        self._p_cmd = np.zeros(3)

    @property
    def stats(self) -> dict:
        return {"recv": self._count, "bad": 0, "port": "mock"}
