"""
microlabbox.py — MicroLabBox とのシリアル通信

プロトコル（IROS時から不変。dSPACE側 Simulink モデルと対）
  送信 Jetson -> MLB  : b'\\xff\\xff' + '>ddd'      = [cmd_DF, cmd_F, cmd_G] (MPa)   50Hz
  受信 MLB -> Jetson  : b'\\xff\\xff' + '>ddddddd'  = [P_DF, P_F, P_G,
                                                      wrist_deg, grip_deg,
                                                      flag, force_N]              200Hz
  Big Endian / baud 230400
"""

from __future__ import annotations

import struct
import threading
import time

HEADER = b"\xff\xff"
SEND_FMT = ">ddd"
RECV_FMT = ">ddddddd"
RECV_PACKET_LEN = len(HEADER) + struct.calcsize(RECV_FMT)   # 2 + 56 = 58
SENSOR_RATE_HZ = 200.0


class SensorReceiver(threading.Thread):
    """200Hzで届くパケットを取りこぼさず受信し、制御ループには最新1フレームを渡す。"""

    def __init__(self, ser):
        super().__init__(daemon=True)
        self.ser = ser
        self.running = True
        self.latest = None
        self.logs: list[dict] = []
        self.n_packets = 0
        self.n_dropped = 0          # ヘッダ同期を失って捨てたバイト数
        self._clear_flag = False
        self._lock = threading.Lock()

    def run(self):
        self.ser.reset_input_buffer()
        buf = b""
        while self.running:
            try:
                n = self.ser.in_waiting
                if n > 0:
                    buf += self.ser.read(n)
                    while len(buf) >= RECV_PACKET_LEN:
                        idx = buf.find(HEADER)
                        if idx == -1:
                            self.n_dropped += max(0, len(buf) - 1)
                            buf = buf[-1:]
                            break
                        if idx > 0:
                            self.n_dropped += idx
                            buf = buf[idx:]
                        if len(buf) < RECV_PACKET_LEN:
                            break
                        packet, buf = buf[:RECV_PACKET_LEN], buf[RECV_PACKET_LEN:]
                        self._on_packet(packet[len(HEADER):])
                else:
                    time.sleep(0.001)
            except Exception as e:  # noqa: BLE001
                print(f"[Rx Error] {e}")
                self.running = False

    def _on_packet(self, payload: bytes):
        try:
            d = struct.unpack(RECV_FMT, payload)
        except struct.error:
            return
        sample = {
            "meas_pres_DF": d[0],
            "meas_pres_F": d[1],
            "meas_pres_G": d[2],
            "wrist_angle_deg": d[3],
            "grip_angle_deg": d[4],
            "flag": d[5],
            "force_N": d[6],
            "t_recv": time.perf_counter(),   # ジッタ確認用の実測受信時刻
        }
        with self._lock:
            if self._clear_flag:
                self.logs = []
                self.n_packets = 0
                self.n_dropped = 0
                self._clear_flag = False
            self.logs.append(sample)
            self.latest = sample
            self.n_packets += 1

    def clear_for_sync(self):
        """t=0 を揃えるため、USBバッファとログを一掃する。"""
        self.ser.reset_input_buffer()
        with self._lock:
            self._clear_flag = True

    def get_latest(self):
        with self._lock:
            return None if self.latest is None else dict(self.latest)

    def get_logs(self):
        with self._lock:
            return list(self.logs)

    def stop(self):
        self.running = False


def send_pressure(ser, df: float, f: float, g: float) -> None:
    ser.write(HEADER + struct.pack(SEND_FMT, float(df), float(f), float(g)))


def open_serial(port: str, baud: int, timeout: float = 0.01):
    import serial
    return serial.Serial(port, baud, timeout=timeout)


# =============================================================================
# ドライラン用の疑似MicroLabBox（実機なしで制御ループ全体を通すため）
# =============================================================================
class MockLink:
    """serial.Serial と同じインタフェースを持つ疑似デバイス。
    200Hzでもっともらしいセンサ値を生成し、送られた圧力指令をそのまま
    「計測圧力」として返す。タイミング・パケット整合・obs構築の確認用であり、
    物理的な妥当性は無い。"""

    def __init__(self, rate_hz: float = SENSOR_RATE_HZ):
        import math
        self._math = math
        self.dt = 1.0 / rate_hz
        self.t0 = time.perf_counter()
        self.n_emitted = 0
        self._pending = b""
        self._last_cmd = (0.0, 0.0, 0.0)
        self.sent_packets = 0
        self.is_open = True

    def _generate(self):
        now = time.perf_counter() - self.t0
        should = int(now / self.dt)
        m = self._math
        while self.n_emitted < should:
            t = self.n_emitted * self.dt
            wrist = 10.0 * m.sin(2 * m.pi * 2.0 * t)
            grip = 8.0 + 2.0 * m.sin(2 * m.pi * 1.0 * t)
            force = max(0.0, 15.0 * m.sin(2 * m.pi * 2.0 * t) ** 8)
            self._pending += HEADER + struct.pack(
                RECV_FMT, self._last_cmd[0], self._last_cmd[1], self._last_cmd[2],
                wrist, grip, 1.0, force)
            self.n_emitted += 1

    @property
    def in_waiting(self) -> int:
        self._generate()
        return len(self._pending)

    def read(self, n: int) -> bytes:
        self._generate()
        out, self._pending = self._pending[:n], self._pending[n:]
        return out

    def write(self, data: bytes) -> int:
        if data.startswith(HEADER) and len(data) == len(HEADER) + struct.calcsize(SEND_FMT):
            self._last_cmd = struct.unpack(SEND_FMT, data[len(HEADER):])
            self.sent_packets += 1
        return len(data)

    def reset_input_buffer(self):
        self._pending = b""
        self.t0 = time.perf_counter()
        self.n_emitted = 0

    def reset_output_buffer(self):
        pass

    def close(self):
        self.is_open = False
