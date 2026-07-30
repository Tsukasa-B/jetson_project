import serial
import struct
import time
import numpy as np
import pandas as pd
import threading
import sys
import termios
import tty
import os
import argparse

# ==========================================
# 設定
# ==========================================
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
DT = 0.01          # 10ms

MAX_PRESSURE = 0.6 # [MPa]
MIN_PRESSURE = 0.0 # [MPa]

# 通信フォーマット (Verification Script準拠)
SEND_FMT = '>ddd' # Header + DF, F, G (double x3)
RECV_FMT = '>dddddd' # Header + Pres(3) + Ang(2) + Flag(1)
RECV_PACKET_LEN = 2 + 6 * 8

class RealDataCollector:
    def __init__(self, mode="train"):
        self.ser = None
        self.is_running = True
        self.is_recording = False
        self.logs = []
        self.segment_id = 0
        self.mode = mode
        self.serial_buffer = b''

    def connect(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            print(f"[INFO] Serial connected to {SERIAL_PORT}")
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            sys.exit(1)

    def get_keypress(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def _generate_single_channel_wave(self, t, seed_offset, wave_type="random"):
        """チャンネルごとの波形生成 (内部関数)"""
        target = 0.0
        
        if wave_type == "random":
            # Seedをチャンネルごとにずらすことで独立した動きを作る
            # 2秒ごとにステップ変化
            seed_step = int(t / 2.0) + seed_offset * 1000
            np.random.seed(seed_step)
            step_val = np.random.uniform(MIN_PRESSURE, MAX_PRESSURE)
            
            # Chirp振動 (周波数も微妙にずらす)
            freq = 0.5 + 4.5 * (np.sin(t / (15.0 + seed_offset)) ** 2)
            sine_val = 0.1 * np.sin(2 * np.pi * freq * t)
            target = step_val + sine_val

        elif wave_type == "triangle":
            # チャンネルごとに周期を変える (素数を使うと位相が重なりにくい)
            # offset 0(DF):7s, 1(F):11s, 2(G):13s
            periods = [7.0, 11.0, 13.0]
            T = periods[seed_offset % 3]
            
            phase = (t % T) / T
            if phase < 0.5:
                target = (phase * 2) * MAX_PRESSURE
            else:
                target = (2.0 - phase * 2) * MAX_PRESSURE

        elif wave_type == "step":
            # 3秒ごとの階段波。位相をずらす
            # DF: 0->0.3->0.6, F: 0.6->0->0.3, G: ...
            T_step = 3.0
            idx = (int(t / T_step) + seed_offset) % 4
            vals = [0.0, 0.3, 0.6, 0.0]
            target = vals[idx]

        return np.clip(target, MIN_PRESSURE, MAX_PRESSURE)

    def generate_multichannel_command(self, t):
        """3つのPAMに対する指令を一括生成"""
        cmds = [0.0, 0.0, 0.0] # [DF, F, G]
        
        if self.mode == "train":
            # 全チャンネル独立ランダム (ActuatorNet学習用)
            cmds[0] = self._generate_single_channel_wave(t, seed_offset=0, wave_type="random") # DF
            cmds[1] = self._generate_single_channel_wave(t, seed_offset=1, wave_type="random") # F
            cmds[2] = self._generate_single_channel_wave(t, seed_offset=2, wave_type="random") # G
            
        elif self.mode == "hysteresis":
            # 非同期マルチ三角波 (相互干渉ヒステリシス同定用)
            cmds[0] = self._generate_single_channel_wave(t, seed_offset=0, wave_type="triangle") # 7s周期
            cmds[1] = self._generate_single_channel_wave(t, seed_offset=1, wave_type="triangle") # 11s周期
            cmds[2] = self._generate_single_channel_wave(t, seed_offset=2, wave_type="triangle") # 13s周期
            
        elif self.mode == "step":
            # 位相ずらしステップ (Sim比較用)
            cmds[0] = self._generate_single_channel_wave(t, seed_offset=0, wave_type="step")
            cmds[1] = self._generate_single_channel_wave(t, seed_offset=1, wave_type="step")
            cmds[2] = self._generate_single_channel_wave(t, seed_offset=2, wave_type="step")
            
        return cmds

    def send_packet(self, cmd_list):
        if self.ser and self.ser.is_open:
            header = b'\xFF\xFF'
            payload = struct.pack(SEND_FMT, *cmd_list)
            self.ser.write(header + payload)

    def run(self):
        self.connect()
        print("\n" + "="*60)
        print(f" 【ActuatorNet 3-AXIS COLLECTOR: {self.mode.upper()} MODE】")
        if self.mode == "train":
            print("  ★全軸独立ランダム駆動: 拮抗・連成動作を網羅学習します")
        elif self.mode == "hysteresis":
            print("  ★非同期三角波スイープ: 相互干渉を含むヒステリシスを取得")
        print("-" * 60)
        print(" [S] Start (開始)")
        print(" [P] Pause (一時停止 - コンプレッサー回復待機)")
        print(" [Q] Quit  (終了)")
        print("="*60 + "\n")

        input_thread = threading.Thread(target=self.input_listener)
        input_thread.daemon = True
        input_thread.start()

        t_start_program = time.time()

        try:
            while self.is_running:
                loop_start = time.time()
                t_current = loop_start - t_start_program

                # 1. 指令生成 (3ch分)
                cmds = [0.0, 0.0, 0.0]
                if self.is_recording:
                    cmds = self.generate_multichannel_command(t_current)
                
                # 2. 送信
                self.send_packet(cmds)

                # 3. 受信
                if self.ser.in_waiting > 0:
                    self.serial_buffer += self.ser.read(self.ser.in_waiting)

                while len(self.serial_buffer) >= RECV_PACKET_LEN:
                    if self.serial_buffer[0] == 0xFF and self.serial_buffer[1] == 0xFF:
                        try:
                            packet_data = self.serial_buffer[2:RECV_PACKET_LEN]
                            recv = struct.unpack(RECV_FMT, packet_data)
                            
                            if self.is_recording:
                                self.logs.append({
                                    "timestamp": time.time(),
                                    "segment_id": self.segment_id,
                                    "mode": self.mode,
                                    # 指令値
                                    "cmd_DF": cmds[0],
                                    "cmd_F":  cmds[1],
                                    "cmd_G":  cmds[2],
                                    # 実測値
                                    "meas_pres_DF": recv[0],
                                    "meas_pres_F":  recv[1],
                                    "meas_pres_G":  recv[2],
                                    "meas_ang_wrist": recv[3],
                                    "meas_ang_hand":  recv[4],
                                    "p_flag": recv[5]
                                })
                            
                            self.serial_buffer = self.serial_buffer[RECV_PACKET_LEN:]
                        except Exception as e:
                            self.serial_buffer = self.serial_buffer[1:]
                    else:
                        self.serial_buffer = self.serial_buffer[1:]

                elapsed = time.time() - loop_start
                if elapsed < DT:
                    time.sleep(DT - elapsed)

        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def input_listener(self):
        while self.is_running:
            k = self.get_keypress().lower()
            if k == 's':
                if not self.is_recording:
                    print(f"\n>>> START: Multi-Axis {self.mode.upper()} (Seg {self.segment_id}) <<<")
                    self.is_recording = True
            elif k == 'p':
                if self.is_recording:
                    print(f"\n||| PAUSE ||| Seg {self.segment_id} Done.")
                    self.is_recording = False
                    self.segment_id += 1
                    self.send_packet([0.0, 0.0, 0.0])
            elif k == 'q':
                self.is_running = False
                break

    def close(self):
        self.send_packet([0.0, 0.0, 0.0])
        if self.ser: self.ser.close()
        if self.logs:
            filename = f"real_data_multi_{self.mode}_{int(time.time())}.csv"
            print(f"\n[SAVE] {filename}")
            pd.DataFrame(self.logs).to_csv(filename, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "step", "hysteresis"])
    args = parser.parse_args()
    collector = RealDataCollector(mode=args.mode)
    collector.run()