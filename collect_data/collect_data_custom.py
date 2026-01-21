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
import itertools

# ==========================================
# 設定
# ==========================================
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
DT = 0.01          # 10ms

MAX_PRESSURE = 0.6 # [MPa]
MIN_PRESSURE = 0.0 # [MPa]
GRIP_PRESSURE = 0.3 # 手首動作中にグリップがぶらつかないように軽く入れる

# 通信フォーマット
SEND_FMT = '>ddd'    # Header + DF, F, G (double x3)
RECV_FMT = '>dddddd' # Header + Pres(3) + Ang(2) + Flag(1)
RECV_PACKET_LEN = 2 + 6 * 8

class CustomDataCollector:
    def __init__(self, args):
        self.args = args
        self.ser = None
        self.is_running = True
        self.is_recording = False
        self.logs = []
        self.start_time = 0.0
        self.serial_buffer = b''
        
        # Exp 1: 完全総当たりステップ応答 (Full Permutation Step Response)
        if self.args.exp == 1:
            # 0.0 から 0.6 まで 0.1 刻み
            levels = [round(x * 0.1, 1) for x in range(7)] # [0.0, 0.1, ... 0.6]
            
            # 自分自身への遷移を除く全ペア (Start, Target) を生成
            # Permutations は順序を区別するため (0, 0.1) と (0.1, 0) 両方が生成される -> 42通り
            self.pairs = list(itertools.permutations(levels, 2))
            
            # 各ステップの構成: [準備フェーズ(Start圧), 計測フェーズ(Target圧)]
            self.phase_duration = 3.0 # 各フェーズ3秒
            
            print(f"[Exp 1] Total Transitions: {len(self.pairs)} patterns")
            print(f"[Exp 1] Estimated Time: {(len(self.pairs) * self.phase_duration * 2) / 60:.1f} min")

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

    def _get_commands(self, t):
        """実験モードごとの指令値生成ロジック"""
        cmd_DF, cmd_F, cmd_G = 0.0, 0.0, 0.0
        status_msg = ""

        # ---------------------------------------------------------
        # 【実験 1】 完全総当たりステップ応答 (Reset & Step)
        # ---------------------------------------------------------
        if self.args.exp == 1:
            # 1つのペアにつき "準備(3s)" + "計測(3s)" = 6.0s
            cycle_duration = self.phase_duration * 2
            
            idx = int(t / cycle_duration)
            phase_t = t % cycle_duration

            if idx < len(self.pairs):
                p_start, p_target = self.pairs[idx]
                
                if phase_t < self.phase_duration:
                    # 前半: 準備フェーズ (Start圧力で待機)
                    cmd_F = p_start
                    status_msg = f"[Exp1] ({idx+1}/{len(self.pairs)}) Prep: {p_start:.1f} -> (Wait)"
                else:
                    # 後半: 計測フェーズ (Target圧力へステップ)
                    cmd_F = p_target
                    status_msg = f"[Exp1] ({idx+1}/{len(self.pairs)}) Step: {p_start:.1f} -> {p_target:.1f}"
            else:
                # 全パターン終了
                cmd_F = 0.0
                status_msg = "[Exp1] Finished."
                if idx > len(self.pairs): # 少し余裕を見て停止
                    self.is_recording = False
                    self.is_running = False

            cmd_DF = 0.0
            cmd_G = 0.0

        # ---------------------------------------------------------
        # 【実験 2】 3軸連動・位相差正弦波 (Multi-Axis Hysteresis)
        # ---------------------------------------------------------
        elif self.args.exp == 2:
            f = self.args.freq
            cmd_F = 0.3 + 0.2 * np.sin(2 * np.pi * f * t)
            cmd_DF = 0.3 + 0.2 * np.sin(2 * np.pi * f * t + np.pi)
            cmd_G = 0.3 + 0.15 * np.sin(2 * np.pi * f * t + np.pi / 2)

        # ---------------------------------------------------------
        # 【実験 3】 有効収縮率 (Slack) 同定
        # ---------------------------------------------------------
        elif self.args.exp == 3:
            ramp_val = 0.05 * t # 0.05 MPa/s
            if self.args.target == 'DF':
                cmd_DF = np.clip(ramp_val, 0.0, 0.6)
            elif self.args.target == 'F':
                cmd_DF = 0.6 # ロック用
                cmd_F = np.clip(ramp_val, 0.0, 0.6)
            elif self.args.target == 'G':
                cmd_DF = 0.6 # ロック用
                cmd_G = np.clip(ramp_val, 0.0, 0.6)

        # ---------------------------------------------------------
        # 【実験 4】 Validation Sequence
        # ---------------------------------------------------------
        elif self.args.exp == 4:
            # (省略: 以前と同じ)
            pass

        return np.clip([cmd_DF, cmd_F, cmd_G], MIN_PRESSURE, MAX_PRESSURE), status_msg

    def send_packet(self, cmd_list):
        if self.ser and self.ser.is_open:
            header = b'\xFF\xFF'
            payload = struct.pack(SEND_FMT, *cmd_list)
            self.ser.write(header + payload)

    def run(self):
        self.connect()
        print("\n" + "="*60)
        print(f" 【Experiment {self.args.exp}】")
        if self.args.exp == 1: print("  Mode: Full Permutation Step Response (Reset-and-Step)")
        print("-" * 60)
        print(" [S] Start Recording (実験開始)")
        print(" [Q] Quit  (終了)")
        print("="*60 + "\n")

        input_thread = threading.Thread(target=self.input_listener)
        input_thread.daemon = True
        input_thread.start()

        try:
            while self.is_running:
                loop_start = time.time()
                t_current = 0.0
                status = ""
                
                if self.is_recording:
                    t_current = time.time() - self.start_time
                
                # 1. 指令生成
                if self.is_recording:
                    cmds, status = self._get_commands(t_current)
                    if status:
                        print(f"\r{status}   CMD: [{cmds[0]:.1f}, {cmds[1]:.1f}, {cmds[2]:.1f}]", end="")
                else:
                    # 待機中は実験3の設定によっては固定圧を入れる（前回同様）
                    if self.args.exp == 3 and self.args.target in ['F', 'G']:
                         cmds = [0.6, 0.0, 0.0]
                    else:
                         cmds = [0.0, 0.0, 0.0]

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
                                    "timestamp": t_current,
                                    "exp_mode": self.args.exp,
                                    "cmd_DF": cmds[0], "cmd_F": cmds[1], "cmd_G": cmds[2],
                                    "meas_pres_DF": recv[0], "meas_pres_F": recv[1], "meas_pres_G": recv[2],
                                    "meas_ang_wrist": recv[3], "meas_ang_hand": recv[4],
                                    "p_flag": recv[5]
                                })
                            self.serial_buffer = self.serial_buffer[RECV_PACKET_LEN:]
                        except Exception:
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
                    print(f"\n>>> START RECORDING <<<")
                    self.start_time = time.time()
                    self.is_recording = True
            elif k == 'q':
                self.is_running = False
                break

    def close(self):
        print("\nStopping...")
        self.send_packet([0.0, 0.0, 0.0])
        if self.ser: self.ser.close()
        if self.logs:
            suffix = ""
            if self.args.exp == 1: suffix = "_step_full_perm"
            elif self.args.exp == 2: suffix = f"_hysteresis_{self.args.freq}Hz"
            elif self.args.exp == 3: suffix = f"_slack_pam{self.args.target.lower()}"
            elif self.args.exp == 4: suffix = "_validation_seq"
            
            filename = f"data_exp{self.args.exp}{suffix}_{int(time.time())}.csv"
            print(f"[SAVE] {filename}")
            pd.DataFrame(self.logs).to_csv(filename, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4], help="Experiment ID (1-4)")
    parser.add_argument("--freq", type=float, default=0.5, help="Frequency for Exp 2 [Hz]")
    parser.add_argument("--target", type=str, default="DF", choices=["DF", "F", "G"], help="Target Muscle for Exp 3")
    args = parser.parse_args()
    
    collector = CustomDataCollector(args)
    collector.run()