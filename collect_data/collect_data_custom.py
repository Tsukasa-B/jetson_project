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
# 設定 (実験計画書準拠)
# ==========================================
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
DT = 0.01          # 10ms

MAX_PRESSURE = 0.6 # [MPa]
MIN_PRESSURE = 0.0 # [MPa]

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
        
        # Exp 1用: ステップ遷移の生成 (42通り)
        if self.args.exp == 1:
            levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            # 全順列 (source -> target) を生成
            self.step_sequence = list(itertools.permutations(levels, 2))
            self.step_duration = 3.0 # [sec]
            print(f"[Exp 1] Total Step Transitions: {len(self.step_sequence)}")

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

        # ---------------------------------------------------------
        # 【実験 1】 PAMF単独ステップ応答 (Pure Dynamics)
        # ---------------------------------------------------------
        if self.args.exp == 1:
            # 3秒ごとにターゲットを切り替え
            idx = int(t / self.step_duration)
            if idx < len(self.step_sequence):
                # 遷移の後半(target)を出力
                # ※厳密な過渡応答を見るため、瞬時にtargetへ切り替える
                target = self.step_sequence[idx][1]
                cmd_F = target
            else:
                cmd_F = 0.0 # 終了後は0
            
            cmd_DF = 0.0
            cmd_G = 0.0

        # ---------------------------------------------------------
        # 【実験 2】 3軸連動・位相差正弦波 (Multi-Axis Hysteresis)
        # ---------------------------------------------------------
        elif self.args.exp == 2:
            f = self.args.freq
            # Offset + Amp * sin(...)
            # PAMF: 0.3 + 0.2 sin(wt)
            cmd_F = 0.3 + 0.2 * np.sin(2 * np.pi * f * t)
            # PAMDF: 0.3 + 0.2 sin(wt + pi) (逆位相)
            cmd_DF = 0.3 + 0.2 * np.sin(2 * np.pi * f * t + np.pi)
            # PAMG: 0.3 + 0.15 sin(wt + pi/2) (90degずれ)
            cmd_G = 0.3 + 0.15 * np.sin(2 * np.pi * f * t + np.pi / 2)

        # ---------------------------------------------------------
        # 【実験 3】 有効収縮率 (Slack) 同定
        # ---------------------------------------------------------
        elif self.args.exp == 3:
            # Ramp速度: 0.01 MPa/s
            ramp_val = 0.01 * t
            
            if self.args.target == 'DF': # 3-A
                cmd_DF = np.clip(ramp_val, 0.0, 0.3) # 0 -> 0.3
                cmd_F = 0.0
                cmd_G = 0.0
                
            elif self.args.target == 'F': # 3-B
                cmd_DF = 0.5 # 固定 (Holding)
                cmd_F = np.clip(ramp_val, 0.0, 0.6) # 0 -> 0.6
                cmd_G = 0.0
                
            elif self.args.target == 'G': # 3-C
                cmd_DF = 0.6 # 最大背屈固定
                cmd_F = 0.0
                cmd_G = np.clip(ramp_val, 0.0, 0.6) # 0 -> 0.6

        # ---------------------------------------------------------
        # 【実験 4】 Sim-to-Real検証用 (Validation)
        # ---------------------------------------------------------
        elif self.args.exp == 4:
            # ActuatorNet学習データに近いランダム波形
            # 2秒ごとにシードを変えてランダムステップ＋チャープ
            for i, offset in enumerate([0, 1, 2]):
                seed_step = int(t / 2.0) + offset * 1000
                np.random.seed(seed_step)
                step = np.random.uniform(0.0, 0.6)
                
                freq = 0.5 + 4.5 * (np.sin(t / (15.0 + offset)) ** 2)
                sine = 0.1 * np.sin(2 * np.pi * freq * t)
                val = np.clip(step + sine, 0.0, 0.6)
                
                if i == 0: cmd_DF = val
                elif i == 1: cmd_F = val
                elif i == 2: cmd_G = val

        return np.clip([cmd_DF, cmd_F, cmd_G], MIN_PRESSURE, MAX_PRESSURE)

    def send_packet(self, cmd_list):
        if self.ser and self.ser.is_open:
            header = b'\xFF\xFF'
            payload = struct.pack(SEND_FMT, *cmd_list)
            self.ser.write(header + payload)

    def run(self):
        self.connect()
        print("\n" + "="*60)
        print(f" 【Experiment {self.args.exp}: {self.get_exp_name()}】")
        if self.args.exp == 2: print(f"  Frequency: {self.args.freq} Hz")
        if self.args.exp == 3: print(f"  Target: PAM-{self.args.target}")
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
                
                # 録画中のみ時間を進める（開始時にt=0リセット）
                t_current = 0.0
                if self.is_recording:
                    t_current = time.time() - self.start_time
                
                # 1. 指令生成
                if self.is_recording:
                    cmds = self._get_commands(t_current)
                else:
                    # 待機中はExperiment 3の初期位置保持のため、
                    # 3-B/3-Cの場合は拮抗筋に圧を入れる必要がある
                    if self.args.exp == 3 and self.args.target == 'F':
                        cmds = [0.5, 0.0, 0.0] # DF保持
                    elif self.args.exp == 3 and self.args.target == 'G':
                        cmds = [0.6, 0.0, 0.0] # DF保持
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
                                    "target_arg": self.args.target if self.args.exp==3 else "",
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

    def get_exp_name(self):
        names = {
            1: "Step Response (PAM-F)",
            2: "Multi-Axis Hysteresis",
            3: "Slack Identification",
            4: "Validation Data"
        }
        return names.get(self.args.exp, "Unknown")

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
        self.send_packet([0.0, 0.0, 0.0])
        if self.ser: self.ser.close()
        if self.logs:
            # 保存ファイル名の生成
            suffix = ""
            if self.args.exp == 1: suffix = "_step_pamf"
            elif self.args.exp == 2: suffix = f"_hysteresis_{self.args.freq}Hz"
            elif self.args.exp == 3: suffix = f"_slack_pam{self.args.target.lower()}"
            elif self.args.exp == 4: suffix = "_validation"
            
            filename = f"data_exp{self.args.exp}{suffix}_{int(time.time())}.csv"
            print(f"\n[SAVE] {filename}")
            pd.DataFrame(self.logs).to_csv(filename, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4], help="Experiment ID (1-4)")
    parser.add_argument("--freq", type=float, default=0.5, help="Frequency for Exp 2 [Hz]")
    parser.add_argument("--target", type=str, default="DF", choices=["DF", "F", "G"], help="Target Muscle for Exp 3")
    args = parser.parse_args()
    
    collector = CustomDataCollector(args)
    collector.run()