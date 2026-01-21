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
from collections import deque
import queue

# ==========================================
# 設定
# ==========================================
SERIAL_PORT = "/dev/ttyUSB0"
# 【重要】115200では1kHz計測には帯域不足(約23Hz制限)になります。
# 可能であればMicroLabBox側も合わせ、ここを 460800 や 921600 に上げてください。
BAUD_RATE = 115200 
DT = 0.01          # 制御ループ周期 (100Hz)

MAX_PRESSURE = 0.6 
MIN_PRESSURE = 0.0 

# 通信フォーマット
SEND_FMT = '>ddd'    # Header + DF, F, G (double x3)
RECV_FMT = '>dddddd' # Header + Pres(3) + Ang(2) + Flag(1)
RECV_PACKET_LEN = 2 + 6 * 8

class SerialReceiver(threading.Thread):
    """
    受信専用スレッド
    メインループの遅延に影響されず、データ到着時刻を正確に記録する
    """
    def __init__(self, ser, data_queue):
        super().__init__()
        self.ser = ser
        self.data_queue = data_queue
        self.running = True
        self.buffer = b''

    def run(self):
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    # データ到着！即座に読み込み
                    chunk = self.ser.read(self.ser.in_waiting)
                    arrival_time = time.time() # 【重要】到着時刻を刻印
                    self.buffer += chunk

                    while len(self.buffer) >= RECV_PACKET_LEN:
                        # ヘッダ探索 (0xFF, 0xFF)
                        if self.buffer[0] == 0xFF and self.buffer[1] == 0xFF:
                            packet_data = self.buffer[2:RECV_PACKET_LEN]
                            try:
                                values = struct.unpack(RECV_FMT, packet_data)
                                # (時刻, データタプル) をキューに送る
                                self.data_queue.put((arrival_time, values))
                            except Exception as e:
                                print(f"[Receiver Error] {e}")
                            
                            self.buffer = self.buffer[RECV_PACKET_LEN:]
                        else:
                            # ヘッダ不一致、1バイト進める
                            self.buffer = self.buffer[1:]
                else:
                    time.sleep(0.001) # CPU負荷軽減
            except Exception as e:
                if self.running: print(f"[Serial Error] {e}")
                break

    def stop(self):
        self.running = False

class CustomDataCollector:
    def __init__(self, args):
        self.args = args
        self.ser = None
        self.is_running = True
        self.is_recording = False
        self.logs = []
        self.start_time = 0.0
        
        # スレッド間通信用キュー
        self.recv_queue = queue.Queue()
        self.receiver_thread = None

        # Exp 1 設定
        if self.args.exp == 1:
            levels = [round(x * 0.1, 1) for x in range(7)] 
            self.pairs = list(itertools.permutations(levels, 2))
            self.phase_duration = 3.0 
            print(f"[Exp 1] Total Transitions: {len(self.pairs)} patterns")

    def connect(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            print(f"[INFO] Serial connected to {SERIAL_PORT} @ {BAUD_RATE} bps")
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            sys.exit(1)

    def start_receiver(self):
        self.receiver_thread = SerialReceiver(self.ser, self.recv_queue)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()

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
        """ 実験モードごとの指令生成 """
        cmd_DF, cmd_F, cmd_G = 0.0, 0.0, 0.0
        status_msg = ""

        if self.args.exp == 1:
            cycle_duration = self.phase_duration * 2
            idx = int(t / cycle_duration)
            phase_t = t % cycle_duration
            if idx < len(self.pairs):
                p_start, p_target = self.pairs[idx]
                if phase_t < self.phase_duration:
                    cmd_F = p_start
                    status_msg = f"[Exp1] ({idx+1}/{len(self.pairs)}) Prep: {p_start:.1f}"
                else:
                    cmd_F = p_target
                    status_msg = f"[Exp1] ({idx+1}/{len(self.pairs)}) Step: {p_start:.1f} -> {p_target:.1f}"
            else:
                cmd_F = 0.0
                status_msg = "[Exp1] Finished."
                if idx > len(self.pairs): 
                    self.is_recording = False
                    self.is_running = False

        elif self.args.exp == 2:
            f = self.args.freq
            cmd_F = 0.3 + 0.2 * np.sin(2 * np.pi * f * t)
            cmd_DF = 0.3 + 0.2 * np.sin(2 * np.pi * f * t + np.pi)
            cmd_G = 0.3 + 0.15 * np.sin(2 * np.pi * f * t + np.pi / 2)

        elif self.args.exp == 3:
            ramp_val = 0.05 * t 
            if self.args.target == 'DF': cmd_DF = np.clip(ramp_val, 0.0, 0.6)
            elif self.args.target == 'F':  cmd_DF = 0.6; cmd_F = np.clip(ramp_val, 0.0, 0.6)
            elif self.args.target == 'G':  cmd_DF = 0.6; cmd_G = np.clip(ramp_val, 0.0, 0.6)
            
        elif self.args.exp == 4:
            # Validation Sequence (Sin + Step mix)
            t_mod = t % 10.0
            if t_mod < 5.0: # Sine
                 cmd_F = 0.3 + 0.2 * np.sin(2 * np.pi * 1.0 * t)
                 status_msg = "[Exp4] Sine Phase"
            else: # Step
                 cmd_F = 0.5 if (t_mod < 7.5) else 0.1
                 status_msg = "[Exp4] Step Phase"
            cmd_DF = 0.6 # Antagonist lock

        return np.clip([cmd_DF, cmd_F, cmd_G], MIN_PRESSURE, MAX_PRESSURE), status_msg

    def send_packet(self, cmd_list):
        if self.ser and self.ser.is_open:
            header = b'\xFF\xFF'
            payload = struct.pack(SEND_FMT, *cmd_list)
            self.ser.write(header + payload)

    def run(self):
        self.connect()
        self.start_receiver() # 受信スレッド開始
        
        print("\n" + "="*60)
        print(f" 【Experiment {self.args.exp} (Async Receiver)】")
        print(" [S] Start Recording")
        print(" [Q] Quit")
        print("="*60 + "\n")

        input_thread = threading.Thread(target=self.input_listener)
        input_thread.daemon = True
        input_thread.start()

        last_cmds = [0.0, 0.0, 0.0]

        try:
            while self.is_running:
                loop_start = time.time()
                t_current = 0.0
                
                if self.is_recording:
                    t_current = time.time() - self.start_time

                # 1. 指令生成 (Main Thread)
                if self.is_recording:
                    cmds, status = self._get_commands(t_current)
                    if status and int(t_current*10)%10 == 0: # 間引いて表示
                        print(f"\r{status} t={t_current:.2f}", end="")
                    last_cmds = cmds
                else:
                    if self.args.exp == 3 and self.args.target in ['F', 'G']:
                         cmds = [0.6, 0.0, 0.0]
                    else:
                         cmds = [0.0, 0.0, 0.0]

                # 2. 送信
                self.send_packet(cmds)

                # 3. 受信データの回収 (Queueから吸い出し)
                while not self.recv_queue.empty():
                    arrival_time, recv = self.recv_queue.get()
                    
                    if self.is_recording:
                        # 記録開始時刻からの相対時間
                        t_log = arrival_time - self.start_time
                        if t_log >= 0:
                            self.logs.append({
                                "timestamp": t_log, # これで階段状にならない！
                                "exp_mode": self.args.exp,
                                "cmd_DF": last_cmds[0], "cmd_F": last_cmds[1], "cmd_G": last_cmds[2],
                                "meas_pres_DF": recv[0], "meas_pres_F": recv[1], "meas_pres_G": recv[2],
                                "meas_ang_wrist": recv[3], "meas_ang_hand": recv[4],
                                "p_flag": recv[5]
                            })

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
        if self.receiver_thread: self.receiver_thread.stop()
        self.send_packet([0.0, 0.0, 0.0])
        if self.ser: self.ser.close()
        
        if self.logs:
            suffix = f"_exp{self.args.exp}_async"
            filename = f"data{suffix}_{int(time.time())}.csv"
            print(f"[SAVE] {filename} ({len(self.logs)} rows)")
            pd.DataFrame(self.logs).to_csv(filename, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4], help="Experiment ID")
    parser.add_argument("--freq", type=float, default=0.5, help="Frequency for Exp 2")
    parser.add_argument("--target", type=str, default="DF", choices=["DF", "F", "G"], help="Target Muscle for Exp 3")
    args = parser.parse_args()
    
    collector = CustomDataCollector(args)
    collector.run()