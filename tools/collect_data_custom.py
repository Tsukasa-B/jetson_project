"""
Porcaro Robot: Pneumatic Characteristics Data Collector
Target: Jetson Orin Nano + MicroLabBox
Feature: Async 200Hz Receive / 50Hz Control with Absolute Timing Sync
"""

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
import copy

# ==========================================
# 設定
# ==========================================
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 230400       # 最新仕様に統一
CONTROL_DT = 0.02        # 制御ループ周期 (50Hz)

MAX_PRESSURE = 0.6 
MIN_PRESSURE = 0.0 

# 通信フォーマット
SEND_FMT = '>ddd'        # Header + DF, F, G (double x3)
RECV_FMT = '>ddddddd'    # Header + P_DF, P_F, P_G, Angle, Vel, Flag, Force (double x7)
HEADER = b'\xff\xff'
RECV_PACKET_LEN = 2 + 8 * 7

# ==========================================
# 1. センサー受信クラス (Async 200Hz)
# ==========================================
class SensorReceiver(threading.Thread):
    def __init__(self, ser):
        super().__init__()
        self.ser = ser
        self.running = True
        self.sensor_logs = []
        self.clear_flag = False
        self.lock = threading.Lock()
        self.daemon = True

    def run(self):
        self.ser.reset_input_buffer()
        buffer = b''
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    buffer += self.ser.read(self.ser.in_waiting)
                    while len(buffer) >= RECV_PACKET_LEN:
                        idx = buffer.find(HEADER)
                        if idx == -1:
                            buffer = buffer[-1:]
                            break
                        if idx > 0:
                            buffer = buffer[idx:]
                        if len(buffer) < RECV_PACKET_LEN:
                            break

                        packet = buffer[:RECV_PACKET_LEN]
                        buffer = buffer[RECV_PACKET_LEN:]
                        self._process_packet(packet[2:])
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"[Rx Error] {e}")
                self.running = False

    def _process_packet(self, packet_bytes):
        try:
            data = struct.unpack(RECV_FMT, packet_bytes)
            sample = {
                'meas_pres_DF': data[0],
                'meas_pres_F':  data[1],
                'meas_pres_G':  data[2],
                'angle_deg':    data[3],
                'velocity':     data[4],
                'flag':         data[5],
                'force_N':      data[6]
            }
            with self.lock:
                if self.clear_flag:
                    self.sensor_logs = []
                    self.clear_flag = False
                self.sensor_logs.append(sample)
        except: pass

    def clear_buffer_for_sync(self):
        self.ser.reset_input_buffer()
        with self.lock:
            self.clear_flag = True

    def get_all_logs(self):
        with self.lock:
            return self.sensor_logs[:]

    def stop(self):
        self.running = False

# ==========================================
# 2. メイン制御クラス
# ==========================================
class CustomDataCollector:
    def __init__(self, args):
        self.args = args
        self.ser = None
        self.is_running = True
        self.is_recording = False
        self.cmd_logs = []
        self.start_time = 0.0
        self.step_idx = 0
        
        # Exp 1 設定 (全順列のステップ応答・ヒステリシス計測)
        if self.args.exp == 1:
            levels = [round(x * 0.1, 1) for x in range(7)] 
            self.pairs = list(itertools.permutations(levels, 2))
            self.phase_duration = 3.0 
            print(f"[Exp 1] Total Transitions: {len(self.pairs)} patterns")

    def connect(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            print(f"[INFO] Serial connected to {SERIAL_PORT} @ {BAUD_RATE} bps")
            self.receiver = SensorReceiver(self.ser)
            self.receiver.start()
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

    def input_listener(self):
        while self.is_running:
            k = self.get_keypress().lower()
            if k == 's':
                if not self.is_recording:
                    print(f"\n>>> START RECORDING <<<")
                    # 時刻0の厳密な同期
                    self.receiver.clear_buffer_for_sync()
                    self.start_time = time.perf_counter()
                    self.step_idx = 0
                    self.is_recording = True
            elif k == 'q':
                self.is_running = False
                break

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
                if idx >= len(self.pairs): 
                    self.is_recording = False
                    self.is_running = False

        elif self.args.exp == 2:
            f = self.args.freq
            cmd_F = 0.3 + 0.2 * np.sin(2 * np.pi * f * t)
            cmd_DF = 0.3 + 0.2 * np.sin(2 * np.pi * f * t + np.pi)
            cmd_G = 0.3 + 0.15 * np.sin(2 * np.pi * f * t + np.pi / 2)
            status_msg = f"[Exp2] Sine Sweep ({f}Hz)"

        elif self.args.exp == 3:
            ramp_val = 0.05 * t 
            if self.args.target == 'DF': cmd_DF = np.clip(ramp_val, 0.0, 0.6)
            elif self.args.target == 'F':  cmd_DF = 0.6; cmd_F = np.clip(ramp_val, 0.0, 0.6)
            elif self.args.target == 'G':  cmd_DF = 0.6; cmd_G = np.clip(ramp_val, 0.0, 0.6)
            status_msg = f"[Exp3] Ramp up {self.args.target}"
            
        elif self.args.exp == 4:
            t_mod = t % 10.0
            if t_mod < 5.0:
                 cmd_F = 0.3 + 0.2 * np.sin(2 * np.pi * 1.0 * t)
                 status_msg = "[Exp4] Sine Phase"
            else:
                 cmd_F = 0.5 if (t_mod < 7.5) else 0.1
                 status_msg = "[Exp4] Step Phase"
            cmd_DF = 0.6

        return np.clip([cmd_DF, cmd_F, cmd_G], MIN_PRESSURE, MAX_PRESSURE), status_msg

    def send_packet(self, cmd_list):
        if self.ser and self.ser.is_open:
            packet = HEADER + struct.pack(SEND_FMT, *cmd_list)
            self.ser.write(packet)

    def run(self):
        self.connect()
        
        print("\n" + "="*60)
        print(f" 【Experiment {self.args.exp} (Async 200Hz / Control 50Hz)】")
        print(" [S] Start Recording")
        print(" [Q] Quit")
        print("="*60 + "\n")

        input_thread = threading.Thread(target=self.input_listener)
        input_thread.daemon = True
        input_thread.start()

        # 姿勢を初期化して安定するまで待機
        self.send_packet([0,0,0])

        try:
            while self.is_running:
                loop_start = time.perf_counter()
                
                if self.is_recording:
                    # 数学的な理想時刻を算出
                    t_math = self.step_idx * CONTROL_DT
                    cmds, status = self._get_commands(t_math)
                    
                    if status and self.step_idx % 5 == 0:
                        print(f"\r{status} t={t_math:.2f}s", end="")
                    
                    self.send_packet(cmds)
                    
                    # ログ保存
                    self.cmd_logs.append({
                        'cmd_time': t_math,
                        'exp_mode': self.args.exp,
                        'cmd_DF': cmds[0],
                        'cmd_F': cmds[1],
                        'cmd_G': cmds[2]
                    })
                    
                    self.step_idx += 1

                    # 絶対時刻ベースの待機 (Drift防止)
                    target_next_time = self.start_time + (self.step_idx * CONTROL_DT)
                    while time.perf_counter() < target_next_time:
                        time.sleep(0.0005)
                        
                else:
                    if self.args.exp == 3 and self.args.target in ['F', 'G']:
                         cmds = [0.6, 0.0, 0.0]
                    else:
                         cmds = [0.0, 0.0, 0.0]
                    self.send_packet(cmds)
                    
                    # 待機中は相対待機
                    while (time.perf_counter() - loop_start) < CONTROL_DT:
                        time.sleep(0.0005)

        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        print("\nStopping...")
        self.send_packet([0.0, 0.0, 0.0])
        self.is_running = False
        self.receiver.stop()
        self.ser.close()
        self._save_logs()

    def _save_logs(self):
        if not self.cmd_logs:
            print("[Warning] No data recorded.")
            return

        sensor_data = self.receiver.get_all_logs()
        if not sensor_data:
            print("[Error] No sensor data received.")
            return

        print("\nReconstructing Timestamps and Merging...")
        
        # 1. センサーデータ (200Hz) の再構築
        df_sensor = pd.DataFrame(sensor_data)
        df_sensor['time'] = np.arange(len(df_sensor)) * 0.005

        # 2. 指令値データ (50Hz)
        df_cmd = pd.DataFrame(self.cmd_logs)
        df_cmd = df_cmd.rename(columns={'cmd_time': 'time'})

        # 3. マージ (Zero-Order Hold)
        df_merged = pd.merge_asof(
            df_sensor,
            df_cmd,
            on='time',
            direction='backward'
        )

        suffix = f"_exp{self.args.exp}_async"
        filename = f"data_characteristics{suffix}_{int(time.time())}.csv"
        df_merged.to_csv(filename, index=False)
        print(f"[Saved] Reconstructed High-Res Log (200Hz) saved to:\n  -> {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4], help="Experiment ID")
    parser.add_argument("--freq", type=float, default=0.5, help="Frequency for Exp 2")
    parser.add_argument("--target", type=str, default="DF", choices=["DF", "F", "G"], help="Target Muscle for Exp 3")
    args = parser.parse_args()
    
    collector = CustomDataCollector(args)
    collector.run()