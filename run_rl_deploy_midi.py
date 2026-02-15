"""
Porcaro Robot: Sim-to-Real RL Policy Deployment (MIDI Version)
Usage:
  python run_deploy_midi.py --midi songs/drum_pattern.mid
"""

import serial
import struct
import time
import numpy as np
import torch
import pandas as pd
import threading
import os
import argparse
from midi_rhythm_generator import MidiRhythmGenerator # 上記のクラス

# ==========================================
# 0. 引数解析
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--midi", type=str, required=True, help="Path to MIDI file")
parser.add_argument("--model", type=str, default="models/policy.pt", help="Path to policy model")
parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port")
parser.add_argument("--verify", action="store_true", help="Verification mode (no actuation)")
parser.add_argument("--force_scale", type=float, default=20.0, help="Target Force [N] for normalization")
args = parser.parse_args()

# ==========================================
# Configuration
# ==========================================
MODEL_PATH = args.model
SERIAL_PORT = args.port
BAUD_RATE = 230400
P_MAX = 0.6         # [MPa]
CONTROL_DT = 0.02   # 50Hz
SEND_FMT = '>ddd'       
RECV_FMT = '>ddddddd'   
HEADER = b'\xff\xff'
RECV_PACKET_LEN = 2 + 8 * 7

# ==========================================
# Sensor Interface (変更なし)
# ==========================================
class SensorInterface(threading.Thread):
    def __init__(self, ser):
        super().__init__()
        self.ser = ser
        self.running = True
        self.latest_data = None
        self.lock = threading.Lock()
        self.daemon = True

    def run(self):
        buffer = b''
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    buffer += self.ser.read(self.ser.in_waiting)
                    while len(buffer) >= RECV_PACKET_LEN:
                        if buffer[:2] == HEADER:
                            data = struct.unpack(RECV_FMT, buffer[2:RECV_PACKET_LEN])
                            with self.lock:
                                self.latest_data = data
                            buffer = buffer[RECV_PACKET_LEN:]
                        else:
                            buffer = buffer[1:]
                else:
                    time.sleep(0.001)
            except Exception as e:
                self.running = False

    def get_latest(self):
        with self.lock:
            return self.latest_data

# ==========================================
# Main Deployer
# ==========================================
class MidiDeployer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Device: {self.device}")

        # 1. モデルロード
        if not args.verify:
            self.policy = torch.jit.load(MODEL_PATH, map_location=self.device).eval()
        
        # 2. MIDI解析 & 軌道生成
        self.rhythm_gen = MidiRhythmGenerator(
            midi_path=args.midi,
            device=self.device,
            dt=CONTROL_DT,
            target_force=args.force_scale
        )

        # 3. シリアル通信
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            self.sensor = SensorInterface(self.ser)
            self.sensor.start()
        except Exception as e:
            print(f"[Error] Serial failed: {e}")
            exit(1)

        self.last_actions = np.zeros(3, dtype=np.float32)
        self.logs = []

    def run(self):
        print(f"Playing MIDI: {args.midi} (BPM: {self.rhythm_gen.bpm:.1f})")
        input("Press Enter to START...")
        
        start_time = time.perf_counter()
        
        try:
            while True:
                loop_start = time.perf_counter()
                t_elapsed = loop_start - start_time

                # 終了判定
                if t_elapsed > self.rhythm_gen.duration_sec + 2.0:
                    print("Song finished.")
                    break

                # 1. センサーデータ取得
                raw_data = self.sensor.get_latest()
                if raw_data is None: continue
                
                # Sim環境への合わせ込み (Deg -> Rad, 符号反転等は学習環境の定義に依存)
                # 注: porcaro_rl_env.pyではSim:Down+に対してReal:Up+に変換している
                # ここではセンサ値(Real)をそのまま使うか、環境に合わせて変換する
                obs_wrist_pos = np.radians(raw_data[3]) 
                obs_wrist_vel = np.radians(raw_data[4])
                
                # 2. リズム情報取得 (MIDIから)
                # rhythm_bufは既に正規化済み(0.0~1.0)
                phase_rad, rhythm_buf = self.rhythm_gen.get_state(t_elapsed)
                
                sin_phase = np.sin(phase_rad)
                cos_phase = np.cos(phase_rad)
                
                # 3. 観測ベクトル作成 (35次元)
                # [q(2), qd(2), prev_act(3), sin(1), cos(1), bpm(1), lookahead(25)]
                q = torch.tensor([obs_wrist_pos, 0.0], device=self.device)
                qd = torch.tensor([obs_wrist_vel, 0.0], device=self.device)
                prev_act = torch.tensor(self.last_actions, device=self.device)
                phase = torch.tensor([sin_phase, cos_phase], device=self.device)
                bpm_norm = torch.tensor([self.rhythm_gen.bpm / 180.0], device=self.device)
                
                obs = torch.cat([q, qd, prev_act, phase, bpm_norm, rhythm_buf]).unsqueeze(0).float()

                # 4. 推論
                if args.verify:
                    pressures = [0,0,0]
                    # 可視化
                    tgt = rhythm_buf[0].item() * args.force_scale
                    print(f"\rTgt:{tgt:4.1f} | {'#'*int(tgt)}", end="")
                else:
                    with torch.no_grad():
                        action = self.policy(obs).cpu().numpy().flatten()
                    
                    self.last_actions = np.clip(action, -1.0, 1.0)
                    pressures = (self.last_actions + 1.0) / 2.0 * P_MAX
                    
                    self.ser.write(HEADER + struct.pack(SEND_FMT, *pressures))

                # 5. ログ
                self.logs.append({
                    'time': t_elapsed,
                    'target': rhythm_buf[0].item() * args.force_scale,
                    'wrist_angle': raw_data[3],
                    'pressure_df': pressures[0]
                })

                # 6. 50Hz維持
                dt = time.perf_counter() - loop_start
                if dt < CONTROL_DT:
                    time.sleep(CONTROL_DT - dt)

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        self.ser.write(HEADER + struct.pack(SEND_FMT, 0, 0, 0))
        self.ser.close()
        # ログ保存
        pd.DataFrame(self.logs).to_csv(f"logs_midi_{int(time.time())}.csv")
        print("\nFinished.")

if __name__ == "__main__":
    MidiDeployer().run()