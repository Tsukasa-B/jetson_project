"""
Porcaro Robot: Sim-to-Real RL Policy Deployment (MIDI Version)
Target: Jetson Orin Nano + MicroLabBox
Feature: 
  - Async Communication: Recv @ 200Hz / Control @ 50Hz (Absolute Timing Sync)
  - Time Reconstruction for High-Res Logging
  - PyTorch Policy Inference

Usage:
  python run_rl_deploy_midi.py --midi songs/drum_pattern.mid --bpm 120
  python run_rl_deploy_midi.py --midi songs/pattern.mid --model models/best_policy_v2.pt
<<<<<<< HEAD
  python run_rl_deploy_midi.py --midi songs/test_single4_bpm60.mid --model models/modelB_DR_1499_03-1_lookahead5.pt
  python run_rl_deploy_midi.py --midi songs/test_single8_bpm120.mid --model models/modelB_DR_1499_03-1_lookahead5.pt
  python run_rl_deploy_midi.py --midi songs/test_single8_bpm160.mid --model models/modelB_DR_1499_03-1_lookahead5.pt
  python run_rl_deploy_midi.py --midi songs/test_double_bpm60.mid --model models/modelB_DR_1499_03-1_lookahead5.pt
  python run_rl_deploy_midi.py --midi songs/test_double_bpm120.mid --model models/modelB_DR_1499_03-1_lookahead5.pt
  python run_rl_deploy_midi.py --midi songs/test_double_bpm160.mid --model models/modelB_DR_1499_03-1_lookahead5.pt
  python run_rl_deploy_midi.py --midi songs/gmd_04_extreme_bpm170.mid --model models/modelB_DR_1499_03-1_lookahead5.pt
  python run_rl_deploy_midi.py --midi songs/gmd_03_high_bpm138.mid --model models/modelB_DR_1499_03-1_lookahead5.pt
  python run_rl_deploy_midi.py --midi songs/gmd_02_mid_bpm105.mid --model models/modelB_DR_1499_03-1_lookahead5.pt
  python run_rl_deploy_midi.py --midi songs/gmd_01_low_bpm80.mid --model models/modelB_DR_1499_03-1_lookahead5.pt
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
import copy

# ★ 追加: 外部のPyTorch版ジェネレータをインポート
from midi_rhythm_generator import MidiRhythmGenerator

# ==========================================
# 0. 引数解析 & 定数定義
# ==========================================
parser = argparse.ArgumentParser(description="Porcaro MIDI Deployment")
parser.add_argument("--midi", type=str, required=True, help="Path to MIDI file")
parser.add_argument("--bpm", type=float, default=None, help="Override BPM (Optional)")
parser.add_argument("--model", type=str, default="models/policy.pt", help="Path to policy.pt")
parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port")
parser.add_argument("--verify", action="store_true", help="Verification mode (No actuation)")
parser.add_argument("--force_scale", type=float, default=20.0, help="Target Force [N]")
args = parser.parse_args()

# System Config
SERIAL_PORT = args.port
BAUD_RATE = 230400
CONTROL_DT = 0.02
P_MAX = 0.6
LOOKAHEAD_STEPS = 25

SEND_FMT = '>ddd'
RECV_FMT = '>ddddddd'
HEADER = b'\xff\xff'
RECV_PACKET_LEN = 2 + 8 * 7

# ==========================================
# 1. センサー受信クラス (変更なしのため省略せずにそのまま使用)
# ==========================================
class SensorReceiver(threading.Thread):
    def __init__(self, ser):
        super().__init__()
        self.ser = ser
        self.running = True
        self.latest_data = None
        self.sensor_logs = []
        self.clear_flag = False
        self.lock = threading.Lock()
        self.daemon = True

        self.prev_time = time.time()
        self.prev_wrist_angle = 0.0
        self.prev_grip_angle = 0.0

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
                        self._update(packet[2:])
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"[Rx Error] {e}")
                self.running = False

    def _update(self, packet_bytes):
        try:
            data = struct.unpack(RECV_FMT, packet_bytes)
            
            wrist_angle_deg = data[3]
            grip_angle_deg  = data[4]
            
            current_time = time.time()
            dt = current_time - self.prev_time
            
            if dt > 0:
                wrist_velocity = (wrist_angle_deg - self.prev_wrist_angle) / dt
                grip_velocity  = (grip_angle_deg - self.prev_grip_angle) / dt
            else:
                wrist_velocity = 0.0
                grip_velocity  = 0.0
                
            self.prev_time = current_time
            self.prev_wrist_angle = wrist_angle_deg
            self.prev_grip_angle  = grip_angle_deg

            sample = {
                'meas_pres_DF':    data[0],
                'meas_pres_F':     data[1],
                'meas_pres_G':     data[2],
                'wrist_angle_deg': wrist_angle_deg,
                'wrist_velocity':  wrist_velocity,
                'grip_angle_deg':  grip_angle_deg,
                'grip_velocity':   grip_velocity,
                'flag':            data[5],
                'force_N':         data[6]
            }
            with self.lock:
                if self.clear_flag:
                    self.sensor_logs = []
                    self.clear_flag = False
                self.sensor_logs.append(sample)
                self.latest_data = sample
        except: pass

    def clear_buffer_for_sync(self):
        self.ser.reset_input_buffer()
        with self.lock:
            self.clear_flag = True

    def get_latest(self):
        with self.lock:
            if self.latest_data is None:
                return None
            return self.latest_data.copy()

    def get_all_logs(self):
        with self.lock:
            return self.sensor_logs[:]


# ==========================================
# 3. メイン制御クラス (MIDI Deployer)
# ==========================================
class MidiDeployer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Device: {self.device}")

        if not args.verify:
            if not os.path.exists(args.model):
                raise FileNotFoundError(f"Model not found: {args.model}")
            self.policy = torch.jit.load(args.model, map_location=self.device)
            self.policy.eval()

        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            self.receiver = SensorReceiver(self.ser)
            self.receiver.start()
        except Exception as e:
            print(f"[Error] Serial Failed: {e}")
            exit(1)

        # ★ 修正: 外部のPyTorch版に合わせて引数を変更
        self.rhythm_gen = MidiRhythmGenerator(
            midi_path=args.midi, 
            device=self.device,      # デバイスを渡す
            dt=CONTROL_DT, 
            target_force=args.force_scale, 
            lookahead_steps=LOOKAHEAD_STEPS,
            override_bpm=args.bpm  # 追加した引数
        )

        self.cmd_logs = []
        self.last_actions = np.zeros(3) 
        self.start_time = None

        self.prev_q_wrist = None
        self.prev_q_grip = None

    def run(self):
        print("\n=== MIDI DEPLOYMENT START ===")
        print(f" Song: {args.midi}")
        print(f" Control: 50Hz (Absolute Sync) | Recv: Async 200Hz")
        
        self._send_pressure(0, 0, 0)
        time.sleep(2.0)
        input(">>> Press ENTER to Start... ")
        
        self.receiver.clear_buffer_for_sync()
        self.start_time = time.perf_counter()
        step_idx = 0
        
        try:
            while True:
                t_math = step_idx * CONTROL_DT

                if t_math > self.rhythm_gen.duration_sec:
                    print("\nSong Finished.")
                    break

                sensor = self.receiver.get_latest()
                if sensor is None:
                    time.sleep(0.001)
                    continue

                q_wrist = np.radians(sensor['wrist_angle_deg'])
                q_grip  = np.radians(sensor['grip_angle_deg'])
                # 2. ★ 速度の安定計算 (50Hzの安定したループ周期 dt=0.02 で割る)
                if self.prev_q_wrist is None:
                    qd_wrist = 0.0
                    qd_grip  = 0.0
                else:
                    raw_qd_wrist = (q_wrist - self.prev_q_wrist) / CONTROL_DT
                    raw_qd_grip  = (q_grip - self.prev_q_grip) / CONTROL_DT
                    # ±20.0 rad/s 程度（約1140度/秒）にクリップしてノイズスパイクを除去
                    qd_wrist = np.clip(raw_qd_wrist, -20.0, 20.0)
                    qd_grip  = np.clip(raw_qd_grip, -20.0, 20.0)

                self.prev_q_wrist = q_wrist
                self.prev_q_grip = q_grip
                
                q_vec  = torch.tensor([q_wrist, q_grip], device=self.device)
                qd_vec = torch.tensor([qd_wrist, qd_grip], device=self.device)
                obs_action = torch.tensor(self.last_actions, device=self.device)
                
                # ★ 修正: traj はすでにTensorであり、0~1に正規化されている
                phase, traj = self.rhythm_gen.get_state(t_math)
                
                obs_phase = torch.tensor([np.sin(phase), np.cos(phase)], device=self.device)
                obs_bpm = torch.tensor([self.rhythm_gen.bpm / 180.0], device=self.device)
                obs_traj = traj  # 二重で割らない！

                obs = torch.cat([q_vec, qd_vec, obs_action, obs_phase, obs_bpm, obs_traj])
                obs = obs.unsqueeze(0).float()

                # 3. 推論 & 送信
                if args.verify:
                    # Verify表示用に実数値(ニュートン)に戻してバーを描画
                    tgt_force_val = traj[0].item() * args.force_scale
                    bar = "#" * int(tgt_force_val)
                    print(f"\r[Verify] t:{t_math:4.2f} | F:{sensor['force_N']:5.1f} | Tgt:{tgt_force_val:5.1f} {bar:10s}", end="")
                    cmd_pres = [0.0, 0.0, 0.0]
                    action = np.zeros(3)
                else:
                    with torch.no_grad():
                        action = self.policy(obs).cpu().numpy().flatten()
                    
                    self.last_actions = np.clip(action, -1.0, 1.0)
                    p_df = (self.last_actions[0] + 1) / 2 * P_MAX
                    p_f  = (self.last_actions[1] + 1) / 2 * P_MAX
                    p_g  = (self.last_actions[2] + 1) / 2 * P_MAX
                    
                    cmd_pres = [p_df, p_f, p_g]
                    self._send_pressure(*cmd_pres)

                # 4. 指令値ログの保存
                self.cmd_logs.append({
                    'cmd_time': t_math,
                    'target_force': traj[0].item() * args.force_scale, # ★ ログには実数値で保存
                    'action_DF': action[0],
                    'action_F':  action[1],
                    'action_G':  action[2],
                    'cmd_DF': cmd_pres[0],
                    'cmd_F':  cmd_pres[1],
                    'cmd_G':  cmd_pres[2]
                })

                step_idx += 1

                target_next_time = self.start_time + (step_idx * CONTROL_DT)
                while time.perf_counter() < target_next_time:
                    time.sleep(0.0005)

        except KeyboardInterrupt:
            print("\n[Stop] Stopping...")
        finally:
            self._shutdown()

    def _send_pressure(self, df, f, g):
        packet = HEADER + struct.pack(SEND_FMT, df, f, g)
        self.ser.write(packet)

    def _shutdown(self):
        self._send_pressure(0, 0, 0)
        self.receiver.running = False
        self.ser.close()
        self._save_logs()

    def _save_logs(self):
        sensor_data = self.receiver.get_all_logs()
        if not sensor_data:
            print("[Error] No sensor data received for logging.")
            return

        print("\nReconstructing Timestamps and Merging...")
        df_sensor = pd.DataFrame(sensor_data)
        df_sensor['time'] = np.arange(len(df_sensor)) * 0.005

        if self.cmd_logs:
            df_cmd = pd.DataFrame(self.cmd_logs)
            df_cmd = df_cmd.rename(columns={'cmd_time': 'time'})

            df_merged = pd.merge_asof(
                df_sensor,
                df_cmd,
                on='time',
                direction='backward'
            )
        else:
            df_merged = df_sensor

        model_name = os.path.splitext(os.path.basename(args.model))[0] if not args.verify else "verify"
        midi_name = os.path.splitext(os.path.basename(args.midi))[0]
        filename = f"deploy_{midi_name}_{model_name}_{int(time.time())}.csv"
        
        os.makedirs("deploy_results", exist_ok=True)
        path = os.path.join("deploy_results", filename)
        
        df_merged.to_csv(path, index=False)
        print(f"[Log] High-Res Reconstructed Log saved to:\n  -> {path}")

if __name__ == "__main__":
    MidiDeployer().run()