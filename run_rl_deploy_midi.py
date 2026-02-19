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
  python run_rl_deploy_midi.py --midi songs/test_single4_bpm60.mid --model models/modelB_DR_2999_02-17.pt
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
import mido  # pip install mido

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
BAUD_RATE = 230400      # MicroLabBox (200Hz Send)
CONTROL_DT = 0.02       # 50Hz Control Loop
P_MAX = 0.6             # Max Pressure [MPa]

# Lookahead Config
LOOKAHEAD_STEPS = 25    # 0.5s / 0.02s

# Communication Protocol
SEND_FMT = '>ddd'       # [Cmd_DF, Cmd_F, Cmd_G]
RECV_FMT = '>ddddddd'   # [P_DF, P_F, P_G, Angle, Vel, Flag, Force]
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
        self.latest_data = None
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
                        self._update(packet[2:])
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"[Rx Error] {e}")
                self.running = False

    def _update(self, packet_bytes):
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
                self.latest_data = sample  # RL推論用に常に最新を保持
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
# 2. MIDI リズム生成クラス
# ==========================================
class MidiRhythmGenerator:
    def __init__(self, midi_path, dt, horizon_steps, target_force, override_bpm=None):
        self.dt = dt
        self.horizon_steps = horizon_steps
        self.target_force = target_force
        self.sigma = 0.025  
        
        try:
            mid = mido.MidiFile(midi_path)
            print(f"[MIDI] Loaded: {midi_path}")
        except Exception as e:
            print(f"[Error] Failed to load MIDI: {e}")
            exit(1)

        self.bpm = 120.0 
        for msg in mid:
            if msg.type == 'set_tempo':
                self.bpm = mido.tempo2bpm(msg.tempo)
                break
        
        if override_bpm:
            print(f"[MIDI] Overriding BPM: {self.bpm:.1f} -> {override_bpm:.1f}")
            self.bpm = override_bpm

        self.spikes = []
        ticks_per_beat = mid.ticks_per_beat
        sec_per_tick = (60.0 / self.bpm) / ticks_per_beat

        for track in mid.tracks:
            track_time = 0.0
            for msg in track:
                track_time += msg.time * sec_per_tick
                if msg.type == 'note_on' and msg.velocity > 0:
                    self.spikes.append(track_time)
        
        self.spikes = np.array(sorted(self.spikes))
        self.duration_sec = self.spikes[-1] + 2.0 if len(self.spikes) > 0 else 10.0
        print(f"[MIDI] Total Notes: {len(self.spikes)} | Duration: {self.duration_sec:.1f}s | BPM: {self.bpm:.1f}")

    def get_state(self, t_now):
        phase = (t_now * (self.bpm / 60.0) * 2 * np.pi) % (2 * np.pi)
        traj = np.zeros(self.horizon_steps)
        t_futures = t_now + np.arange(self.horizon_steps) * self.dt
        
        search_radius = 0.5 
        idx_start = np.searchsorted(self.spikes, t_now - 0.1)
        idx_end = np.searchsorted(self.spikes, t_now + search_radius + 0.1)
        
        relevant_spikes = self.spikes[idx_start:idx_end]

        if len(relevant_spikes) > 0:
            diffs = t_futures[:, None] - relevant_spikes[None, :]
            mask = np.abs(diffs) < (self.sigma * 4)
            weights = np.zeros_like(diffs)
            weights[mask] = np.exp(-0.5 * (diffs[mask] / self.sigma)**2)
            traj = np.max(weights, axis=1) * self.target_force

        return phase, traj

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

        self.rhythm_gen = MidiRhythmGenerator(
            args.midi, CONTROL_DT, LOOKAHEAD_STEPS, args.force_scale, args.bpm
        )

        self.cmd_logs = []
        self.last_actions = np.zeros(3) 
        self.start_time = None

    def run(self):
        print("\n=== MIDI DEPLOYMENT START ===")
        print(f" Song: {args.midi}")
        print(f" Control: 50Hz (Absolute Sync) | Recv: Async 200Hz")
        
        self._send_pressure(0, 0, 0)
        time.sleep(2.0)
        input(">>> Press ENTER to Start... ")
        
        # 変更箇所: 時刻0の厳密な同期
        self.receiver.clear_buffer_for_sync()
        self.start_time = time.perf_counter()
        step_idx = 0
        
        try:
            while True:
                # 変更箇所: 数学的な理想時刻 (t_math) の算出
                # USBジッタや推論の処理落ちに影響されない完璧なメトロノーム時間
                t_math = step_idx * CONTROL_DT

                if t_math > self.rhythm_gen.duration_sec:
                    print("\nSong Finished.")
                    break

                # 1. 最新のセンサーデータ取得 (RL用)
                sensor = self.receiver.get_latest()
                if sensor is None:
                    # まだデータが来ていない場合は微小待機してスキップ
                    time.sleep(0.001)
                    continue

                # 2. 観測ベクトル構築 (Dim=35)
                q_wrist = np.radians(sensor['angle_deg'])
                qd_wrist = np.radians(sensor['velocity'])
                q_vec = torch.tensor([q_wrist, 0.0], device=self.device)
                qd_vec = torch.tensor([qd_wrist, 0.0], device=self.device)
                
                obs_action = torch.tensor(self.last_actions, device=self.device)
                
                # 変更箇所: t_elapsed(実時間)ではなくt_math(理想時間)でリズム生成
                # これにより推論遅れなどでBPMがヨレるのを防ぐ
                phase, traj = self.rhythm_gen.get_state(t_math)
                
                obs_phase = torch.tensor([np.sin(phase), np.cos(phase)], device=self.device)
                obs_bpm = torch.tensor([self.rhythm_gen.bpm / 180.0], device=self.device)
                obs_traj = torch.tensor(traj / args.force_scale, device=self.device)

                obs = torch.cat([q_vec, qd_vec, obs_action, obs_phase, obs_bpm, obs_traj])
                obs = obs.unsqueeze(0).float()

                # 3. 推論 & 送信
                if args.verify:
                    bar = "#" * int(traj[0])
                    print(f"\r[Verify] t:{t_math:4.2f} | F:{sensor['force_N']:5.1f} | Tgt:{traj[0]:5.1f} {bar:10s}", end="")
                    cmd_pres = [0.0, 0.0, 0.0]
                    action = np.zeros(3)
                else:
                    with torch.no_grad():
                        action = self.policy(obs).cpu().numpy().flatten()
                    
                    self.last_actions = np.clip(action, -1.0, 1.0)
                    p_df = (self.last_actions[0] + 1) / 2 * P_MAX
                    p_f  = (self.last_actions[1] + 1) / 2 * P_MAX
                    p_g  = 0.5 
                    
                    cmd_pres = [p_df, p_f, p_g]
                    self._send_pressure(*cmd_pres)

                # 4. 指令値ログの保存
                self.cmd_logs.append({
                    'cmd_time': t_math,
                    'target_force': traj[0],
                    'action_DF': action[0],
                    'action_F':  action[1],
                    'cmd_DF': cmd_pres[0],
                    'cmd_F':  cmd_pres[1],
                    'cmd_G':  cmd_pres[2]
                })

                step_idx += 1

                # 5. 絶対時刻ベースの待機 (Drift防止)
                # 次のステップの理想開始時刻までスリープする
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
        # 1. 200Hzセンサーデータの再構築
        df_sensor = pd.DataFrame(sensor_data)
        df_sensor['time'] = np.arange(len(df_sensor)) * 0.005

        # 2. 50Hz指令値・推論データのマージ
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

        # ファイル出力
        model_name = os.path.splitext(os.path.basename(args.model))[0] if not args.verify else "verify"
        midi_name = os.path.splitext(os.path.basename(args.midi))[0]
        filename = f"deploy_{midi_name}_{model_name}_{int(time.time())}.csv"
        
        # 保存先ディレクトリの作成
        os.makedirs("deploy_results", exist_ok=True)
        path = os.path.join("deploy_results", filename)
        
        df_merged.to_csv(path, index=False)
        print(f"[Log] High-Res Reconstructed Log saved to:\n  -> {path}")

if __name__ == "__main__":
    MidiDeployer().run()