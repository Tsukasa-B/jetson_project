"""
Porcaro Robot: Sim-to-Real RL Policy Deployment (MIDI Version)
Target: Jetson Orin Nano + MicroLabBox
Feature: 
  - Async Communication: Recv @ 200Hz / Control @ 50Hz
  - MIDI Parsing & Lookahead Trajectory Generation
  - PyTorch Policy Inference

Usage:
  python run_deploy_midi.py --midi songs/drum_pattern.mid --bpm 120
  python run_deploy_midi.py --midi songs/pattern.mid --model models/best_policy_v2.pt
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
    """
    MicroLabBoxからの高速データ(200Hz)を取りこぼさず受信し、
    制御ループ(50Hz)に「最新の1フレーム」を提供する
    """
    def __init__(self, ser):
        super().__init__()
        self.ser = ser
        self.running = True
        self.latest_data = None
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
                        # Header Check
                        idx = buffer.find(HEADER)
                        if idx == -1:
                            buffer = buffer[-RECV_PACKET_LEN:]
                            break
                        if idx > 0:
                            buffer = buffer[idx:]
                        if len(buffer) < RECV_PACKET_LEN:
                            break

                        # Extract Packet
                        packet = buffer[:RECV_PACKET_LEN]
                        buffer = buffer[RECV_PACKET_LEN:]
                        self._update(packet[2:]) # Skip Header
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"[Rx Error] {e}")
                self.running = False

    def _update(self, packet_bytes):
        try:
            data = struct.unpack(RECV_FMT, packet_bytes)
            with self.lock:
                self.latest_data = {
                    'meas_pres_DF': data[0],
                    'meas_pres_F':  data[1],
                    'meas_pres_G':  data[2],
                    'angle_deg':    data[3],
                    'velocity':     data[4],
                    'flag':         data[5],
                    'force_N':      data[6]
                }
        except: pass

    def get_latest(self):
        with self.lock:
            return copy.deepcopy(self.latest_data)

# ==========================================
# 2. MIDI リズム生成クラス
# ==========================================
class MidiRhythmGenerator:
    """
    MIDIファイルを読み込み、強化学習エージェント用の状態(Phase, Trajectory)を生成する
    """
    def __init__(self, midi_path, dt, horizon_steps, target_force, override_bpm=None):
        self.dt = dt
        self.horizon_steps = horizon_steps
        self.target_force = target_force
        self.sigma = 0.025  # Gaussian Kernel Width
        
        # MIDI Load
        try:
            mid = mido.MidiFile(midi_path)
            print(f"[MIDI] Loaded: {midi_path}")
        except Exception as e:
            print(f"[Error] Failed to load MIDI: {e}")
            exit(1)

        # BPM解析
        self.bpm = 120.0 # Default
        for msg in mid:
            if msg.type == 'set_tempo':
                self.bpm = mido.tempo2bpm(msg.tempo)
                break
        
        if override_bpm:
            print(f"[MIDI] Overriding BPM: {self.bpm:.1f} -> {override_bpm:.1f}")
            self.bpm = override_bpm

        # Note On イベントの抽出 (絶対時刻 [s] に変換)
        self.spikes = []
        current_time = 0.0
        
        # midoのTick変換係数
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
        # 1. 位相 (Phase)
        # MIDI再生における「小節内の位置」ではなく、エージェントの内部クロックとしての位相
        phase = (t_now * (self.bpm / 60.0) * 2 * np.pi) % (2 * np.pi)

        # 2. Lookahead Trajectory
        traj = np.zeros(self.horizon_steps)
        t_futures = t_now + np.arange(self.horizon_steps) * self.dt
        
        # 検索範囲の絞り込み (高速化)
        search_radius = 0.5 # [s]
        idx_start = np.searchsorted(self.spikes, t_now - 0.1)
        idx_end = np.searchsorted(self.spikes, t_now + search_radius + 0.1)
        
        relevant_spikes = self.spikes[idx_start:idx_end]

        if len(relevant_spikes) > 0:
            # (Steps, 1) - (1, Spikes)
            diffs = t_futures[:, None] - relevant_spikes[None, :]
            
            # Gaussian Kernel
            mask = np.abs(diffs) < (self.sigma * 4)
            weights = np.zeros_like(diffs)
            weights[mask] = np.exp(-0.5 * (diffs[mask] / self.sigma)**2)
            
            # Max Pooling for overlapping notes
            traj = np.max(weights, axis=1) * self.target_force

        return phase, traj

# ==========================================
# 3. メイン制御クラス (MIDI Deployer)
# ==========================================
class MidiDeployer:
    def __init__(self):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Device: {self.device}")

        # Model
        if not args.verify:
            if not os.path.exists(args.model):
                raise FileNotFoundError(f"Model not found: {args.model}")
            self.policy = torch.jit.load(args.model, map_location=self.device)
            self.policy.eval()

        # Serial
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            self.receiver = SensorReceiver(self.ser)
            self.receiver.start()
        except Exception as e:
            print(f"[Error] Serial Failed: {e}")
            exit(1)

        # Rhythm
        self.rhythm_gen = MidiRhythmGenerator(
            args.midi, CONTROL_DT, LOOKAHEAD_STEPS, args.force_scale, args.bpm
        )

        self.logs = []
        self.last_actions = np.zeros(3) # [DF, F, G]
        self.start_time = None

    def run(self):
        print("\n=== MIDI DEPLOYMENT START ===")
        print(f" Song: {args.midi}")
        print(f" Control: 50Hz | Recv: Async")
        
        self._send_pressure(0, 0, 0)
        time.sleep(2.0)
        input(">>> Press ENTER to Start... ")
        
        self.start_time = time.perf_counter()
        
        try:
            while True:
                loop_start = time.perf_counter()
                t_elapsed = loop_start - self.start_time

                # Song Finish Check
                if t_elapsed > self.rhythm_gen.duration_sec:
                    print("Song Finished.")
                    break

                # 1. センサーデータ取得
                sensor = self.receiver.get_latest()
                if sensor is None:
                    time.sleep(0.001)
                    continue

                # 2. 観測ベクトル構築 (Dim=35)
                # [q_wrist, qd_wrist, grip_pos, grip_vel, prev_act(3), phase(2), bpm(1), traj(25)]
                
                # Joint
                q_wrist = np.radians(sensor['angle_deg'])
                qd_wrist = np.radians(sensor['velocity'])
                # Grip is fixed (0.5MPa -> approx pos/vel 0 for observation)
                q_vec = torch.tensor([q_wrist, 0.0], device=self.device)
                qd_vec = torch.tensor([qd_wrist, 0.0], device=self.device)
                
                # Action & Rhythm
                obs_action = torch.tensor(self.last_actions, device=self.device)
                phase, traj = self.rhythm_gen.get_state(t_elapsed)
                
                obs_phase = torch.tensor([np.sin(phase), np.cos(phase)], device=self.device)
                obs_bpm = torch.tensor([self.rhythm_gen.bpm / 180.0], device=self.device)
                obs_traj = torch.tensor(traj / args.force_scale, device=self.device) # Normalize

                # Concatenate
                obs = torch.cat([q_vec, qd_vec, obs_action, obs_phase, obs_bpm, obs_traj])
                obs = obs.unsqueeze(0).float()

                # 3. 推論 & 送信
                if args.verify:
                    bar = "#" * int(traj[0])
                    print(f"\r[Verify] F:{sensor['force_N']:5.1f} | Tgt:{traj[0]:5.1f} {bar:10s}", end="")
                    cmd_pres = [0.0, 0.0, 0.0]
                else:
                    with torch.no_grad():
                        action = self.policy(obs).cpu().numpy().flatten()
                    
                    self.last_actions = np.clip(action, -1.0, 1.0)
                    
                    # Output scaling: (-1,1) -> (0, P_MAX)
                    p_df = (self.last_actions[0] + 1) / 2 * P_MAX
                    p_f  = (self.last_actions[1] + 1) / 2 * P_MAX
                    p_g  = 0.5 # Grip Fixed
                    
                    cmd_pres = [p_df, p_f, p_g]
                    self._send_pressure(*cmd_pres)

                # 4. ログ
                self.logs.append({
                    'time': t_elapsed,
                    'angle': sensor['angle_deg'],
                    'force': sensor['force_N'],
                    'target': traj[0],
                    'cmd_DF': cmd_pres[0],
                    'cmd_F': cmd_pres[1]
                })

                # 5. 50Hz Wait
                while (time.perf_counter() - loop_start) < CONTROL_DT:
                    pass

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
        
        if self.logs:
            df = pd.DataFrame(self.logs)
            name = f"logs_midi_{int(time.time())}.csv"
            df.to_csv(name, index=False)
            print(f"[Log] Saved: {name}")

if __name__ == "__main__":
    MidiDeployer().run()