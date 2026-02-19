"""
Porcaro Robot: Sim-to-Real RL Policy Deployment (Ver.5 - AsyncIO)
Target: Jetson Orin Nano + MicroLabBox
Feature: 
  - Async Communication: Recv @ 200Hz / Control @ 50Hz
  - Real-time Rudiment Generation with Lookahead
  - PyTorch Policy Inference

Usage:
  python run_deploy.py --pattern single_4 --bpm 100 --port /dev/ttyUSB0
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

# ==========================================
# 0. 引数解析 & 定数定義
# ==========================================
parser = argparse.ArgumentParser(description="Porcaro Real-time Deployment")
parser.add_argument("--pattern", type=str, default="single_4", 
                    choices=["single_4", "single_8", "double", "paradiddle", "upbeat", "clave", "rest"],
                    help="Rhythm pattern")
parser.add_argument("--bpm", type=float, default=60.0, help="Target BPM")
parser.add_argument("--model", type=str, default="models/policy.pt", help="Path to policy.pt")
parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port")
parser.add_argument("--verify", action="store_true", help="Verification mode (No actuation)")
args = parser.parse_args()

# System Config
SERIAL_PORT = args.port
BAUD_RATE = 230400      # MicroLabBox (200Hz Send) Setting
CONTROL_DT = 0.02       # 50Hz Control Loop
P_MAX = 0.6             # Max Pressure [MPa]

# Rhythm Config
TARGET_FORCE = 20.0     # [N]
LOOKAHEAD_TIME = 0.5    # [s]
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
                # 制御に必要な最新値だけ保持
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
# 2. リアルタイム・リズム生成器
# ==========================================
class RealTimeRudimentGenerator:
    """
    現在時刻に基づき、将来25ステップ分のターゲット軌道を生成する
    """
    def __init__(self, mode, bpm, dt, steps, target_force):
        self.bpm = bpm
        self.dt = dt
        self.steps = steps
        self.target_force = target_force
        self.sigma = 0.025 # Kernel Width

        # パターン定義 (16分音符グリッド 0-15)
        patterns = {
            "single_4":   [0, 4, 8, 12],
            "single_8":   [0, 2, 4, 6, 8, 10, 12, 14],
            "double":     [0, 1, 4, 5, 8, 9, 12, 13],
            "paradiddle": [0, 2, 4, 5, 8, 10, 12, 13],
            "upbeat":     [2, 6, 10, 14],
            "clave":      [0, 3, 6, 8, 10, 12],
            "rest":       []
        }
        self.grid = patterns.get(mode, patterns["single_4"])
        
        # 16分音符1つあたりの秒数
        self.tick_duration = (60.0 / bpm) / 4.0
        # 1小節(16ticks)の秒数
        self.bar_duration = self.tick_duration * 16.0

    def get_state(self, t_now):
        # 1. 位相 (Phase) [0, 2pi] - 4分音符周期のサイン波用
        # beat_duration = 60 / bpm
        phase = (t_now * (self.bpm / 60.0) * 2 * np.pi) % (2 * np.pi)

        # 2. Lookahead Trajectory (25 steps)
        traj = np.zeros(self.steps)
        
        # 未来時刻の配列
        t_futures = t_now + np.arange(self.steps) * self.dt
        
        # 各未来時刻が、どの「グリッド(打撃点)」に近いかを計算
        # 高速化のため、t_futures全体に対してベクトル演算
        
        # 現在の小節番号と、次の小節番号までを考慮
        current_bar_idx = int(t_now / self.bar_duration)
        
        # 検索するスパイクの絶対時刻リストを作成
        candidate_spikes = []
        for b in [current_bar_idx, current_bar_idx + 1]:
            base_t = b * self.bar_duration
            for g in self.grid:
                candidate_spikes.append(base_t + g * self.tick_duration)
        
        candidate_spikes = np.array(candidate_spikes)

        # 各未来ステップについて、最も近いスパイクの影響を計算
        if len(candidate_spikes) > 0:
            # (Steps, 1) - (1, Spikes) = (Steps, Spikes)
            diffs = t_futures[:, None] - candidate_spikes[None, :]
            
            # ガウスカーネル: exp(-0.5 * (d/sigma)^2)
            # 近いものだけ計算 (マスク処理)
            mask = np.abs(diffs) < (self.sigma * 4)
            weights = np.zeros_like(diffs)
            weights[mask] = np.exp(-0.5 * (diffs[mask] / self.sigma)**2)
            
            # 各ステップごとに最大の重みを採用 (Max Pooling的な合成)
            traj = np.max(weights, axis=1) * self.target_force

        return phase, traj

# ==========================================
# 3. メイン制御クラス
# ==========================================
class RLDeployer:
    def __init__(self):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Device: {self.device}")

        # Model Load
        if not args.verify:
            if not os.path.exists(args.model):
                raise FileNotFoundError(f"Model not found: {args.model}")
            self.policy = torch.jit.load(args.model, map_location=self.device)
            self.policy.eval()
            print(f"[Init] Policy Loaded: {args.model}")

        # Serial
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            self.receiver = SensorReceiver(self.ser)
            self.receiver.start()
            print(f"[Init] Serial Opened: {SERIAL_PORT} @ {BAUD_RATE}")
        except Exception as e:
            print(f"[Error] Serial Failed: {e}")
            exit(1)

        # Rhythm
        self.rhythm = RealTimeRudimentGenerator(
            args.pattern, args.bpm, CONTROL_DT, LOOKAHEAD_STEPS, TARGET_FORCE
        )

        self.logs = []
        self.last_actions = np.zeros(3) # [DF, F, G]
        self.start_time = None

    def run(self):
        print("\n=== PORCARO DEPLOYMENT START ===")
        print(f" Pattern: {args.pattern} | BPM: {args.bpm}")
        print(f" Control: 50Hz | Recv: Async")
        
        # 安全初期化
        self._send_pressure(0, 0, 0)
        time.sleep(2.0)
        input(">>> Press ENTER to Start... ")
        
        self.start_time = time.perf_counter()
        
        try:
            while True:
                loop_start = time.perf_counter()
                t_elapsed = loop_start - self.start_time

                # 1. 最新データ取得
                sensor = self.receiver.get_latest()
                if sensor is None:
                    # データがまだ来てないときは待つ
                    time.sleep(0.001)
                    continue

                # 2. 観測ベクトル構築 (Dim=35)
                # [Joint(4), Action(3), Phase(2), BPM(1), Traj(25)]
                
                # Joint: [WristPos, WristVel, GripPos, GripVel]
                # Simはrad単位。Realはdegで来るので変換。
                # Gripは固定(0.5MPa)なので、角度・速度は0と仮定するか、センサあれば入れる
                q_wrist = np.radians(sensor['angle_deg'])
                qd_wrist = np.radians(sensor['velocity'])
                obs_joint = torch.tensor([q_wrist, qd_wrist, 0.0, 0.0], device=self.device)

                # Prev Action
                obs_action = torch.tensor(self.last_actions, device=self.device)

                # Rhythm State
                phase, traj = self.rhythm.get_state(t_elapsed)
                obs_phase = torch.tensor([np.sin(phase), np.cos(phase)], device=self.device)
                obs_bpm = torch.tensor([args.bpm / 180.0], device=self.device) # Normalize
                obs_traj = torch.tensor(traj / TARGET_FORCE, device=self.device) # Normalize

                # Concatenate
                obs = torch.cat([obs_joint, obs_joint, obs_action, obs_phase, obs_bpm, obs_traj])
                # Note: obs_jointを2回繰り返しているのは q, qd の代わりか？ 
                # -> いえ、q(2) + qd(2) です。上で obs_joint にまとめてしまったので修正します。
                
                # 正しい構成: q(2), qd(2), a(3), phase(2), bpm(1), traj(25) = 35
                q_vec = torch.tensor([q_wrist, 0.0], device=self.device)
                qd_vec = torch.tensor([qd_wrist, 0.0], device=self.device)
                
                obs = torch.cat([q_vec, qd_vec, obs_action, obs_phase, obs_bpm, obs_traj])
                obs = obs.unsqueeze(0).float() # Add Batch Dim

                # 3. 推論 & 送信
                if args.verify:
                    # Verifyモード: ターゲット波形を表示するだけ
                    bar = "#" * int(traj[0])
                    print(f"\r[Verify] F:{sensor['force_N']:5.1f} | Tgt:{traj[0]:5.1f} {bar:10s}", end="")
                    cmd_pres = [0.0, 0.0, 0.0]
                else:
                    with torch.no_grad():
                        action = self.policy(obs).cpu().numpy().flatten()
                    
                    self.last_actions = np.clip(action, -1.0, 1.0)
                    # Sim Action (-1~1) -> Real Pressure (0~0.6MPa)
                    # Grip(Index 2) は固定0.5MPaに上書き (安全のため)
                    p_df = (self.last_actions[0] + 1) / 2 * P_MAX
                    p_f  = (self.last_actions[1] + 1) / 2 * P_MAX
                    p_g  = 0.5 
                    
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
        # Safety Clip
        df = max(0.0, min(0.6, df))
        f  = max(0.0, min(0.6, f))
        g  = max(0.0, min(0.6, g))
        packet = HEADER + struct.pack(SEND_FMT, df, f, g)
        self.ser.write(packet)

    def _shutdown(self):
        self._send_pressure(0, 0, 0)
        self.receiver.running = False
        self.ser.close()
        
        # Save Log
        if self.logs:
            df = pd.DataFrame(self.logs)
            name = f"deploy_{args.pattern}_{int(time.time())}.csv"
            df.to_csv(name, index=False)
            print(f"[Log] Saved to {name}")

if __name__ == "__main__":
    deployer = RLDeployer()
    deployer.run()