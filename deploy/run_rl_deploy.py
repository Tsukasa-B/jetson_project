"""
Porcaro Robot: Sim-to-Real RL Policy Deployment (Ver.4)
Target: Jetson Orin Nano + MicroLabBox
Feature: 
  - 複雑なリズムパターン(Rudiments)のリアルタイム生成
  - 実行時のパターン/BPM指定 (--pattern, --bpm)
  - 完全な観測空間の再現

Usage Examples:
  python run_deploy_v4.py --pattern single_4 --bpm 60
  python run_deploy_v4.py --pattern double --bpm 120
  python run_deploy_v4.py --pattern paradiddle --bpm 90
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
import math

# ==========================================
# 0. 引数解析 (Runtime Arguments)
# ==========================================
parser = argparse.ArgumentParser(description="Porcaro Real-time Deployment")
parser.add_argument("--pattern", type=str, default="single_4", 
                    choices=["single_4", "single_8", "double", "paradiddle", "upbeat", "clave", "rest"],
                    help="Rhythm pattern to generate")
parser.add_argument("--bpm", type=float, default=60.0, help="Target BPM")
parser.add_argument("--model", type=str, default="models/policy.pt", help="Path to policy model")
parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port")
parser.add_argument("--verify", action="store_true", help="Run in verification mode (no actuation)")
args = parser.parse_args()

# ==========================================
# ユーザー設定 (Configuration)
# ==========================================
MODEL_PATH = args.model
VERIFY_MODE = args.verify
USE_FIXED_GRIP = True

# 制御・通信設定
SERIAL_PORT = args.port
BAUD_RATE = 230400  # MicroLabBox設定に合わせる
P_MAX = 0.6         # [MPa]
CONTROL_DT = 0.02   # 50Hz (学習環境の dt_step に合わせる)

# リズム生成設定
TARGET_FORCE = 20.0     # [N]
LOOKAHEAD_TIME = 0.5    # [s]
LOOKAHEAD_STEPS = int(LOOKAHEAD_TIME / CONTROL_DT) # 25 steps

# 通信プロトコル
SEND_FMT = '>ddd'        # [Cmd_DF, Cmd_F, Cmd_G]
RECV_FMT = '>ddddddd'    # [P_DF, P_F, P_G, Angle, Vel, Flag, Force]
HEADER = b'\xff\xff'
RECV_PACKET_LEN = 2 + 8 * 7

# ==========================================
# 1. センサーインターフェース (非同期受信)
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
                if self.running: print(f"[Sensor Error] {e}")
                self.running = False

    def get_latest(self):
        with self.lock:
            return self.latest_data

# ==========================================
# 2. リアルタイム・ルーディメンツ生成器
# ==========================================
class RealTimeRudimentGenerator:
    """
    rhythm_generator.py のロジックをNumPyでリアルタイム用に移植したクラス
    """
    def __init__(self, mode="single_4", bpm=60.0, dt=0.02, horizon_steps=25, target_force=20.0):
        self.mode = mode
        self.bpm = bpm
        self.dt = dt
        self.horizon_steps = horizon_steps
        self.target_force = target_force
        
        # --- ルーディメンツ定義 (16分音符グリッド: 0~15) ---
        # 1小節(4拍) = 16個の16分音符スロット
        self.rudiments = {
            "single_4":  [0, 4, 8, 12],            # 4分音符
            "single_8":  [0, 2, 4, 6, 8, 10, 12, 14], # 8分音符
            "double":    [0, 1, 4, 5, 8, 9, 12, 13],  # RRLL...
            "paradiddle":[0, 2, 4, 5, 8, 10, 12, 13], # RLRR LRLL
            "upbeat":    [2, 6, 10, 14],           # 裏拍
            "clave":     [0, 3, 6, 8, 10, 12],     # 3-3-2
            "rest":      []
        }
        
        # 現在のパターンのオフセットリストを取得
        self.current_offsets = self.rudiments.get(mode, self.rudiments["single_4"])
        print(f"[Rhythm] Initialized Mode: {mode}, BPM: {bpm}")
        print(f"[Rhythm] Grid Offsets: {self.current_offsets}")

        # ガウスカーネル設定
        width_sec = 0.05
        self.sigma = width_sec / 2.0

    def get_state(self, t_now):
        """
        現在時刻 t_now における位相と、未来のターゲット軌道を計算
        """
        # -------------------------------------------------
        # A. 位相計算 (Phase)
        # -------------------------------------------------
        # Sim: phase = time * (bpm/60) * (2pi)
        # 1拍ごとの位相
        total_beats = t_now * (self.bpm / 60.0)
        phase_rad = total_beats * (2 * np.pi)

        # -------------------------------------------------
        # B. ターゲット軌道生成 (Trajectory)
        # -------------------------------------------------
        trajectory = np.zeros(self.horizon_steps)
        future_times = t_now + np.arange(self.horizon_steps) * self.dt

        # 計算範囲内の「小節数(bar_idx)」と「グリッド位置」を特定してスパイクを置く
        
        # 1小節(4拍)の時間 [s]
        bar_duration = 4.0 * (60.0 / self.bpm)
        
        # 検索範囲の開始・終了時刻
        t_start = future_times[0]
        t_end = future_times[-1]

        # 検索対象となる小節インデックスの範囲
        bar_idx_start = int(t_start / bar_duration)
        bar_idx_end = int(t_end / bar_duration) + 1

        # 16分音符1個あたりの時間 [s]
        grid_duration = (60.0 / self.bpm) / 4.0

        for b_idx in range(bar_idx_start, bar_idx_end + 1):
            # この小節の開始時刻
            bar_start_time = b_idx * bar_duration
            
            # パターン内の有効なグリッド(打撃位置)についてループ
            for grid_offset in self.current_offsets:
                # 打撃予定時刻
                t_spike = bar_start_time + grid_offset * grid_duration
                
                # 未来の時刻配列との差分
                dt_vec = future_times - t_spike
                
                # カーネルの範囲内(±0.1s)にある点だけ計算
                mask = np.abs(dt_vec) < 0.1
                if np.any(mask):
                    val = self.target_force * np.exp(-0.5 * (dt_vec[mask] / self.sigma) ** 2)
                    # 重ね合わせ (MaxをとるかAddするか。SimはConv1dなのでAddに近いが、スパイクが離れていればMaxでも同じ)
                    # Simの実装(Conv1d)は加算(Add)的な挙動。
                    # しかし重なりが少ない前提なら np.maximum の方が波形が崩れにくい。ここではMaximum採用。
                    trajectory[mask] = np.maximum(trajectory[mask], val)

        return phase_rad, trajectory / self.target_force


# ==========================================
# 3. メイン制御クラス
# ==========================================
class RLDeployer:
    def __init__(self):
        # A. モデルロード
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Device: {self.device}")
        
        if not VERIFY_MODE:
            if not os.path.exists(MODEL_PATH):
                print(f"[Error] Model file not found: {MODEL_PATH}")
                exit(1)
            try:
                self.policy = torch.jit.load(MODEL_PATH, map_location=self.device)
                self.policy.eval()
                print(f"[Init] Policy loaded: {MODEL_PATH}")
            except Exception as e:
                print(f"[Error] Failed to load model. {e}")
                exit(1)
        
        # B. 通信接続
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            self.sensor = SensorInterface(self.ser)
            self.sensor.start()
            print(f"[Init] Serial connected: {SERIAL_PORT}")
        except Exception as e:
            print(f"[Error] Serial connection failed: {e}")
            exit(1)

        # C. リズム生成器 (引数で指定されたパターンを使用)
        self.rhythm_gen = RealTimeRudimentGenerator(
            mode=args.pattern, 
            bpm=args.bpm, 
            dt=CONTROL_DT, 
            horizon_steps=LOOKAHEAD_STEPS,
            target_force=TARGET_FORCE
        )

        self.last_actions = np.zeros(3, dtype=np.float32) # [DF, F, G]
        self.start_time = None
        self.logs = []

    def run(self):
        print("\n" + "="*60)
        print(f"  Porcaro Deployment (Ver.4)")
        print(f"  - Pattern: {args.pattern}")
        print(f"  - BPM:     {args.bpm}")
        print(f"  - Verify:  {VERIFY_MODE}")
        print(f"  - Port:    {SERIAL_PORT}")
        print("="*60 + "\n")
        
        input("Press Enter to START control...")
        self.start_time = time.perf_counter()

        try:
            while True:
                loop_start = time.perf_counter()
                t_elapsed = loop_start - self.start_time

                # 1. センサーデータ取得
                raw_data = self.sensor.get_latest()
                if raw_data is None:
                    continue 
                
                # raw_data: [P_DF, P_F, P_G, Angle, Vel, Flag, Force]
                raw_angle_deg = raw_data[3]
                raw_vel_deg = raw_data[4]

                # 2. 座標系 & 単位変換 (Sim環境に合わせる: rad, Up=Positive)
                obs_wrist_pos = np.radians(raw_angle_deg)
                obs_wrist_vel = np.radians(raw_vel_deg)

                if USE_FIXED_GRIP:
                    obs_grip_pos = 0.0 
                    obs_grip_vel = 0.0
                
                # 3. リズム・位相観測 (ここで選択したパターンの波形が出る)
                phase_rad, rhythm_buf = self.rhythm_gen.get_state(t_elapsed)
                
                sin_phase = np.sin(phase_rad)
                cos_phase = np.cos(phase_rad)
                
                # Tensor化
                q = torch.tensor([obs_wrist_pos, obs_grip_pos], device=self.device)
                qd = torch.tensor([obs_wrist_vel, obs_grip_vel], device=self.device)
                prev_actions = torch.tensor(self.last_actions, device=self.device)
                phase_feats = torch.tensor([sin_phase, cos_phase], device=self.device)
                bpm_feat = torch.tensor([self.rhythm_gen.bpm / 180.0], device=self.device)
                rhythm_feat = torch.tensor(rhythm_buf, dtype=torch.float32, device=self.device)

                # 4. Observation 結合
                obs_tensor = torch.cat([
                    q, qd, prev_actions, phase_feats, bpm_feat, rhythm_feat
                ]).unsqueeze(0)

                # 5. 推論 & 制御
                if VERIFY_MODE:
                    # 検証モード: ターゲット波形のみ表示
                    tgt_val = rhythm_buf[0] * TARGET_FORCE
                    # 簡易バー表示
                    bar_len = int(tgt_val)
                    bar_str = "#" * bar_len
                    print(f"\r[Verify] Tgt: {tgt_val:5.1f}N | {bar_str:20s} | Ang: {raw_angle_deg:5.1f}", end="")
                    
                    pressures = [0.0, 0.0, 0.0]
                else:
                    # 制御モード
                    with torch.no_grad():
                        actions = self.policy(obs_tensor).cpu().numpy().flatten()

                    self.last_actions = np.clip(actions, -1.0, 1.0)
                    pressures = (self.last_actions + 1.0) / 2.0 * P_MAX
                    
                    packet = HEADER + struct.pack(SEND_FMT, *pressures)
                    self.ser.write(packet)
                    self.ser.flush()

                # 6. ログ記録
                self.logs.append({
                    'time': t_elapsed,
                    'pattern': args.pattern,
                    'bpm': args.bpm,
                    'obs_wrist': obs_wrist_pos,
                    'cmd_DF': self.last_actions[0],
                    'real_P_DF': pressures[0],
                    'target_val': rhythm_buf[0] * TARGET_FORCE
                })

                # 7. 周期維持 (50Hz)
                dt = time.perf_counter() - loop_start
                if dt < CONTROL_DT:
                    time.sleep(CONTROL_DT - dt)

        except KeyboardInterrupt:
            print("\n[Stop] Stopping control...")
        finally:
            self.shutdown()

    def shutdown(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            print("Sending Zero Pressure...")
            for _ in range(5):
                self.ser.write(HEADER + struct.pack(SEND_FMT, 0, 0, 0))
                time.sleep(0.01)
            self.ser.close()
        
        if self.logs:
            df = pd.DataFrame(self.logs)
            os.makedirs("data_logs", exist_ok=True)
            fname = f"data_logs/deploy_log_{args.pattern}_{int(time.time())}.csv"
            df.to_csv(fname, index=False)
            print(f"[Log] Saved: {fname}")

if __name__ == "__main__":
    deployer = RLDeployer()
    deployer.run()