"""
Porcaro Robot: Sim-to-Real RL Policy Deployment (Ver.2)
Target: Jetson Orin Nano + MicroLabBox
Feature: 座標系整合性確保 + リズム生成 + 状態検証モード

Usage:
  1. MODEL_PATH を実際のパスに変更
  2. VERIFY_MODE = True でセンサー値と座標系を確認
  3. VERIFY_MODE = False で制御実行
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

# ==========================================
# ユーザー設定 (Configuration)
# ==========================================
# 学習済みモデルのパス
MODEL_PATH = "models/policy_single_bpm60.pt" 

# 動作モード設定
VERIFY_MODE = True    # True: 推論・制御せず、センサー値とObs変換の確認のみ行う
USE_FIXED_GRIP = True  # グリップのセンサーがない場合、固定値を使う

# 制御・通信設定
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 230400
CONTROL_DT = 0.01       # 10ms (100Hz)
P_MAX = 0.6             # [MPa]

# リズム生成設定 (学習条件と一致させる: Single Stroke, BPM60)
RHYTHM_MODE = "single"  # "single", "double"
TARGET_BPM = 60.0
TARGET_FORCE = 20.0     # [N] Obs正規化用
LOOKAHEAD_TIME = 0.5    # [s] 先読み時間
LOOKAHEAD_STEPS = 50    # 0.5s / 0.01s (Decimation=1と仮定。Sim設定に合わせて調整推奨)
# ※Simで dt=0.005, decimation=4 (ctrl=0.02) の場合、0.5sは 25 steps です。
#   学習時の dt_ctrl に合わせてここを変更してください。
#   本スクリプトは 100Hz (0.01s) 駆動なので、もし学習が 50Hz (0.02s) なら
#   リズム生成の dt を合わせるか、間引きが必要です。
#   ここでは「学習が 50Hz (dt=0.02)」と仮定し、2回に1回推論するか、
#   あるいは100Hzで推論するか方針を決める必要があります。
#   -> 安全のため「100Hzで推論」しますが、学習時dt=0.02ならリズム生成dtも0.02相当に合わせます。

# 通信プロトコル (run_iros_validation.py 準拠)
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
# 2. リズム生成器 (NumPy軽量版)
# ==========================================
class RealTimeRhythmGenerator:
    def __init__(self, mode="single", bpm=60.0, dt=0.01, horizon_steps=25, target_force=20.0):
        self.mode = mode
        self.bpm = bpm
        self.dt = dt
        self.horizon_steps = horizon_steps
        self.target_force = target_force
        
        # ガウスカーネルの作成
        width_sec = 0.05
        sigma = width_sec / 2.0
        radius = int(width_sec / dt)
        t_kern = np.arange(-radius, radius + 1) * dt
        self.kernel = target_force * np.exp(-0.5 * (t_kern / sigma) ** 2)
        
        # タイムライン管理
        self.t_current = 0.0

    def get_lookahead(self, t_now):
        """現在時刻 t_now から未来 horizon_steps 分のターゲット力を生成して返す"""
        # 単純化のため、必要な区間だけオンデマンド生成（あるいは循環バッファ）
        # ここでは解析的に計算します（スパイク位置との距離）
        
        future_times = t_now + np.arange(self.horizon_steps) * self.dt
        trajectory = np.zeros(self.horizon_steps)
        
        # パターン定義
        if self.mode == "single":
            # 2秒周期、0.5秒オフセット (SimpleRhythmGenerator準拠)
            cycle = 2.0
            offset = 0.5
            
            # 範囲内のすべての拍を検索
            # t_now <= k*cycle + offset <= t_now + horizon*dt
            k_start = int((future_times[0] - offset) / cycle)
            k_end = int((future_times[-1] - offset) / cycle) + 2
            
            for k in range(k_start, k_end):
                t_spike = k * cycle + offset
                # ガウスカーネルの適用
                # target += kernel(t - t_spike)
                dt_vec = future_times - t_spike
                
                # カーネル範囲内の点のみ計算（高速化）
                mask = np.abs(dt_vec) < 0.05 # 幅0.1s
                if np.any(mask):
                    trajectory[mask] += self.target_force * np.exp(-0.5 * (dt_vec[mask] / (0.025)) ** 2)
                    
        # double など他のモードも同様に実装可能
        
        # 正規化して返す (Obsは 0~1)
        return trajectory / self.target_force

# ==========================================
# 3. メイン制御クラス
# ==========================================
class RLDeployer:
    def __init__(self):
        # A. モデルロード
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] Device: {self.device}")
        
        if not VERIFY_MODE:
            try:
                self.policy = torch.jit.load(MODEL_PATH, map_location=self.device)
                self.policy.eval()
                print(f"[Init] Policy loaded: {MODEL_PATH}")
            except Exception as e:
                print(f"[Error] Failed to load model. Check MODEL_PATH. {e}")
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

        # C. リズム生成器
        # ★重要: 学習時の dt (例: 0.02s) と合わせる必要があります。
        # ここでは「制御周期 0.01s」で動かしますが、Obs生成は学習環境の間引き(Decimation)を考慮してください。
        # もし学習が dt=0.02s (Sim 0.005 * Dec 4) なら、0.02s刻みのデータが必要です。
        # ここでは簡易的に 0.01s で生成しますが、本来は合わせるべきです。
        self.rhythm_gen = RealTimeRhythmGenerator(
            mode=RHYTHM_MODE, bpm=TARGET_BPM, 
            dt=0.01, # Obsの時間分解能 (学習環境に合わせる)
            horizon_steps=50, # 0.5s / 0.01s = 50 steps
            target_force=TARGET_FORCE
        )

        self.last_actions = np.zeros(3, dtype=np.float32)
        self.start_time = None
        self.logs = []

    def run(self):
        print("\n" + "="*50)
        if VERIFY_MODE:
            print("  [VERIFY MODE] No actions will be sent.")
            print("  Please move the robot manually to check coordinates.")
        else:
            print("  [CONTROL MODE] AI Agent is Active!")
        print("="*50 + "\n")
        
        input("Press Enter to start...")
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
                raw_vel = raw_data[4]

                # 2. 座標系変換 (Sim-to-Real 整合性確保)
                # Sim: Down=Positive, but Policy trained on Corrected (Up=Positive).
                # Real Sensor: Assume Up=Positive (or whatever 'angle_deg' gives).
                # ★ポリシー入力は [rad]
                
                # --- [修正] ユーザー指示: "エージェントは実機座標系(Up+)で学習済み" ---
                # したがって、Simで行っていた -1.0 倍は不要（実機センサーがすでにUp+なら）。
                # そのまま deg -> rad 変換のみ行う。
                obs_wrist_pos = np.radians(raw_angle_deg)
                obs_wrist_vel = raw_vel  # 必要に応じて単位変換 (deg/s -> rad/s 等)

                # グリップ (センサーがない場合は固定値)
                if USE_FIXED_GRIP:
                    obs_grip_pos = 0.0 # 学習時の初期値など適切な値に
                    obs_grip_vel = 0.0
                else:
                    # プロトコルに含まれていれば取得
                    obs_grip_pos = 0.0 
                    obs_grip_vel = 0.0

                # 3. リズム観測 (Lookahead)
                # 現在時刻に基づいて未来のターゲットを取得
                rhythm_buf = self.rhythm_gen.get_lookahead(t_elapsed)
                rhythm_buf_tensor = torch.tensor(rhythm_buf, dtype=torch.float32, device=self.device)
                
                # BPM Obs
                obs_bpm = torch.tensor([self.rhythm_gen.bpm / 180.0], dtype=torch.float32, device=self.device)

                # 4. Observation 構築
                # 順序: [q_wrist, q_grip, qd_wrist, qd_grip, bpm, rhythm_buf...]
                # ※ porcaro_rl_env.py の _get_observations と完全一致させる
                obs_list = [
                    obs_wrist_pos, obs_grip_pos,
                    obs_wrist_vel, obs_grip_vel
                ]
                obs_base = torch.tensor(obs_list, dtype=torch.float32, device=self.device)
                
                obs_tensor = torch.cat([obs_base, obs_bpm, rhythm_buf_tensor]).unsqueeze(0) # [1, Dim]

                # 5. 検証用出力 (VERIFY_MODE)
                if VERIFY_MODE:
                    print(f"\r[Verify] Time: {t_elapsed:.2f}s | "
                          f"RawAngle: {raw_angle_deg:6.2f} deg -> Obs: {obs_wrist_pos:6.3f} rad | "
                          f"Target(0.1s): {rhythm_buf[10]:.2f}", end="")
                    time.sleep(CONTROL_DT)
                    continue

                # 6. 推論 (Inference)
                with torch.no_grad():
                    actions = self.policy(obs_tensor).cpu().numpy().flatten()

                # 7. アクション変換 & 送信
                # [-1, 1] -> [0, P_MAX]
                self.last_actions = np.clip(actions, -1.0, 1.0)
                pressures = (self.last_actions + 1.0) / 2.0 * P_MAX
                
                packet = HEADER + struct.pack(SEND_FMT, *pressures)
                self.ser.write(packet)
                self.ser.flush()

                # 8. ログ記録
                self.logs.append({
                    'time': t_elapsed,
                    'obs_wrist': obs_wrist_pos,
                    'raw_angle': raw_angle_deg,
                    'cmd_DF': pressures[0],
                    'cmd_F': pressures[1],
                    'target_now': rhythm_buf[0] * TARGET_FORCE
                })

                # 周期維持
                dt = time.perf_counter() - loop_start
                if dt < CONTROL_DT:
                    time.sleep(CONTROL_DT - dt)

        except KeyboardInterrupt:
            print("\n[Stop] Stopping control...")
        finally:
            self.shutdown()

    def shutdown(self):
        # 安全停止
        if hasattr(self, 'ser') and self.ser.is_open:
            for _ in range(5):
                self.ser.write(HEADER + struct.pack(SEND_FMT, 0, 0, 0))
                time.sleep(0.01)
            self.ser.close()
        
        # ログ保存
        if self.logs:
            df = pd.DataFrame(self.logs)
            os.makedirs("data_logs", exist_ok=True)
            df.to_csv(f"data_logs/deploy_log_{int(time.time())}.csv", index=False)
            print("[Log] Saved log file.")

if __name__ == "__main__":
    deployer = RLDeployer()
    deployer.run()