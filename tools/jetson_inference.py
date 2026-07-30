import torch
import time
import math
import numpy as np
import socket
import struct
import warnings

# --- 警告を無視 (ログを見やすくするため) ---
warnings.filterwarnings("ignore")

# --- 設定 (Configuration) ---
MODEL_PATH = "policy.pt"
LOOP_RATE_HZ = 50.0

# リズム設定
TARGET_BPM = 120.0
TARGET_FORCE = 20.0

# 通信設定 (MicroLabBox / 下位コントローラの設定に合わせて変更してください)
#UDP_TARGET_IP = "192.168.12.1"  # ★送り先のIPアドレス (MicroLabBox等)
UDP_TARGET_IP = "127.0.0.1"
UDP_TARGET_PORT = 5000          # ★送り先のポート番号

# 物理パラメータ (Isaac Labの torque.py と一致させる)
P_MAX = 0.6  # [MPa]

def main():
    print("="*50)
    print(f"[INFO] Starting Porcaro Jetson Controller")
    print(f"[INFO] Target: {UDP_TARGET_IP}:{UDP_TARGET_PORT}")
    print("="*50)

    # 1. UDPソケットの作成
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 2. モデルのロード
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Inference Device: {device}")
    
    try:
        policy = torch.jit.load(MODEL_PATH, map_location=device)
        policy.eval()
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # 内部変数の初期化
    beat_interval = 60.0 / TARGET_BPM
    start_time = time.time()
    
    print("[INFO] Loop started. Press Ctrl+C to stop.")
    
    try:
        while True:
            loop_start = time.time()
            
            # ==========================================
            # A. 観測データの作成 (Observation)
            # ==========================================
            t = time.time() - start_time
            
            # --- センサー値の取得 (現在はダミー) ---
            # ※ 本来はここでUDP受信などを行い、実際の角度を入れる
            q_wrist = 0.0
            q_grip  = -0.14
            qd_wrist = 0.0
            qd_grip  = 0.0
            
            # --- リズム情報の計算 ---
            beat_index = math.floor(t / beat_interval) + 1
            next_beat_time = beat_index * beat_interval
            time_to_next_hit = next_beat_time - t
            phase_signal = math.sin(2 * math.pi * t / beat_interval)
            
            # 入力テンソル作成
            obs_list = [
                q_wrist, q_grip, 
                qd_wrist, qd_grip,
                time_to_next_hit,
                TARGET_FORCE,
                phase_signal,
                0.0 # hit_status
            ]
            obs_tensor = torch.tensor([obs_list], dtype=torch.float32, device=device)
            
            # ==========================================
            # B. 推論 (Inference)
            # ==========================================
            with torch.no_grad():
                actions = policy(obs_tensor)
                
            # CPU/Numpyへ変換
            act = actions.cpu().numpy()[0] # [theta_eq, K_w, K_g]

            act = np.clip(act, -1.0, 1.0)
            
            # ==========================================
            # C. 翻訳 (Translator: Action -> Pressure)
            # ==========================================
            # Isaac Lab の torque.py のロジックを完全再現 [cite: 2025 12 10-1]
            
            # 1. Base Pressure (剛性) & Diff Pressure (平衡点)
            # act[1] (Kw) : -1~1 -> 0~0.3 MPa
            max_p_base = P_MAX * 0.5
            p_base = max_p_base * (act[1] + 1.0) * 0.5
            
            # act[0] (Eq) : -1~1 -> -0.3~0.3 MPa
            max_p_diff = P_MAX * 0.5
            p_diff = max_p_diff * act[0]
            
            # 2. Wrist Pressures (拮抗)
            p_df = np.clip(p_base + p_diff, 0.0, P_MAX)
            p_f  = np.clip(p_base - p_diff, 0.0, P_MAX)
            
            # 3. Grip Pressure (単一)
            # act[2] (Kg) : -1~1 -> 0~0.6 MPa
            p_g  = P_MAX * (act[2] + 1.0) * 0.5
            
            # ==========================================
            # D. 送信 (UDP Transmission)
            # ==========================================
            
            # 3つの圧力値をバイナリパック (Double x 3 = 24 bytes)
            # 'd' = double (8byte), '>' = Big Endian (ネットワーク標準)
            # ※ MicroLabBox側の仕様に合わせて '<' (Little Endian) に変更が必要かも
            payload = struct.pack('>ddd', p_df, p_f, p_g)
            sock.sendto(payload, (UDP_TARGET_IP, UDP_TARGET_PORT))

            # ==========================================
            # E. ログ & 周期管理
            # ==========================================
            # 簡易ログ (圧力値を表示)
            print(f"t={t:.2f} | P_DF={p_df:.3f}, P_F={p_f:.3f}, P_G={p_g:.3f} [MPa]")

            # 50Hz維持
            elapsed = time.time() - loop_start
            sleep_time = (1.0 / LOOP_RATE_HZ) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Safe shutdown. Sending zero pressure...")
        # 安全のためゼロを送って終了
        zero_payload = struct.pack('>ddd', 0.0, 0.0, 0.0)
        sock.sendto(zero_payload, (UDP_TARGET_IP, UDP_TARGET_PORT))
        print("[INFO] Done.")

if __name__ == "__main__":
    main()