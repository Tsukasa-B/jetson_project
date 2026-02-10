import torch
import serial
import struct
import time
import numpy as np

# ==========================================
# ユーザー設定セクション
# ==========================================
MODEL_PATH = "policy.pt"
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 115200

# 入力: [q1, q2, dq1, dq2, force] + [last_action_1, last_action_2, last_action_3]
DUMMY_OBS = [
    0.0, 0.0, 0.0, 0.0, 0.0,  # Sensors
    0.0, 0.0, 0.0             # Last Actions
]

# 出力設定
PAM_MAX_PRESSURE = 0.6  # [MPa] 
# ※Simulink側でこの値を受け取って電圧に変換してください

# ==========================================
# システム実装セクション
# ==========================================

def load_model(path, device):
    print(f"Loading model from {path}...")
    try:
        policy = torch.jit.load(path).to(device)
        policy.eval()
        print("Model loaded successfully!")
        return policy
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def send_to_microlabbox(ser, pressures):
    """
    MicroLabBoxへ圧力データ(MPa)を送信
    Header (2bytes) + Data (8bytes * 3 doubles = 24bytes)
    """
    # Header: 255, 255
    header = struct.pack('BB', 255, 255)
    
    # Data: 3つの圧力値 (double, Big Endian)
    payload = struct.pack('>ddd', pressures[0], pressures[1], pressures[2])
    
    ser.write(header + payload)

def process_actions_to_pressure(raw_actions):
    """
    AI生出力 -> 圧力(MPa) 変換のみ行う
    Mapping: [-1.0, 1.0] -> [0.0, Pmax]
    """
    # 1. クリップ処理 (Sim側の安全策: -1~1に制限)
    clipped_actions = np.clip(raw_actions, -1.0, 1.0)
    
    # 2. 圧力への変換 [MPa]
    # -1.0 -> 0.0 MPa
    # +1.0 -> 0.6 MPa (Pmax)
    pressures_mpa = (clipped_actions + 1.0) / 2.0 * PAM_MAX_PRESSURE
    
    return clipped_actions, pressures_mpa

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    policy = load_model(MODEL_PATH, device)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
        print(f"Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"Serial Error: {e}")
        ser = None

    print("\n=== Sim-to-Real Pressure Output Test ===")
    print(f"Sending Pressure values [0.0 - {PAM_MAX_PRESSURE} MPa] to MicroLabBox")
    print("Press Ctrl+C to stop.\n")

    # 前回のActionを保持する変数（初期値0）
    last_actions = np.zeros(3, dtype=np.float32)

    try:
        with torch.no_grad():
            while True:
                start_time = time.time()

                # 1. 観測データの作成
                # ダミーセンサー(5) + 前回のAction(3) を結合
                current_sensors = DUMMY_OBS[:5] # ユーザー設定のセンサー部分
                
                # リスト結合してTensor化
                obs_list = current_sensors + last_actions.tolist()
                obs_tensor = torch.tensor(obs_list, device=device, dtype=torch.float32).unsqueeze(0)

                # 2. 推論実行
                actions_tensor = policy(obs_tensor)
                raw_actions = actions_tensor.cpu().numpy().flatten()

                # 3. 圧力変換 (MPa)
                clipped, pressures = process_actions_to_pressure(raw_actions)

                # 4. 次のステップのためにActionを更新（正規化された値を保存するのが一般的）
                # ※RSL_RLでは通常 clip後の -1~1 を次の入力に使います
                last_actions = clipped

                # 5. 表示
                print(f"\rRaw: {raw_actions[:2]}.. -> Clipped: {clipped[:2]}.. -> Pressure: {pressures} MPa", end="")

                # 6. MicroLabBoxへ送信 (MPaを送る)
                if ser is not None and ser.is_open:
                    send_to_microlabbox(ser, pressures)

                # 制御周期 (100Hz = 10ms)
                elapsed = time.time() - start_time
                if elapsed < 0.01:
                    time.sleep(0.01 - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")
        if ser is not None:
            ser.close()
        print("Done.")

if __name__ == "__main__":
    main()