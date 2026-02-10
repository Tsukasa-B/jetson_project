import torch
import time

# --- 設定 ---
MODEL_PATH = "policy.pt"
INPUT_DIM = 4  # 観測空間の次元数（関節角度x2, 角速度x2）
HZ = 50        # 目標制御周期

# 1. デバイス準備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. モデル読み込み
print(f"Loading model from {MODEL_PATH}...")
try:
    # TorchScript形式のモデルをロード
    policy = torch.jit.load(MODEL_PATH, map_location=device)
    policy.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 3. ダミー入力 (Batch=1, Dim=4)
obs = torch.zeros((1, INPUT_DIM), device=device)

# 4. 推論ループ
print(f"Starting loop test (Target: {HZ}Hz)... Press Ctrl+C to stop.")
try:
    while True:
        start_time = time.time()

        # 推論実行
        with torch.no_grad():
            action = policy(obs)

        # 処理時間計測
        process_time = time.time() - start_time

        # 周期調整
        wait_time = (1.0 / HZ) - process_time
        if wait_time > 0:
            time.sleep(wait_time)

        # 定期的に生存報告（1秒に1回くらい）
        if int(time.time()) % 1 == 0 and wait_time > 0:
             # 最初の1つの環境のアクションを表示
             print(f"Running... Action: {action[0].cpu().numpy()} (Process: {process_time*1000:.2f}ms)")

except KeyboardInterrupt:
    print("\nTest finished.")