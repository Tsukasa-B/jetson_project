import serial
import time

# 設定
PORT = '/dev/ttyUSB0'
BAUD = 115200  # MicroLabBox側の速度に合わせて変更してください

def main():
    print(f"接続試行中: {PORT}...")
    try:
        # ポートを開く
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print("✅ ポートオープン成功！通信待機中...")
        
        # MicroLabBoxへ適当なデータを送ってみる（挨拶）
        ser.write(b'Hello from Jetson\n')
        
        while True:
            # データが来ていたら表示
            if ser.in_waiting > 0:
                data = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"受信: {data}")
            time.sleep(0.1)

    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    main()