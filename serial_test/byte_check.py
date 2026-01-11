import serial
import struct
import time

# --- 設定 ---
PORT = '/dev/ttyUSB0'
BAUD = 115200 

# ★重要修正1: '>' をつけて「ビッグエンディアン」にする
# dSPACE(PowerPC)はビッグエンディアン、Jetson(ARM)はリトルエンディアンだからです。
FMT_RX = '>dddd'  # 受信: センサー4つ (32byte)
FMT_TX = '>ddd'   # 送信: 指令3つ (24byte) も '>' をつける

def main():
    print(f"--- 通信開始: {PORT} ---")
    
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        ser.reset_input_buffer()
        print("✅ 接続完了！ヘッダー(FF FF)を探しています...")
        
        while True:
            # -----------------------------------------------
            # 1. ヘッダー同期（FF, FF が来るまで読み捨てる）
            # -----------------------------------------------
            # 1バイト読む
            b1 = ser.read(1)
            if b1 == b'\xff':
                # もう1バイト読む
                b2 = ser.read(1)
                if b2 == b'\xff':
                    # ★ビンゴ！ヘッダー発見！
                    # 続く32バイトが本物のデータ
                    raw_data = ser.read(32)
                    
                    if len(raw_data) == 32:
                        # 変換（ビッグエンディアンで）
                        sensors = struct.unpack(FMT_RX, raw_data)
                        print(f"📡 センサー受信: {sensors}")
                        
                        # --- 受信できたら送信する ---
                        cmd_theta = 1.0
                        cmd_k_wrist = 2.0
                        cmd_k_grip = 3.0
                        tx_bytes = struct.pack(FMT_TX, cmd_theta, cmd_k_wrist, cmd_k_grip)
                        ser.write(tx_bytes)
                    else:
                        pass # データ不足ならスキップ

            # CPUを休ませるならここに入れるが、ヘッダー探索中は回したほうがいい
            # time.sleep(0.001) 

    except KeyboardInterrupt:
        print("\n停止")
    except Exception as e:
        print(f"エラー: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()