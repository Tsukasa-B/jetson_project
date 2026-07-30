import serial
import struct
import time

# --- 設定 ---
PORT = '/dev/ttyUSB0'
BAUD = 115200  # ★Simulinkも115200に戻してビルド必須！
FMT_RX = 'dddd' # センサー4つ (32byte)
FMT_TX = 'ddd'  # 指令3つ (24byte)
SIZE_RX = struct.calcsize(FMT_RX)

def main():
    print(f"--- 強制送信モード: {PORT} ---")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
        print("✅ 接続成功！一方的にデータを送り始めます...")
        
        # スタートダッシュ：バッファをクリア
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        while True:
            # ---------------------------
            # 1. 受信処理 (データがあれば読む)
            # ---------------------------
            if ser.in_waiting >= SIZE_RX:
                try:
                    data_bytes = ser.read(SIZE_RX)
                    sensors = struct.unpack(FMT_RX, data_bytes)
                    print(f"受信: {sensors}")
                except:
                    pass # エラーが出ても気にせず進む

            # ---------------------------
            # 2. 送信処理 (★ここが重要！ifの外に出す)
            # ---------------------------
            # 相手が黙っていても、こちらは 1, 2, 3 を送り続ける
            cmd_theta = 1.0
            cmd_k_wrist = 2.0
            cmd_k_grip = 3.0
            
            # dSPACEがリトルエンディアンか確認するため '<' をつけておく
            tx_bytes = struct.pack('<ddd', cmd_theta, cmd_k_wrist, cmd_k_grip)
            ser.write(tx_bytes)
            
            # 少し待つ (SimulinkのSampleTime 0.01に合わせておく)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("停止")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()