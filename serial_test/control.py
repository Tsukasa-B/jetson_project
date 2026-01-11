import serial
import struct
import time

# --- 設定 ---
PORT = '/dev/ttyUSB0'
BAUD = 115200       # dSPACE側の設定と合わせる
TIMEOUT = 0.1

# --- 通信フォーマット (Big Endian '>' を指定) ---
# 受信: dSPACE -> Jetson (センサー4つ: 32バイト)
FMT_RX = '>dddd' 
SIZE_DATA_RX = struct.calcsize(FMT_RX) # 32

# 送信: Jetson -> dSPACE (指令3つ: 24バイト)
FMT_TX = '>ddd'
SIZE_DATA_TX = struct.calcsize(FMT_TX) # 24

# ヘッダー (2バイト)
HEADER_BYTE = b'\xff'
HEADER_BYTES = b'\xff\xff'

def main():
    print(f"--- 双方向通信システム起動: {PORT} ---")
    print("受信待機中 (Header: FF FF)...")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        while True:
            # ==========================================
            # 1. 受信処理 (dSPACEからのデータを受け取る)
            # ==========================================
            # 1バイトずつ読んでヘッダー(FF, FF)を探す
            if ser.read(1) == HEADER_BYTE:
                if ser.read(1) == HEADER_BYTE:
                    # ヘッダー発見！続く32バイト（本体）を読む
                    raw_rx = ser.read(SIZE_DATA_RX)
                    
                    if len(raw_rx) == SIZE_DATA_RX:
                        # 変換 (Big Endian)
                        sensors = struct.unpack(FMT_RX, raw_rx)
                        print(f"📥 受信: {sensors}")

                        # ==========================================
                        # 2. 送信処理 (データが来たら、すぐに送り返す)
                        # ==========================================
                        # テスト用の指令値 (1.0, 2.0, 3.0)
                        cmd_theta   = 1.0
                        cmd_k_wrist = 2.0
                        cmd_k_grip  = 3.0

                        # データをパック (Big Endian)
                        payload = struct.pack(FMT_TX, cmd_theta, cmd_k_wrist, cmd_k_grip)
                        
                        # ヘッダー (FF FF) を先頭にくっつけて送信！
                        # 合計 26バイト (2 + 24) 送られる
                        ser.write(HEADER_BYTES + payload)
                        
                        # print("📤 送信完了") # デバッグ用

    except KeyboardInterrupt:
        print("\n停止しました")
    except Exception as e:
        print(f"エラー発生: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()