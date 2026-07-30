import serial
import struct
import time
import csv
import threading
import os
from datetime import datetime

# --- 設定 (環境に合わせて変更してください) ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
CSV_DIR = "logs_verification"

# --- データ仕様定義 ---
# 受信 (MicroLabBox -> Jetson): 6個 (double x 6 = 48 bytes)
# [0:PAMDF内圧, 1:PAMF内圧, 2:PAMG内圧, 3:WristAngle, 4:HandAngle, 5:p_flag]
RECV_FMT = '>dddddd' 
RECV_PAYLOAD_LEN = 48
RECV_PACKET_LEN = 2 + RECV_PAYLOAD_LEN # Header(2) + Payload

# 送信 (Jetson -> MicroLabBox): 3個 (double x 3 = 24 bytes)
# [0:PAMDF指令, 1:PAMF指令, 2:PAMG指令]
SEND_FMT = '>ddd'

# --- グローバル変数 ---
# 初期値はすべて0
current_cmd = [0.3, 0.0, 0.0]  # [DF, F, G]MPa
is_running = True

def serial_worker():
    """通信とデータ保存を行うバックグラウンド処理"""
    global current_cmd, is_running
    
    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)
    filename = f"{CSV_DIR}/check_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        print(f"\n[System] Connected to {SERIAL_PORT}")
        print(f"[System] Logging to: {filename}")
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # CSVヘッダー作成
            header = [
                "time", 
                "cmd_DF", "cmd_F", "cmd_G",             # 指令値 (3)
                "meas_DF", "meas_F", "meas_G",          # 実測圧力 (3)
                "ang_wrist", "ang_hand", "p_flag_recv"  # 角度・フラグ (3)
            ]
            writer.writerow(header)
            
            start_time = time.time()
            buffer = b''

            while is_running:
                # ----------------------------------------
                # 1. 送信 (Jetson -> MicroLabBox)
                # ----------------------------------------
                header_byte = b'\xff\xff'
                payload = struct.pack(SEND_FMT, *current_cmd)
                ser.write(header_byte + payload)
                
                # ----------------------------------------
                # 2. 受信 (MicroLabBox -> Jetson)
                # ----------------------------------------
                # データがあるだけバッファに読み込む
                if ser.in_waiting > 0:
                    buffer += ser.read(ser.in_waiting)
                
                # パケットサイズ分たまっているか確認
                while len(buffer) >= RECV_PACKET_LEN:
                    # ヘッダーチェック
                    if buffer[0] == 0xFF and buffer[1] == 0xFF:
                        try:
                            # デコード
                            packet_data = buffer[2:RECV_PACKET_LEN]
                            recv_vals = struct.unpack(RECV_FMT, packet_data)
                            
                            # タイムスタンプ
                            t = time.time() - start_time
                            
                            # コンソール表示 (見やすく整形)
                            # p_flagが 1.0 なら [ON], 0.0 なら [OFF] と表示
                            flag_status = "[ON]" if recv_vals[5] > 0.5 else "[OFF]"
                            
                            print(f"\r[T:{t:.2f} s] {flag_status} "
                                  f"Pres(DF/F/G): {recv_vals[0]:.2f}/{recv_vals[1]:.2f}/{recv_vals[2]:.2f} | "
                                  f"Ang(W/H): {recv_vals[3]:.2f}/{recv_vals[4]:.2f}", end="")
                            
                            # CSV書き込み
                            # row = [時間] + [指令3つ] + [受信6つ]
                            row = [t] + list(current_cmd) + list(recv_vals)
                            writer.writerow(row)
                            
                            # 処理した分をバッファから削除
                            buffer = buffer[RECV_PACKET_LEN:]
                            
                        except Exception as e:
                            print(f"\n[Error] Parse failed: {e}")
                            buffer = buffer[1:] # 1バイトずらして再試行
                    else:
                        # ヘッダーじゃないなら1バイト捨てる
                        buffer = buffer[1:]

                time.sleep(0.01) # 10msループ維持

    except Exception as e:
        print(f"\n[Error] Serial port error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

def main():
    global current_cmd, is_running
    
    t = threading.Thread(target=serial_worker)
    t.start()
    
    print("\n" + "="*50)
    print(" Jetson-MicroLabBox Signal Checker V2")
    print("="*50)
    print(" 操作方法:")
    print("  入力形式: [DF] [F] [G]")
    print("  例1: '0.3 0.0 0.0' -> DFのみ0.3MPa")
    print("  例2: '0.2 0.2 0.0' -> 手首拮抗配置を0.2MPaで剛性化")
    print("  例3: '0'           -> 全て脱力 (0,0,0)")
    print("  'q' -> 終了")
    print("="*50)

    try:
        while True:
            user_input = input() # 入力待ち
            
            if user_input.lower() == 'q':
                is_running = False
                break
            
            try:
                # スペース区切りで数値を取得
                vals = list(map(float, user_input.split()))
                
                if len(vals) == 1:
                    # 1つだけ入力されたら、すべて0にする安全策（または全指定）
                    # ここでは "0" と打たれたら全停止とみなす
                    if vals[0] == 0:
                        current_cmd = [0.0, 0.0, 0.0]
                        print(">> ALL STOP (0.0 MPa)")
                    else:
                        print(">> Error: Please enter 3 values (e.g., '0.2 0.2 0')")
                        
                elif len(vals) == 3:
                    # 3つ入力されたら適用
                    # 安全リミット (例: 0.6MPa)
                    if all(0.0 <= v <= 0.6 for v in vals):
                        current_cmd = vals
                        print(f">> Command Set: DF={vals[0]}, F={vals[1]}, G={vals[2]}")
                    else:
                        print(">> Error: Value out of range (0.0 - 0.6 MPa)")
                else:
                    print(">> Error: Invalid format.")
                    
            except ValueError:
                print(">> Error: Invalid number.")

    except KeyboardInterrupt:
        is_running = False

    t.join()
    print("\n[System] Done.")

if __name__ == "__main__":
    main()