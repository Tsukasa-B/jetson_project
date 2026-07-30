import serial
import struct
import time

PORT = '/dev/ttyUSB0'
BAUD = 115200

# 正しいフォーマット: Header(FFFF) + Big Endian Double x 3
# dSPACE側では、Serial Receiveブロックの設定を以下に合わせる必要があります:
# - Header: [255, 255] (または 0xFF 0xFF)
# - Data Type: double
# - Byte Order: Big Endian
FMT_TX = '>ddd'

def main():
    print(f"--- dSPACE送信テスト (Correct Format) ---")
    ser = serial.Serial(PORT, BAUD)
    
    try:
        count = 0
        while True:
            # テストパターン: 
            # DF: 0.1, 0.2, ... 
            # F : 1.0 (固定)
            # G : 鋸波
            val_df = (count % 10) * 0.1
            val_f  = 1.0
            val_g  = (count % 100) * 0.01
            
            # パック (Header付き)
            payload = struct.pack(FMT_TX, val_df, val_f, val_g)
            packet = b'\xFF\xFF' + payload
            
            ser.write(packet)
            print(f"送信: DF={val_df:.1f}, F={val_f:.1f}, G={val_g:.2f}")
            
            time.sleep(0.01) # 10ms周期
            count += 1
            
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

if __name__ == "__main__":
    main()