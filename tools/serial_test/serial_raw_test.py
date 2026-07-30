import serial
import time

PORT = '/dev/ttyUSB0'
BAUD = 115200

def main():
    print(f"接続: {PORT} (Rawモード)")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print("✅ 待機中... 何か受信すればすぐに表示します")
        
        while True:
            # データが1バイトでもあれば読む
            if ser.in_waiting > 0:
                # 溜まっている分を全部読む
                data = ser.read(ser.in_waiting)
                print(f"受信データ: {data}")
            time.sleep(0.1)

    except Exception as e:
        print(f"エラー: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()