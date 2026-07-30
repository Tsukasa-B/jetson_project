import socket
import struct

# 受信設定 (送信側と合わせる)
UDP_IP = "127.0.0.1"
UDP_PORT = 5000

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"[INFO] Listening on {UDP_IP}:{UDP_PORT}...")
    print("[INFO] Press Ctrl+C to stop.")

    try:
        while True:
            # データ受信 (24バイト = double(8) * 3)
            data, addr = sock.recvfrom(1024)
            
            if len(data) == 24:
                # バイナリを数値に戻す (Big Endian double 3つ)
                p_df, p_f, p_g = struct.unpack('>ddd', data)
                
                print(f"Received from {addr}:")
                print(f"  P_DF (伸筋)   : {p_df:.4f} MPa")
                print(f"  P_F  (屈筋)   : {p_f:.4f} MPa")
                print(f"  P_G  (グリップ): {p_g:.4f} MPa")
                print("-" * 30)
            else:
                print(f"[WARN] Unexpected data length: {len(data)} bytes")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")

if __name__ == "__main__":
    main()