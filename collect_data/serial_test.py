import serial
import serial.tools.list_ports
import os
import sys
import time
import struct

# ==========================================
# 設定
# ==========================================
TARGET_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200
CHECK_DURATION = 5.0  # 受信チェックを行う時間(秒)

def print_status(msg, status="INFO"):
    colors = {
        "INFO": "\033[94m[INFO]\033[0m",
        "OK": "\033[92m[OK]\033[0m",
        "WARN": "\033[93m[WARN]\033[0m",
        "ERROR": "\033[91m[ERROR]\033[0m"
    }
    print(f"{colors.get(status, '[?]')} {msg}")

def check_device_existence():
    print_status(f"Checking device existence: {TARGET_PORT}...", "INFO")
    if os.path.exists(TARGET_PORT):
        print_status(f"Device found: {TARGET_PORT}", "OK")
        return True
    else:
        print_status(f"Device NOT found: {TARGET_PORT}", "ERROR")
        # 他の候補を探す
        ports = list(serial.tools.list_ports.comports())
        if ports:
            print_status("Available ports:", "INFO")
            for p in ports:
                print(f"  - {p.device} ({p.description})")
        else:
            print_status("No serial ports found on this system.", "WARN")
        return False

def check_permission():
    print_status("Checking file permissions...", "INFO")
    try:
        # 読み書き権限のチェック
        r_ok = os.access(TARGET_PORT, os.R_OK)
        w_ok = os.access(TARGET_PORT, os.W_OK)
        
        if r_ok and w_ok:
            print_status("Permissions (Read/Write): Granted", "OK")
            return True
        else:
            print_status(f"Permissions: Read={r_ok}, Write={w_ok}", "ERROR")
            print_status(f"Run: sudo chmod 666 {TARGET_PORT}", "WARN")
            return False
    except Exception as e:
        print_status(f"Permission check failed: {e}", "ERROR")
        return False

def check_connection_and_traffic():
    print_status(f"Attempting to open {TARGET_PORT} @ {BAUD_RATE}bps...", "INFO")
    ser = None
    try:
        ser = serial.Serial(TARGET_PORT, BAUD_RATE, timeout=0.1)
        print_status("Port opened successfully.", "OK")
        print(f"  > Settings: {ser.get_settings()}")
        
        # 1. 送信テスト
        print_status("Testing TX (Sending dummy packet)...", "INFO")
        try:
            # Header(FFFF) + data(1.0, 2.0, 3.0)
            dummy_data = b'\xFF\xFF' + struct.pack('>ddd', 1.0, 2.0, 3.0)
            ser.write(dummy_data)
            print_status(f"Sent {len(dummy_data)} bytes. (No error raised)", "OK")
        except Exception as e:
            print_status(f"TX Failed: {e}", "ERROR")

        # 2. 受信テスト (5秒間リッスン)
        print_status(f"Testing RX (Listening for {CHECK_DURATION}s)...", "INFO")
        start_time = time.time()
        received_bytes = b''
        
        while time.time() - start_time < CHECK_DURATION:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                received_bytes += chunk
                # 少しデータが溜まったらループを抜ける（全部待つ必要はない）
                if len(received_bytes) > 20: 
                    break
            time.sleep(0.01)
            
        if len(received_bytes) > 0:
            print_status(f"RX Data received: {len(received_bytes)} bytes", "OK")
            print(f"  > Raw Head (Hex): {received_bytes[:16].hex(' ')}")
            
            # ヘッダーチェック
            if b'\xff\xff' in received_bytes:
                print_status("Header (FF FF) detected!", "OK")
            else:
                print_status("Header (FF FF) NOT found in received data.", "WARN")
                print("  > Possible baudrate mismatch or wrong endian.")
        else:
            print_status("No data received from dSPACE.", "WARN")
            print("  > Check: 1. Cable Connection (Cross cable?)")
            print("  > Check: 2. Is dSPACE model running?")
            
    except serial.SerialException as e:
        print_status(f"Port Open Failed: {e}", "ERROR")
        if "Permission denied" in str(e):
            print_status("Try adding user to dialout group or chmod.", "WARN")
        elif "Device or resource busy" in str(e):
            print_status("Another process is using this port! (lsof | grep ttyUSB0)", "WARN")
            
    finally:
        if ser and ser.is_open:
            ser.close()
            print_status("Port closed.", "INFO")

if __name__ == "__main__":
    print("="*50)
    print(" Jetson Serial Diagnostic Tool")
    print("="*50)
    
    if check_device_existence():
        if check_permission():
            check_connection_and_traffic()
            
    print("="*50)