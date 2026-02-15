"""
Porcaro Robot: IROS 2026 Validation Experiment Runner (No-Drum / Free Motion)
Target: Jetson Orin Nano + MicroLabBox via UART
Author: Robo-Dev Partner

python IROS/run_iros_validation.py exp1_static_hysteresis
python IROS/run_iros_validation.py exp2_step_response
python IROS/run_iros_validation.py exp3_frequency_sweep

"""
import serial
import struct
import time
import numpy as np
import pandas as pd
import threading
import argparse
import os
import sys

# ==========================================
# System Configuration
# ==========================================
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 230400       # MLBの上限に合わせる
CONTROL_DT = 0.02        # 50Hz

# 通信パケット定義 (MLBの構成に合わせてください)
# 例: Header(2) + Double(8)*7 = 58 bytes
SEND_FMT = '>ddd'        # [Cmd_DF, Cmd_F, Cmd_G]
RECV_FMT = '>ddddddd'    # [Meas_DF, Meas_F, Meas_G, Angle, Vel, Flag, Force]
HEADER = b'\xff\xff'
RECV_PACKET_LEN = 2 + 8 * 7

class SensorInterface(threading.Thread):
    def __init__(self, ser):
        super().__init__()
        self.ser = ser
        self.running = True
        self.latest_data = None
        self.lock = threading.Lock()
        self.daemon = True

    def run(self):
        self.ser.reset_input_buffer()
        buffer = b''
        
        while self.running:
            try:
                # バッファにあるデータを全て読み出す
                if self.ser.in_waiting > 0:
                    buffer += self.ser.read(self.ser.in_waiting)
                    
                    # パケット単位で処理
                    while len(buffer) >= RECV_PACKET_LEN:
                        # ヘッダ検索
                        idx = buffer.find(HEADER)
                        if idx == -1:
                            # ヘッダが見つからないなら捨てる（次のデータ待ち）
                            buffer = buffer[-RECV_PACKET_LEN:]
                            break
                        
                        if idx > 0:
                            buffer = buffer[idx:]
                            
                        if len(buffer) < RECV_PACKET_LEN:
                            break

                        # パケット抽出
                        packet = buffer[:RECV_PACKET_LEN]
                        buffer = buffer[RECV_PACKET_LEN:] # 使った分を削除

                        # 最新データとして解析
                        self._parse(packet[2:]) # ヘッダ除く
                else:
                    time.sleep(0.001)

            except Exception as e:
                print(f"[Sensor Error] {e}")
                self.running = False

    def _parse(self, packet_bytes):
        try:
            data = struct.unpack(RECV_FMT, packet_bytes)
            with self.lock:
                self.latest_data = {
                    'meas_pres_DF': data[0],
                    'meas_pres_F':  data[1],
                    'meas_pres_G':  data[2],
                    'angle_deg':    data[3],
                    'velocity':     data[4],
                    'flag':         data[5],
                    'force_N':      data[6]  # MLB側でピークホールド処理されている前提
                }
        except:
            pass

    def get_latest(self):
        with self.lock:
            return self.latest_data

    def stop(self):
        self.running = False

class ExperimentController:
    def __init__(self, csv_name):
        self.csv_path = self._resolve_path(csv_name)
        print(f"[Init] Loading Sequence: {self.csv_path}")
        self.cmd_df = pd.read_csv(self.csv_path)

        print(f"[Init] Opening Serial Port {SERIAL_PORT} @ {BAUD_RATE}...")
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            self.sensor = SensorInterface(self.ser)
            self.logs = []
        except serial.SerialException as e:
            print(f"[Error] Serial Port Open Failed: {e}")
            sys.exit(1)

    def _resolve_path(self, name):
        if not name.endswith('.csv'): name += '.csv'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "test_signals", name)
        if os.path.exists(path): return path
        if os.path.exists(name): return os.path.abspath(name)
        print(f"[Error] File not found: {name}"); sys.exit(1)

    def run(self):
        self.sensor.start()
        print(f"\n=== STARTING EXPERIMENT (50Hz Control) ===")
        
        # 安全開始
        self._send([0,0,0])
        time.sleep(2.0)
        print("GO!")
        
        start_time = time.perf_counter()
        
        try:
            for idx, row in self.cmd_df.iterrows():
                loop_start = time.perf_counter()
                t_elapsed = loop_start - start_time
                
                # 1. 指令値送信
                cmd = [row['cmd_pressure_DF'], row['cmd_pressure_F'], row['cmd_pressure_G']]
                self._send(cmd)
                
                # 2. センサ値取得 (MLBから送られてきた最新のピークホールド値)
                sensor_data = self.sensor.get_latest()
                
                # 3. ログ保存
                log = {'time': t_elapsed, 'cmd_DF': cmd[0], 'cmd_F': cmd[1], 'cmd_G': cmd[2]}
                if sensor_data:
                    log.update(sensor_data)
                else:
                    log.update({k: 0.0 for k in ['meas_pres_DF', 'meas_pres_F', 'meas_pres_G', 'angle_deg', 'velocity', 'flag', 'force_N']})
                self.logs.append(log)
                
                # 4. 周期維持 (50Hz)
                while (time.perf_counter() - loop_start) < CONTROL_DT:
                    pass
                    
        except KeyboardInterrupt:
            print("\nAborted by user.")
        finally:
            self.shutdown()

    def _send(self, cmd):
        cmd = [max(0.0, min(1.0, v)) for v in cmd]
        self.ser.write(HEADER + struct.pack(SEND_FMT, *cmd))

    def shutdown(self):
        print("Shutting down...")
        self._send([0,0,0])
        self.sensor.stop()
        self.ser.close()
        self.save_logs()

    def save_logs(self):
        if not self.logs: return
        name = f"data_{os.path.splitext(os.path.basename(self.csv_path))[0]}_{int(time.time())}.csv"
        path = os.path.join(os.path.dirname(self.csv_path), name)
        pd.DataFrame(self.logs).to_csv(path, index=False)
        print(f"Saved: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_name", type=str)
    args = parser.parse_args()
    ExperimentController(args.csv_name).run()