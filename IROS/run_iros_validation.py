"""
Porcaro Robot: IROS 2026 Validation Experiment Runner
Target: Jetson Orin Nano + MicroLabBox via UART
Author: Robo-Dev Partner
"""
"""使い方ディレクトリは指定してね
# 実験1: 静特性（ヒステリシス）
python run_iros_validation.py exp1_static_hysteresis

# 実験2: ステップ応答
python run_iros_validation.py exp2_step_response

# 実験3: 周波数スイープ
python run_iros_validation.py exp3_frequency_sweep

# 実験4: ドラム演奏
python run_iros_validation.py exp4_drumming_task
"""

import serial
import struct
import time
import numpy as np
import pandas as pd
import threading
import sys
import argparse
import os
import signal

# ==========================================
# System Configuration
# ==========================================
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 460800       # High-speed UART required for 100Hz control
CONTROL_DT = 0.01        # 10ms (100Hz) Loop

# Communication Protocol
# SEND: Header(2) + DF(8) + F(8) + G(8) = 26 bytes
# RECV: Header(2) + P_DF(8)+P_F(8)+P_G(8)+Ang(8)+Vel(8)+Flag(8)+Force(8) = 58 bytes
SEND_FMT = '>ddd'        # [Cmd_DF, Cmd_F, Cmd_G]
RECV_FMT = '>ddddddd'    # [Meas_DF, Meas_F, Meas_G, Angle, Vel, Flag, Force]
HEADER = b'\xff\xff'
RECV_PACKET_LEN = 2 + 8 * 7

class SensorInterface(threading.Thread):
    """
    非同期シリアル受信クラス
    メインループをブロックせずに常に最新のセンサ値をバッファリングする
    """
    def __init__(self, ser):
        super().__init__()
        self.ser = ser
        self.running = True
        self.latest_data = None
        self.lock = threading.Lock()
        self.daemon = True

    def run(self):
        buffer = b''
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    buffer += self.ser.read(self.ser.in_waiting)
                    
                    # パケットの切り出し処理
                    while len(buffer) >= RECV_PACKET_LEN:
                        # ヘッダ検索
                        if buffer[:2] == HEADER:
                            packet = buffer[2:RECV_PACKET_LEN]
                            self._parse(packet)
                            buffer = buffer[RECV_PACKET_LEN:]
                        else:
                            # ヘッダがずれている場合は1バイト送る
                            buffer = buffer[1:]
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"[Sensor Error] {e}")
                self.running = False

    def _parse(self, packet_bytes):
        try:
            data = struct.unpack(RECV_FMT, packet_bytes)
            with self.lock:
                # [P_DF, P_F, P_G, Ang, Vel, Flag, Force]
                self.latest_data = {
                    'meas_pres_DF': data[0],
                    'meas_pres_F':  data[1],
                    'meas_pres_G':  data[2],
                    'angle_deg':    data[3],
                    'velocity':     data[4],
                    'flag':         data[5],
                    'force_N':      data[6]  # 力センサ値
                }
        except struct.error:
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
        
        # Check required columns
        required = ['cmd_pressure_DF', 'cmd_pressure_F', 'cmd_pressure_G']
        if not all(col in self.cmd_df.columns for col in required):
            raise ValueError(f"CSV must contain columns: {required}")

        print("[Init] Opening Serial Port...")
        self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        self.sensor = SensorInterface(self.ser)
        self.logs = []

    def _resolve_path(self, name):
        """ ファイル名からパスを解決 (拡張子なしでもOK) """
        if not name.endswith('.csv'):
            name += '.csv'
        
        # 優先順位:
        # 1. カレントの test_signals/
        # 2. カレント直下
        candidates = [
            os.path.join("test_signals", name),
            name
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Could not find signal file: {name}")

    def run(self):
        self.sensor.start()
        print("\n" + "="*40)
        print(f"  STARTING IROS VALIDATION RUN")
        print(f"  Sequence Length: {len(self.cmd_df) * CONTROL_DT:.1f} sec")
        print("="*40 + "\n")
        
        # 安全のため少し待機
        time.sleep(1.0)
        print(">>> 3...")
        time.sleep(1.0)
        print(">>> 2...")
        time.sleep(1.0)
        print(">>> 1... GO!")

        start_time = time.perf_counter()
        
        try:
            for idx, row in self.cmd_df.iterrows():
                loop_start = time.perf_counter()
                
                # 1. 現在時刻の計算
                t_elapsed = loop_start - start_time
                
                # 2. 指令値の取得
                cmd = [
                    row['cmd_pressure_DF'],
                    row['cmd_pressure_F'],
                    row['cmd_pressure_G']
                ]
                
                # 3. 送信 (Send Command)
                self._send(cmd)
                
                # 4. 受信データの取得 (Log Data)
                sensor_data = self.sensor.get_latest()
                
                log_entry = {
                    'time': t_elapsed,
                    'cmd_DF': cmd[0],
                    'cmd_F': cmd[1],
                    'cmd_G': cmd[2]
                }
                
                if sensor_data:
                    log_entry.update(sensor_data)
                else:
                    # データがまだ来ていない場合はNaN埋め（あるいは前の値）
                    log_entry.update({k: np.nan for k in ['meas_pres_DF', 'force_N', 'angle_deg']})
                
                self.logs.append(log_entry)

                # 5. 周期維持 (Sleep)
                process_time = time.perf_counter() - loop_start
                if process_time < CONTROL_DT:
                    time.sleep(CONTROL_DT - process_time)
                else:
                    # 処理落ち警告 (たまに出る分にはOK)
                    pass 

        except KeyboardInterrupt:
            print("\n[Abort] User interrupted.")
        except Exception as e:
            print(f"\n[Error] {e}")
        finally:
            self.shutdown()

    def _send(self, cmd_values):
        # 安全リミット
        clamped = [max(0.0, min(0.6, v)) for v in cmd_values]
        packet = HEADER + struct.pack(SEND_FMT, *clamped)
        self.ser.write(packet)

    def shutdown(self):
        print("\n[System] Shutting down...")
        # 全バルブ開放 (Safety)
        try:
            for _ in range(5):
                self._send([0.0, 0.0, 0.0])
                time.sleep(0.01)
        except:
            pass
            
        if self.sensor:
            self.sensor.stop()
        if self.ser:
            self.ser.close()
            
        self.save_logs()

    def save_logs(self):
        if not self.logs:
            print("[Warn] No data to save.")
            return

        df = pd.DataFrame(self.logs)
        
        # 元のファイル名 + タイムスタンプ
        base_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        timestamp = int(time.time())
        filename = f"data_{base_name}_{timestamp}.csv"
        
        save_dir = os.path.join("IROS", "test_signals") # 出力先もここに合わせておく
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, filename)
        df.to_csv(save_path, index=False)
        print(f"\n[Saved] Log file saved to:\n  -> {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IROS Validation Runner")
    parser.add_argument("csv_name", type=str, help="Name of the CSV file in IROS/test_signals (e.g. exp1_static_hysteresis)")
    args = parser.parse_args()

    controller = ExperimentController(args.csv_name)
    controller.run()