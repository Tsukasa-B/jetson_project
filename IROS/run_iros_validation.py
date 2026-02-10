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
BAUD_RATE = 230400       # High-speed UART required for 100Hz control
CONTROL_DT = 0.02        # 20ms (50Hz) Loop - Matches RL Agent Cycle

# Communication Protocol (Match with MicroLabBox)
# SEND: Header(2) + DF(8) + F(8) + G(8) = 26 bytes
# RECV: Header(2) + P_DF(8)+P_F(8)+P_G(8)+Ang(8)+Vel(8)+Flag(8)+Force(8) = 58 bytes
SEND_FMT = '>ddd'        # [Cmd_DF, Cmd_F, Cmd_G]
RECV_FMT = '>ddddddd'    # [Meas_DF, Meas_F, Meas_G, Angle, Vel, Flag, Force]
HEADER = b'\xff\xff'
RECV_PACKET_LEN = 2 + 8 * 7

class SensorInterface(threading.Thread):
    """
    非同期シリアル受信クラス
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
                    
                    # パケット処理 (バッファ最適化)
                    while len(buffer) >= RECV_PACKET_LEN:
                        # ヘッダ検索
                        idx = buffer.find(HEADER)
                        if idx == -1:
                            buffer = buffer[-1:] 
                            break
                        
                        if idx > 0:
                            buffer = buffer[idx:]

                        if len(buffer) < RECV_PACKET_LEN:
                            break

                        packet = buffer[2:RECV_PACKET_LEN]
                        self._parse(packet)
                        buffer = buffer[RECV_PACKET_LEN:]
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
                    'force_N':      data[6]
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
        # パス解決ロジックを改良
        self.csv_path = self._resolve_path(csv_name)
        print(f"[Init] Loading Sequence: {self.csv_path}")
        
        try:
            self.cmd_df = pd.read_csv(self.csv_path)
        except Exception as e:
            print(f"[Error] Failed to read CSV: {e}")
            sys.exit(1)
        
        # Check required columns
        required = ['cmd_pressure_DF', 'cmd_pressure_F', 'cmd_pressure_G']
        if not all(col in self.cmd_df.columns for col in required):
            raise ValueError(f"CSV must contain columns: {required}")

        print("[Init] Opening Serial Port...")
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            self.sensor = SensorInterface(self.ser)
            self.logs = []
        except serial.SerialException as e:
            print(f"[Error] Serial Port Open Failed: {e}")
            sys.exit(1)

    def _resolve_path(self, name):
        """ 
        ファイル名からパスを解決 
        このスクリプト(run_iros_validation.py)のある場所を基準に test_signals を探す
        """
        if not name.endswith('.csv'):
            name += '.csv'
        
        # 1. このスクリプトのあるディレクトリを取得 (jetson_project/IROS/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. ターゲットディレクトリを構築 (jetson_project/IROS/test_signals/)
        target_dir = os.path.join(script_dir, "test_signals")
        
        # 3. 候補パスを作成
        target_path = os.path.join(target_dir, name)
        
        if os.path.exists(target_path):
            return target_path
            
        # 念のため、実行場所からの相対パスでも探す
        if os.path.exists(name):
            return os.path.abspath(name)

        print(f"[Error] CSV File not found: {name}")
        print(f"  Searched in: {target_dir}")
        sys.exit(1)

    def run(self):
        self.sensor.start()
        print("\n" + "="*40)
        print(f"  STARTING VALIDATION RUN (No Drum)")
        print(f"  Sequence Length: {len(self.cmd_df) * CONTROL_DT:.1f} sec")
        print("="*40 + "\n")
        
        # 安全のため初期化
        self._send([0, 0, 0])
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
                
                # 1. 経過時間
                t_elapsed = loop_start - start_time
                
                # 2. 指令値取得
                cmd = [
                    row['cmd_pressure_DF'],
                    row['cmd_pressure_F'],
                    row['cmd_pressure_G']
                ]
                
                # 3. 送信
                self._send(cmd)
                
                # 4. ログ記録
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
                    log_entry.update({k: np.nan for k in ['meas_pres_DF', 'force_N', 'angle_deg']})
                
                self.logs.append(log_entry)

                # 5. 高精度周期維持 (Busy Wait)
                while (time.perf_counter() - loop_start) < CONTROL_DT:
                    pass

        except KeyboardInterrupt:
            print("\n[Abort] User interrupted.")
        except Exception as e:
            print(f"\n[Error] {e}")
        finally:
            self.shutdown()

    def _send(self, cmd_values):
        # 安全リミット [0.0 - 1.0] or [0.0 - 0.6MPa] 
        clamped = [max(0.0, min(1.0, v)) for v in cmd_values]
        packet = HEADER + struct.pack(SEND_FMT, *clamped)
        self.ser.write(packet)

    def shutdown(self):
        print("\n[System] Shutting down...")
        try:
            for _ in range(5):
                self._send([0.0, 0.0, 0.0]) # 脱力
                time.sleep(0.02)
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
        
        # ファイル名生成
        base_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        timestamp = int(time.time())
        filename = f"data_{base_name}_{timestamp}.csv"
        
        # 保存先: このスクリプトと同じ階層にある test_signals フォルダ
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, "test_signals")
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, filename)
        
        df.to_csv(save_path, index=False)
        print(f"\n[Saved] Log file saved to:\n  -> {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IROS Validation Runner")
    parser.add_argument("csv_name", type=str, help="CSV file name (e.g. exp1_static_hysteresis)")
    args = parser.parse_args()

    controller = ExperimentController(args.csv_name)
    controller.run()