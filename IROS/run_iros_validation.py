"""
Porcaro Robot: IROS 2026 Validation Experiment Runner (Async: Recv 200Hz / Send 50Hz)
Target: Jetson Orin Nano + MicroLabBox
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
import copy

# ==========================================
# System Configuration
# ==========================================
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 230400       # 200Hz通信ならこれでOK (500Hzは不可)
CONTROL_DT = 0.02        # 制御・送信は50Hz

# MicroLabBox側の送信周期設定: 0.005s (200Hz) に設定してください

SEND_FMT = '>ddd'        # [Cmd_DF, Cmd_F, Cmd_G]
RECV_FMT = '>ddddddd'    # [Meas_DF, Meas_F, Meas_G, Angle, Vel, Flag, Force]
HEADER = b'\xff\xff'
RECV_PACKET_LEN = 2 + 8 * 7

class SensorReceiver(threading.Thread):
    """
    受信専用スレッド
    MicroLabBoxから来る200Hzのデータを全て取りこぼさずにバッファへ格納する
    """
    def __init__(self, ser):
        super().__init__()
        self.ser = ser
        self.running = True
        self.data_buffer = []  # 受信データを溜めておくリスト
        self.latest_sample = None # 制御用に使う「最新の1つ」
        self.lock = threading.Lock()
        self.daemon = True

    def run(self):
        self.ser.reset_input_buffer()
        buffer = b''
        
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    buffer += self.ser.read(self.ser.in_waiting)
                    
                    while len(buffer) >= RECV_PACKET_LEN:
                        # ヘッダ検索
                        idx = buffer.find(HEADER)
                        if idx == -1:
                            buffer = buffer[-RECV_PACKET_LEN:]
                            break
                        if idx > 0:
                            buffer = buffer[idx:]
                        if len(buffer) < RECV_PACKET_LEN:
                            break

                        # パケット抽出
                        packet = buffer[:RECV_PACKET_LEN]
                        buffer = buffer[RECV_PACKET_LEN:]

                        # パースして保存
                        self._process_packet(packet[2:])
                else:
                    # 200Hz (5ms) よりも十分速い周期でチェック
                    time.sleep(0.001) 

            except Exception as e:
                print(f"[Receiver Error] {e}")
                self.running = False

    def _process_packet(self, packet_bytes):
        try:
            data = struct.unpack(RECV_FMT, packet_bytes)
            sample = {
                'timestamp_pc': time.perf_counter(), # PC側着信時刻
                'meas_pres_DF': data[0],
                'meas_pres_F':  data[1],
                'meas_pres_G':  data[2],
                'angle_deg':    data[3],
                'velocity':     data[4],
                'flag':         data[5],
                'force_N':      data[6]
            }
            
            with self.lock:
                self.data_buffer.append(sample) # ログ用に全部保存
                self.latest_sample = sample     # 制御用に最新を更新

        except:
            pass

    def pop_all_buffer(self):
        """ 溜まっているデータを全て取り出し、バッファを空にする """
        with self.lock:
            data = self.data_buffer[:]
            self.data_buffer = []
            return data, self.latest_sample

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
            self.receiver = SensorReceiver(self.ser)
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
        sys.exit(1)

    def run(self):
        self.receiver.start()
        print(f"\n=== STARTING ASYNC CONTROL ===")
        print(f"  Receive: ~200Hz (All Logged)")
        print(f"  Control: 50Hz (Fixed)")
        
        self._send([0,0,0])
        time.sleep(2.0)
        print("GO!")
        
        start_time = time.perf_counter()
        
        try:
            for idx, row in self.cmd_df.iterrows():
                loop_start = time.perf_counter()
                
                # 1. 現在の指令値
                cmd = [row['cmd_pressure_DF'], row['cmd_pressure_F'], row['cmd_pressure_G']]
                
                # 2. 送信 (50Hz)
                self._send(cmd)
                
                # 3. データの回収 (前回のループ以降に溜まった200Hzデータを全て取得)
                #    通常、50Hzループなら 200/50 = 4個程度のデータが返ってくる
                buffer_data, latest = self.receiver.pop_all_buffer()
                
                # 4. ログ保存
                #    回収した200Hzデータ全てに、現在の指令値を紐づけて保存
                if buffer_data:
                    for sample in buffer_data:
                        log_entry = copy.deepcopy(sample)
                        log_entry['time'] = sample['timestamp_pc'] - start_time
                        log_entry['cmd_DF'] = cmd[0] # Zero-Order Holdとして記録
                        log_entry['cmd_F']  = cmd[1]
                        log_entry['cmd_G']  = cmd[2]
                        self.logs.append(log_entry)
                else:
                    # まだデータが来ていない場合（最初期など）
                    pass

                # 5. 50Hz周期維持
                while (time.perf_counter() - loop_start) < CONTROL_DT:
                    pass
                    
        except KeyboardInterrupt:
            print("\nAborted.")
        finally:
            self.shutdown()

    def _send(self, cmd):
        cmd = [max(0.0, min(1.0, v)) for v in cmd]
        self.ser.write(HEADER + struct.pack(SEND_FMT, *cmd))

    def shutdown(self):
        print("Shutting down...")
        self._send([0,0,0])
        self.receiver.stop()
        self.ser.close()
        self.save_logs()

    def save_logs(self):
        if not self.logs: return
        name = f"data_{os.path.splitext(os.path.basename(self.csv_path))[0]}_{int(time.time())}.csv"
        path = os.path.join(os.path.dirname(self.csv_path), name)
        
        # タイムスタンプ順にソート（スレッド競合の微小ズレ補正）
        df = pd.DataFrame(self.logs)
        if not df.empty:
            df = df.sort_values(by='time')
            
        df.to_csv(path, index=False)
        print(f"\n[Saved] High-Res Log (200Hz) saved to:\n  -> {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_name", type=str)
    args = parser.parse_args()
    ExperimentController(args.csv_name).run()