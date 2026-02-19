"""
Porcaro Robot: IROS 2026 Validation Experiment Runner (Async: Recv 200Hz / Send 50Hz)
Target: Jetson Orin Nano + MicroLabBox

python IROS/run_iros_validation.py exp1_static_hysteresis.csv
python IROS/run_iros_validation.py exp2_step_response.csv
python IROS/run_iros_validation.py exp3_frequency_sweep.csv
python IROS/run_iros_validation.py exp4_drumming_task.csv
python IROS/run_iros_validation.py exp5_amplitude_sweep.csv
python IROS/run_iros_validation.py exp6_duration_sweep.csv
python IROS/run_iros_validation.py exp7_stiffness_sweep.csv
python IROS/run_iros_validation.py exp8_speed_sweep.csv

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
BAUD_RATE = 230400
CONTROL_DT = 0.02        # 制御・送信は50Hz

SEND_FMT = '>ddd'        # [Cmd_DF, Cmd_F, Cmd_G]
RECV_FMT = '>ddddddd'    # [Meas_DF, Meas_F, Meas_G, Angle, Vel, Flag, Force]
HEADER = b'\xff\xff'
RECV_PACKET_LEN = 2 + 8 * 7

class SensorReceiver(threading.Thread):
    def __init__(self, ser):
        super().__init__()
        self.ser = ser
        self.running = True
        self.sensor_logs = []
        self.clear_flag = False
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
                        idx = buffer.find(HEADER)
                        if idx == -1:
                            buffer = buffer[-1:]
                            break
                        if idx > 0:
                            buffer = buffer[idx:]
                        if len(buffer) < RECV_PACKET_LEN:
                            break

                        packet = buffer[:RECV_PACKET_LEN]
                        buffer = buffer[RECV_PACKET_LEN:]
                        self._process_packet(packet[2:])
                else:
                    time.sleep(0.001) 
            except Exception as e:
                print(f"[Receiver Error] {e}")
                self.running = False

    def _process_packet(self, packet_bytes):
        try:
            data = struct.unpack(RECV_FMT, packet_bytes)
            sample = {
                'timestamp_pc': time.perf_counter(), # 参考（USBジッタ確認用）
                'meas_pres_DF': data[0],
                'meas_pres_F':  data[1],
                'meas_pres_G':  data[2],
                'angle_deg':    data[3],
                'velocity':     data[4],
                'flag':         data[5],
                'force_N':      data[6]
            }
            
            with self.lock:
                # 同期フラグが立ったら過去のゴミデータを一掃する
                if self.clear_flag:
                    self.sensor_logs = []
                    self.clear_flag = False
                self.sensor_logs.append(sample)
        except:
            pass

    def clear_buffer_for_sync(self):
        """ GOの瞬間にUSBバッファとログリストを完全に空にして時刻0を合わせる """
        self.ser.reset_input_buffer()
        with self.lock:
            self.clear_flag = True

    def get_all_logs(self):
        with self.lock:
            return self.sensor_logs[:]

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
            self.cmd_logs = [] # 指令値専用の履歴
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
        
        # 姿勢を初期化して安定するまで待機
        self._send([0,0,0])
        time.sleep(2.0)
        
        print("GO! (Synchronizing Clocks...)")
        # 変更箇所: ここの瞬間にバッファを吹き飛ばし、最初の受信データを厳密に t=0 とする
        self.receiver.clear_buffer_for_sync()
        
        try:
            for idx, row in self.cmd_df.iterrows():
                loop_start = time.perf_counter()
                
                cmd = [row['cmd_pressure_DF'], row['cmd_pressure_F'], row['cmd_pressure_G']]
                self._send(cmd)
                
                # 変更箇所: 「コマンドを送信した数学的な理想時刻(0.02s刻み)」を個別に記録
                self.cmd_logs.append({
                    'cmd_time': idx * CONTROL_DT, 
                    'cmd_DF': cmd[0],
                    'cmd_F': cmd[1],
                    'cmd_G': cmd[2]
                })

                # 50Hz周期維持
                while (time.perf_counter() - loop_start) < CONTROL_DT:
                    time.sleep(0.0005)
                    
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
        sensor_data = self.receiver.get_all_logs()
        if not sensor_data:
            print("[Error] No sensor data received.")
            return

        # 1. センサーデータ (200Hz) の再構築
        df_sensor = pd.DataFrame(sensor_data)
        # USBのブレを無視し、MicroLabBoxの完璧な200Hz(0.005s)クロックを信用して時刻を生成
        df_sensor['time'] = np.arange(len(df_sensor)) * 0.005

        # 2. 指令値データ (50Hz)
        df_cmd = pd.DataFrame(self.cmd_logs)
        df_cmd = df_cmd.rename(columns={'cmd_time': 'time'})

        # 3. センサーの各時刻(time)に対し、その瞬間にアクティブだった直近の指令値を合成 (Zero-Order Hold)
        df_merged = pd.merge_asof(
            df_sensor,
            df_cmd,
            on='time',
            direction='backward'
        )

        name = f"data_{os.path.splitext(os.path.basename(self.csv_path))[0]}_{int(time.time())}.csv"
        path = os.path.join(os.path.dirname(self.csv_path), name)
        df_merged.to_csv(path, index=False)
        print(f"\n[Saved] Reconstructed High-Res Log (200Hz) saved to:\n  -> {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_name", type=str)
    args = parser.parse_args()
    ExperimentController(args.csv_name).run()