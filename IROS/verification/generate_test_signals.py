'''
generate_test_signals.py (v2)
Target: Generate Exp1-8 test signals for Sim-to-Real Calibration.
Usage: python IROS/verification/generate_test_signals.py
'''
import numpy as np
import pandas as pd
from scipy.signal import chirp
import os

# --- 設定 ---
OUTPUT_DIR = "IROS/test_signals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 変更箇所: 高速動作(Exp6,8)に対応するため分解能を向上
DT = 0.02 
DURATION = 20.0
TIME = np.arange(0, DURATION, DT)

# 共通: Gripは常に0.5MPa
P_GRIP = 0.5

def save_csv(filename, p_df, p_f):
    p_g = np.full_like(TIME, P_GRIP)
    p_df = np.clip(p_df, 0.0, 0.6)
    p_f  = np.clip(p_f, 0.0, 0.6)
    
    df = pd.DataFrame({
        'time': TIME,
        'cmd_pressure_DF': p_df,
        'cmd_pressure_F':  p_f,
        'cmd_pressure_G':  p_g
    })
    path = os.path.join(OUTPUT_DIR, f"{filename}.csv")
    df.to_csv(path, index=False)
    print(f"Generated: {path}")

# ==========================================
# Exp 1-4 (Existing)
# ==========================================
# Exp-1: Static Hysteresis
freq = 0.05
p_df_e1 = 0.3 * np.sin(2 * np.pi * freq * TIME - np.pi/2) + 0.3
p_f_e1  = np.full_like(TIME, 0.2)
save_csv("exp1_static_hysteresis", p_df_e1, p_f_e1)

# Exp-2: Step Response
p_df_e2, p_f_e2 = np.zeros_like(TIME), np.zeros_like(TIME)
for i, t in enumerate(TIME):
    if int(t // 4) % 2 == 0:
        p_df_e2[i], p_f_e2[i] = 0.6, 0.1
    else:
        p_df_e2[i], p_f_e2[i] = 0.1, 0.6
save_csv("exp2_step_response", p_df_e2, p_f_e2)

# Exp-3: Frequency Sweep
base_wave = chirp(TIME, f0=0.1, f1=5.0, t1=DURATION, method='linear')
p_df_e3 = 0.3 + 0.2 * base_wave
p_f_e3  = 0.3 + 0.2 * base_wave * (-1)
save_csv("exp3_frequency_sweep", p_df_e3, p_f_e3)

# Exp-4: Drumming Task (Slow & Heavy)
p_df_e4, p_f_e4 = np.zeros_like(TIME), np.zeros_like(TIME)
for i, t in enumerate(TIME):
    if (t % 5.0) < 1.0: # 1.0s Hit
        p_df_e4[i], p_f_e4[i] = 0.0, 0.6
    else:
        p_df_e4[i], p_f_e4[i] = 0.4, 0.1
save_csv("exp4_drumming_task", p_df_e4, p_f_e4)

# ==========================================
# Exp 5-8 (New Calibration Tasks)
# ==========================================

# Exp-5: Amplitude Sweep (力と圧力の線形性確認)
# 0.2MPa -> 0.4MPa -> 0.6MPa と強さを変える
p_df_e5, p_f_e5 = np.zeros_like(TIME), np.zeros_like(TIME)
for i, t in enumerate(TIME):
    # 5秒ごとに強度切り替え
    if   t < 5.0:  amp = 0.2
    elif t < 10.0: amp = 0.4
    else:          amp = 0.6
    
    # 2.5秒周期で打撃
    if (t % 2.5) < 0.5: # 0.5s Hit
        p_df_e5[i] = 0.0
        p_f_e5[i]  = amp
    else:
        p_df_e5[i] = 0.4
        p_f_e5[i]  = 0.1
save_csv("exp5_amplitude_sweep", p_df_e5, p_f_e5)

# Exp-6: Duration Sweep (バルブ応答・デッドタイム確認)
# 0.05s(Implus) -> 0.15s -> 0.3s と長さを変える
p_df_e6, p_f_e6 = np.zeros_like(TIME), np.zeros_like(TIME)
durations = [0.05, 0.15, 0.3]
for i, t in enumerate(TIME):
    idx = int(t // 5.0) % 3
    dur = durations[idx]
    
    if (t % 2.5) < dur:
        p_df_e6[i], p_f_e6[i] = 0.0, 0.6 # Full Power
    else:
        p_df_e6[i], p_f_e6[i] = 0.4, 0.1
save_csv("exp6_duration_sweep", p_df_e6, p_f_e6)

# Exp-7: Stiffness Sweep (拮抗状態での挙動確認)
# DF(背屈)圧力を 0.0 -> 0.2 -> 0.4 と変えて「バネ性」を見る
p_df_e7, p_f_e7 = np.zeros_like(TIME), np.zeros_like(TIME)
for i, t in enumerate(TIME):
    # 5秒ごとに拮抗圧切り替え
    if   t < 5.0:  stiff = 0.0
    elif t < 10.0: stiff = 0.2
    else:          stiff = 0.4
    
    if (t % 2.5) < 0.5:
        p_df_e7[i] = stiff # 打撃時も拮抗圧を残す
        p_f_e7[i]  = 0.6   # Full Power
    else:
        p_df_e7[i] = 0.4
        p_f_e7[i]  = 0.1
save_csv("exp7_stiffness_sweep", p_df_e7, p_f_e7)

# Exp-8: Speed Sweep (連打時のエア供給・ダイナミクス確認)
# Interval 1.0s(60BPM) -> 0.5s(120BPM) -> 0.25s(240BPM)
p_df_e8, p_f_e8 = np.zeros_like(TIME), np.zeros_like(TIME)
intervals = [1.0, 0.5, 0.25]
for i, t in enumerate(TIME):
    idx = int(t // 5.0) % 3
    inv = intervals[idx]
    
    # 50% Duty Cycle
    if (t % inv) < (inv * 0.5):
        p_df_e8[i], p_f_e8[i] = 0.0, 0.6
    else:
        p_df_e8[i], p_f_e8[i] = 0.4, 0.1
save_csv("exp8_speed_sweep", p_df_e8, p_f_e8)