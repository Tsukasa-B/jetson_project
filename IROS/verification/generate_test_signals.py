'''
generate_test_signals.py
python IROS/verification/generate_test_signals.py
'''
import numpy as np
import pandas as pd
from scipy.signal import chirp
import os

# --- 設定 ---
OUTPUT_DIR = "IROS/test_signals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DT = 0.02
DURATION = 20.0
TIME = np.arange(0, DURATION, DT)

# 共通: Gripは常に0.5MPa (スティック保持)
P_GRIP = 0.5

def save_csv(filename, p_df, p_f):
    # p_g は固定値の配列を作成
    p_g = np.full_like(TIME, P_GRIP)
    
    # 安全リミット (0.0 ~ 0.6)
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
# 1. Exp-1: Static Hysteresis (Isobaric Antagonist)
# ==========================================
# 目的: Fを一定張力(0.2)に保ち、DFだけをゆっくり動かして純粋な特性を見る
freq = 0.05
p_df_exp1 = 0.3 * np.sin(2 * np.pi * freq * TIME - np.pi/2) + 0.3 # 0.0 -> 0.6 -> 0.0
p_f_exp1  = np.full_like(TIME, 0.2) # 拮抗側は弱く引いておく(Slack防止)
save_csv("exp1_static_hysteresis", p_df_exp1, p_f_exp1)

# ==========================================
# 2. Exp-2: Step Response (Full Switching)
# ==========================================
# 目的: DF=0.6/F=0.0 (上) と DF=0.0/F=0.6 (下) を全力で切り替え
p_df_exp2 = np.zeros_like(TIME)
p_f_exp2  = np.zeros_like(TIME)

# 4秒ごとに切り替え
for i, t in enumerate(TIME):
    cycle = int(t // 4)
    if cycle % 2 == 0:
        # State A: Up (DF pull)
        p_df_exp2[i] = 0.6
        p_f_exp2[i]  = 0.1 # 完全に0にすると外れる恐れがあるため0.1残す
    else:
        # State B: Down (F pull)
        p_df_exp2[i] = 0.1
        p_f_exp2[i]  = 0.6

save_csv("exp2_step_response", p_df_exp2, p_f_exp2)

# ==========================================
# 3. Exp-3: Frequency Sweep (Antagonistic Chirp)
# ==========================================
# 目的: 中心圧0.3MPaで、互いに逆位相で振動させる
f0, f1 = 0.1, 5.0
amp = 0.2
offset = 0.3

# Linear Chirp Sine
base_wave = chirp(TIME, f0=f0, f1=f1, t1=DURATION, method='linear')

p_df_exp3 = offset + amp * base_wave        # 0.3 ± 0.2
p_f_exp3  = offset + amp * base_wave * (-1) # 逆位相 (180度ズレ)

save_csv("exp3_frequency_sweep", p_df_exp3, p_f_exp3)

# ==========================================
# 4. Exp-4: Drumming Task (Slow & Heavy) -- 修正箇所 --
# ==========================================
# 目的: 確実に叩くためにBPMを落とし、Hit時間を長くする
p_df_exp4 = np.zeros_like(TIME)
p_f_exp4  = np.zeros_like(TIME)

bpm = 60          # BPM 60 (1秒に1回) に変更
interval = 60 / bpm 
hit_duration = 0.20 # 200ms (0.2秒) まで延長。空気圧が追従する時間を確保。

for i, t in enumerate(TIME):
    # 基本サイクル内での位置
    phase = t % interval
    
    if phase < hit_duration:
        # HIT Action (振り下ろし & 押し付け)
        p_df_exp4[i] = 0.0 # 背屈を脱力
        p_f_exp4[i]  = 0.6 # 底屈を最大加圧
    else:
        # RETRACT / HOLD (構え & 戻り)
        p_df_exp4[i] = 0.4 # 持ち上げる
        p_f_exp4[i]  = 0.1 # テンション維持

save_csv("exp4_drumming_task", p_df_exp4, p_f_exp4)