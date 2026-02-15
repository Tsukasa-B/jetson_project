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
# 4. Exp-4: Drumming Task (Calibration Focus) -- 1.0s HIT Ver. --
# ==========================================
# 目的: 高周波特性による乖離を避け、静的な「力・変位」の関係を正確にキャリブレーションする。
# 1秒間の加圧により、空気圧が目標値(0.6MPa)に完全に到達した状態での挙動を確認。

p_df_exp4 = np.zeros_like(TIME)
p_f_exp4  = np.zeros_like(TIME)

interval = 5.0      # 5秒サイクル
hit_duration = 1.0  # 変更箇所: 1.0秒間振り下ろし(HIT)を継続。

for i, t in enumerate(TIME):
    phase = t % interval
    
    if phase < hit_duration:
        # --- HIT / PRESS (1.0秒間) ---
        # 底屈(F)を最大にして打面に押し付ける。
        # 1秒あればバルブの遅延やチューブ内の圧力伝播の影響が消え、
        # シミュレーション上の「Model B」の静的な剛性と比較可能。
        p_df_exp4[i] = 0.0 
        p_f_exp4[i]  = 0.6 
    else:
        # --- RETRACT / HOLD (4.0秒間) ---
        # 4秒間かけてゆっくり、かつ確実に元の位置へ戻す。
        p_df_exp4[i] = 0.4 
        p_f_exp4[i]  = 0.1 

save_csv("exp4_drumming_slow_1s", p_df_exp4, p_f_exp4)