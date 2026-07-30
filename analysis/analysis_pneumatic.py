"""
空気圧人工筋肉のステップ応答解析スクリプト
- むだ時間 (Deadtime) と時定数 (Tau) の2Dヒートマップ生成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 設定
# ==========================================
# 変更箇所: 解析対象のCSVファイル名を指定
CSV_FILENAME = "data_characteristics_exp1_async_1771481712.csv" 
LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
NUM_LEVELS = len(LEVELS)

def analyze_step_response():
    if not os.path.exists(CSV_FILENAME):
        print(f"[Error] File not found: {CSV_FILENAME}")
        return

    print(f"Loading data from {CSV_FILENAME}...")
    df = pd.read_csv(CSV_FILENAME)
    
    # 指令値が変化したエッジを検出
    # diff()が0以外の箇所がステップ入力のタイミング
    df['cmd_F_diff'] = df['cmd_F'].diff()
    # NaNを埋めてから判定
    step_indices = df.index[(df['cmd_F_diff'].abs() > 0.05)].tolist()

    # 2Dテーブルの初期化 (自己遷移部分は便宜上 0.0010 とする)
    tau_table = np.full((NUM_LEVELS, NUM_LEVELS), 0.0010)
    dead_table = np.full((NUM_LEVELS, NUM_LEVELS), 0.0010)

    print(f"Detected {len(step_indices)} step transitions.")

    # 各ステップ応答の解析
    for i in range(len(step_indices)):
        start_idx_df = step_indices[i]
        
        # 次のステップが来るまで、あるいはデータの最後までを切り出す
        end_idx_df = step_indices[i+1] if i + 1 < len(step_indices) else len(df) - 1
        
        # 遷移前の設定値(Start)と遷移後の設定値(Target)
        start_cmd = df.loc[start_idx_df - 1, 'cmd_F']
        target_cmd = df.loc[start_idx_df, 'cmd_F']
        
        # テーブルのインデックスを算出
        start_i = int(round(start_cmd * 10))
        target_j = int(round(target_cmd * 10))
        
        # 範囲外のノイズを無視
        if start_i >= NUM_LEVELS or target_j >= NUM_LEVELS or start_i == target_j:
            continue

        # 解析用データの切り出し (ステップ入力時から約3秒間)
        # 200Hz想定なので、200 * 3 = 600サンプル
        analyze_len = min(600, end_idx_df - start_idx_df)
        df_step = df.iloc[start_idx_df : start_idx_df + analyze_len].reset_index(drop=True)
        
        t = df_step['time'].values
        t = t - t[0] # ステップ入力時をt=0とする
        p = df_step['meas_pres_F'].values
        
        p_start = p[0]
        # 定常状態の圧力は最後の数サンプルの平均とする
        p_target_actual = np.mean(p[-20:]) 
        p_diff = p_target_actual - p_start
        
        if abs(p_diff) < 0.02:
            continue # 圧力変化が小さすぎる場合はスキップ
            
        # むだ時間 (Deadtime): 圧力変化が目標変動幅の 5% に達した時間
        threshold_dead = p_start + 0.05 * p_diff
        
        # 時定数 (Tau): 圧力変化が目標変動幅の 63.2% に達した時間
        threshold_tau = p_start + 0.632 * p_diff

        dead_time = 0.0010
        tau_time = 0.0010
        
        # しきい値を超えるインデックスを探索
        if p_diff > 0: # 昇圧時
            dead_idx = np.argmax(p >= threshold_dead)
            tau_idx = np.argmax(p >= threshold_tau)
        else:          # 降圧時
            dead_idx = np.argmax(p <= threshold_dead)
            tau_idx = np.argmax(p <= threshold_tau)
            
        if dead_idx > 0:
            dead_time = t[dead_idx]
        if tau_idx > 0:
            # むだ時間を除いた純粋な時定数
            tau_time = max(0.0010, t[tau_idx] - dead_time)

        # テーブルへ格納
        dead_table[start_i, target_j] = dead_time
        tau_table[start_i, target_j] = tau_time

    # ==========================================
    # 結果の出力
    # ==========================================
    # 1. コンソール出力 (Isaac Labの pneumatic.py コピペ用)
    print("\n# --- For Isaac Lab (pneumatic.py) ---")
    print("TAU_TABLE_2D_DATA = [")
    for row in tau_table:
        print("    [" + ", ".join([f"{val:.4f}" for val in row]) + "],")
    print("]")

    print("\nDEAD_TABLE_2D_DATA = [")
    for row in dead_table:
        print("    [" + ", ".join([f"{val:.4f}" for val in row]) + "],")
    print("]")
    print("# -------------------------------------\n")

    # 2. ヒートマップ描画 (Seaborn)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(tau_table, annot=True, fmt=".4f", cmap="YlOrRd", 
                xticklabels=LEVELS, yticklabels=LEVELS, ax=axes[0])
    axes[0].set_title("Tau (Time Constant) [s]")
    axes[0].set_xlabel("Target Pressure [MPa]")
    axes[0].set_ylabel("Start Pressure [MPa]")

    sns.heatmap(dead_table, annot=True, fmt=".4f", cmap="YlGnBu", 
                xticklabels=LEVELS, yticklabels=LEVELS, ax=axes[1])
    axes[1].set_title("Deadtime [s]")
    axes[1].set_xlabel("Target Pressure [MPa]")
    axes[1].set_ylabel("Start Pressure [MPa]")

    plt.tight_layout()
    plt.savefig("dynamics_heatmap_recalc.png")
    print("[Saved] Heatmap image saved as 'dynamics_heatmap_recalc.png'")
    plt.show()

if __name__ == "__main__":
    analyze_step_response()