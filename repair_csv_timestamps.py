import os
import glob
import pandas as pd
import numpy as np

# ==========================================
# 設定
# ==========================================
# 修正対象のディレクトリ
DATA_DIR = "./external_data/jetson_project"

# 出力先（混ざらないように別フォルダ推奨）
OUTPUT_DIR = "./external_data/jetson_project/corrected"

def repair_csv(file_path):
    try:
        # 1. 読み込み
        df = pd.read_csv(file_path)
        
        # 必要なカラムがあるか確認
        if 'meas_pres_DF' not in df.columns:
            print(f"[SKIP] {os.path.basename(file_path)} (Not a data log)")
            return

        original_len = len(df)
        original_duration = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0] if 'timestamp' in df.columns else 0
        
        # ----------------------------------------------------
        # 2. 時刻再構成 (Repair Logic)
        # ----------------------------------------------------
        # 行インデックスこそが真の 1ms 刻みであるとみなす
        # time = Index * 0.001
        df['time'] = df.index * 0.001
        
        # 元の timestamp は混乱を招くので rename して残すか削除
        if 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'timestamp_original_bad'}, inplace=True)

        new_duration = df['time'].iloc[-1]
        
        # ----------------------------------------------------
        # 3. ダウンサンプリング (1kHz -> 100Hz)
        # ----------------------------------------------------
        # Sim-to-Real検定用 (10行に1行)
        df_100hz = df.iloc[::10].reset_index(drop=True)
        
        # ----------------------------------------------------
        # 4. 保存
        # ----------------------------------------------------
        base_name = os.path.basename(file_path)
        save_path = os.path.join(OUTPUT_DIR, base_name)
        
        df_100hz.to_csv(save_path, index=False)
        
        print(f"[FIXED] {base_name}")
        print(f"  - Rows: {original_len} -> {len(df_100hz)} (100Hz)")
        print(f"  - Duration: Old={original_duration:.2f}s -> New={new_duration:.2f}s")
        
        # 警告: もし時間が極端にズレていたら通知 (例: 115200bps制限でパケット落ちしていた場合など)
        if abs(new_duration - original_duration) > 5.0 and original_duration > 1.0:
            print(f"  [WARNING] Large duration mismatch! Data might be lost or baud-limited.")

    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] Created output directory: {OUTPUT_DIR}")

    # 対象ファイルを検索 (Exp1, 2, 3, 4 全て)
    # パターン: data_exp*.csv
    search_pattern = os.path.join(DATA_DIR, "data_exp*.csv")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"[WARN] No files found in {search_pattern}")
        return

    print(f"Found {len(files)} files. Starting repair...\n")
    
    for csv_file in files:
        repair_csv(csv_file)
        
    print("\n" + "="*50)
    print(" Repair Complete!")
    print(f" Corrected files are in: {OUTPUT_DIR}")
    print(" Use these files for 'analyze_sim_real.py' etc.")
    print("="*50)

if __name__ == "__main__":
    main()