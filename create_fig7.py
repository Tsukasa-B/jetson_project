import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy.signal import find_peaks

# --- IROS (IEEE) 論文用プロット設定 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

TASKS = [
    {"label": "single4\n(60)", "key": "test_single4_bpm60"},
    {"label": "single8\n(120)", "key": "test_single8_bpm120"},
    {"label": "gmd\n(105)", "key": "gmd_02_mid_bpm105[*]"}
]

FORCE_THRESHOLD = 1.0 
MATCH_WINDOW_SEC = 0.150 

def find_file(directory, keyword):
    files = glob.glob(os.path.join(directory, f"*{keyword}*.csv"))
    return files[0] if files else None

def calc_timing_errors(csv_path, is_real):
    if not csv_path or not os.path.exists(csv_path):
        return []
        
    df = pd.read_csv(csv_path)
    
    time_col = 'time' if is_real else 'time_s'
    force_col = 'force_N' if is_real else 'force_z'
    
    if time_col not in df.columns or force_col not in df.columns or 'target_force' not in df.columns:
        return []

    t = df[time_col].values
    f_actual = df[force_col].values
    f_target = df['target_force'].values

    target_edges = np.where(np.diff(f_target) > 0.5)[0]
    target_times = t[target_edges]

    actual_peaks, _ = find_peaks(f_actual, height=FORCE_THRESHOLD, distance=10)
    actual_times = t[actual_peaks]

    errors_ms = []
    
    for tt in target_times:
        if len(actual_times) == 0:
            break
        
        diffs = actual_times - tt
        abs_diffs = np.abs(diffs)
        min_idx = np.argmin(abs_diffs)
        
        if abs_diffs[min_idx] <= MATCH_WINDOW_SEC:
            errors_ms.append(diffs[min_idx] * 1000.0)

    return errors_ms

def generate_data():
    records = []
    
    conditions = [
        {"env": "Sim", "model": "Baseline (w/o DR)", "path": "../LSTMDRなし/Sim_experiments", "is_real": False},
        {"env": "Sim", "model": "Proposed (w/ DR)", "path": "../LSTMDR/Sim_experiments", "is_real": False},
        {"env": "Real", "model": "Baseline (w/o DR)", "path": "deploy_results", "is_real": True},
        {"env": "Real", "model": "Proposed (w/ DR)", "path": "../LSTMDR/Real_experiments", "is_real": True}
    ]

    for task in TASKS:
        for c in conditions:
            file_path = find_file(c["path"], task["key"])
            errors = calc_timing_errors(file_path, c["is_real"])
            
            category_name = f"{task['label']}\n[{c['env']}]"
            
            for err in errors:
                records.append({
                    "Category": category_name,
                    "Model": c["model"],
                    "Timing Error (ms)": err,
                    "Env": c["env"] # 色分け用に環境情報を追加
                })
                
    return pd.DataFrame(records)

def plot_fig7(df, output_path):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    
    # 許容誤差の背景帯とゼロライン
    ax.axhspan(-30, 30, color='gray', alpha=0.15, label='Acceptable Human Groove (\u00B130ms)', zorder=0)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.8, zorder=1)

    # 順番の指定
    order = []
    for t in TASKS:
        order.append(f"{t['label']}\n[Sim]")
        order.append(f"{t['label']}\n[Real]")

    # --- 色の定義 (Sim用とReal用で濃淡を変える) ---
    pal_sim = {
        "Baseline (w/o DR)": "#FFA0A0", # 薄い赤
        "Proposed (w/ DR)": "#80BFFF"   # 薄い青
    }
    pal_real = {
        "Baseline (w/o DR)": "#D55E00", # 濃い赤 (朱色)
        "Proposed (w/ DR)": "#0072B2"   # 濃い青
    }

    # 1. Simのデータを描画
    sns.violinplot(
        data=df[df['Env'] == 'Sim'], 
        x="Category", y="Timing Error (ms)", hue="Model", 
        order=order, split=True, inner="quartile",
        palette=pal_sim, linewidth=1.0, ax=ax, zorder=2
    )
    
    # 2. Realのデータを描画 (上書き)
    sns.violinplot(
        data=df[df['Env'] == 'Real'], 
        x="Category", y="Timing Error (ms)", hue="Model", 
        order=order, split=True, inner="quartile",
        palette=pal_real, linewidth=1.0, ax=ax, zorder=2
    )

    ax.set_ylabel('Timing Error (ms)')
    ax.set_xlabel('')
    ax.set_title('Sim-to-Real Gap of Strike Timing Error on Unseen MIDI Data', pad=15)
    
    # --- 凡例のカスタマイズ ---
    # Seabornが複数回呼ばれて凡例が重複するため、手動で綺麗な凡例を作り直す
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    custom_lines = [
        Patch(facecolor='gray', alpha=0.15, label='Acceptable Human Groove (\u00B130ms)'),
        Patch(facecolor=pal_real["Baseline (w/o DR)"], edgecolor='black', linewidth=0.5, label='Baseline (w/o DR)'),
        Patch(facecolor=pal_real["Proposed (w/ DR)"], edgecolor='black', linewidth=0.5, label='Proposed (w/ DR)'),
        Patch(facecolor=pal_sim["Proposed (w/ DR)"], edgecolor='black', linewidth=0.5, label='(Sim data is plotted in lighter colors)')
    ]
    ax.legend(handles=custom_lines, loc='lower left', framealpha=0.9, edgecolor='#808080', fontsize=9)

    # X軸の区切りをわかりやすくするための縦線
    for i in [1.5, 3.5]:
        ax.axvline(i, color='#e0e0e0', linestyle='-', linewidth=1.5, zorder=0)

    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"Fig.7 を保存しました: {output_path}")

if __name__ == "__main__":
    df = generate_data()
    if df.empty:
        print("エラー: データが抽出できませんでした。")
    else:
        plot_fig7(df, "Fig7_Sim2Real_Timing_Error_Colored.svg")