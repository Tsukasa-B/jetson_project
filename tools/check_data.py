import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_data(csv_file):
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} samples from {csv_file}")
    
    # タイムスタンプを0スタートに
    t = df["timestamp"] - df["timestamp"].iloc[0]
    
    # 3つの筋肉をプロット
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    muscles = ["DF", "F", "G"]
    
    for i, muscle in enumerate(muscles):
        ax = axes[i]
        # 指令値 (Target)
        ax.plot(t, df[f"cmd_{muscle}"], label="Command (Target)", color="blue", linestyle="--", alpha=0.7)
        # 実測値 (Measured)
        ax.plot(t, df[f"meas_pres_{muscle}"], label="Measured (Real)", color="darkorange", linewidth=1.5)
        
        ax.set_title(f"Muscle {muscle} Response")
        ax.set_ylabel("Pressure [MPa]")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend(loc="upper right")
        ax.set_ylim(-0.05, 0.65)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    
    # 画像として保存
    out_png = csv_file.replace(".csv", ".png")
    plt.savefig(out_png)
    print(f"Plot saved to: {out_png}")
    # GUI環境なら表示
    # plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 check_data.py <csv_filename>")
    else:
        plot_data(sys.argv[1])