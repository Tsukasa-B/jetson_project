"""
run_experiment_batch.py — 実機実験のバッチ実行（条件×モデル×seed×trial を順に回す）

手で `deploy_policy.py` を100回叩くと、必ずどこかで条件を取り違える。
このスクリプトは実験計画(YAML)を読んで順に実行し、
  * 実行前に必ず内容を表示して確認を取る（--yes で省略可）
  * 各ランの前に人の合図(ENTER)を待つ。コンプレッサ圧の回復待ちに使う
  * 途中で止めても、既に完了したランは results/ に残る（--resume で続きから）
  * 実行順をシャッフルできる（--shuffle。リグの経時変化がモデル順と交絡するのを防ぐ）

Usage:
  python3 tools/run_experiment_batch.py --plan tools/experiment_plan.yaml --dry_run
  python3 tools/run_experiment_batch.py --plan tools/experiment_plan.yaml --shuffle
  python3 tools/run_experiment_batch.py --plan tools/experiment_plan.yaml --resume
"""

from __future__ import annotations

import argparse
import glob
import itertools
import os
import random
import subprocess
import sys
import time

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_plan(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_jobs(plan: dict) -> list[dict]:
    jobs = []
    for block in plan["blocks"]:
        models = block["models"]
        songs = block["songs"]
        trials = int(block.get("trials", 1))
        for model, song, trial in itertools.product(models, songs, range(1, trials + 1)):
            jobs.append({"model": model, "song": song, "trial": trial,
                         "block": block.get("name", "")})
    return jobs


def already_done(job: dict) -> bool:
    """results/ に同じ(model, song, trial)のCSVが既にあるか。"""
    group, name = job["model"].split("/")
    stem = os.path.splitext(os.path.basename(job["song"]))[0]
    pattern = os.path.join(REPO_ROOT, "results", group, f"model{name}",
                           f"deploy_{stem}_{group}-{name}_trial{job['trial']:02d}_*.csv")
    return bool(glob.glob(pattern))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=os.path.join(REPO_ROOT, "tools", "experiment_plan.yaml"))
    ap.add_argument("--dry_run", action="store_true", help="実行せず一覧だけ表示")
    ap.add_argument("--resume", action="store_true", help="既に結果があるランを飛ばす")
    ap.add_argument("--shuffle", action="store_true", help="実行順をランダム化（推奨）")
    ap.add_argument("--seed", type=int, default=0, help="シャッフルの乱数種（再現性のため記録）")
    ap.add_argument("--yes", action="store_true", help="各ラン前のENTER待ちをしない")
    ap.add_argument("--force_scale", type=float, default=None)
    ap.add_argument("--swap_encoders", action="store_true",
                    help="手首/ハンド関節エンコーダの配線が逆のとき指定（全ランに適用）")
    args = ap.parse_args()

    plan = load_plan(args.plan)
    jobs = build_jobs(plan)

    if args.shuffle:
        random.Random(args.seed).shuffle(jobs)
        print(f"[batch] 実行順をシャッフルしました (seed={args.seed})")

    if args.resume:
        before = len(jobs)
        jobs = [j for j in jobs if not already_done(j)]
        print(f"[batch] resume: {before - len(jobs)} 件は完了済みとしてスキップ")

    print(f"\n=== 実験計画: {plan.get('name', '(無名)')} ===")
    print(f"    総ラン数: {len(jobs)}")
    est_min = len(jobs) * plan.get("est_sec_per_run", 60) / 60
    print(f"    所要目安: 約 {est_min:.0f} 分（1ラン {plan.get('est_sec_per_run', 60)}秒想定）\n")
    for i, j in enumerate(jobs, 1):
        print(f"  {i:3d}. {j['model']:<16s} {os.path.basename(j['song']):<28s} trial{j['trial']}")

    if args.dry_run:
        print("\n=== --dry_run のため実行しません ===")
        return 0

    print("\nこの内容で実行しますか？ 続けるなら ENTER、中止は Ctrl+C")
    input()

    ok, ng = 0, []
    for i, j in enumerate(jobs, 1):
        print(f"\n{'='*70}")
        print(f" [{i}/{len(jobs)}] {j['model']} | {os.path.basename(j['song'])} | trial{j['trial']}")
        print(f"{'='*70}")
        if not args.yes:
            print(">>> ロボットとコンプレッサの状態を確認してから ENTER（スキップは s + ENTER）")
            if input().strip().lower() == "s":
                print("    スキップしました")
                continue

        cmd = [sys.executable, "src/deploy_policy.py",
               "--model", j["model"], "--midi", j["song"],
               "--trial", str(j["trial"]), "--no-input"]
        if args.force_scale is not None:
            cmd += ["--force_scale", str(args.force_scale)]
        if args.swap_encoders:
            cmd += ["--swap_encoders"]

        r = subprocess.run(cmd, cwd=REPO_ROOT)
        if r.returncode == 0:
            ok += 1
        else:
            ng.append(j)
            print(f"[batch] FAILED (returncode={r.returncode})")
        time.sleep(plan.get("cooldown_sec", 2))

    print(f"\n=== 完了: 成功 {ok} / 失敗 {len(ng)} ===")
    for j in ng:
        print(f"  FAILED: {j['model']} {j['song']} trial{j['trial']}")
    print("\n集計:  python3 analysis/strike_metrics.py results/RAL --summary results/RAL/summary.csv")
    return 0 if not ng else 1


if __name__ == "__main__":
    sys.exit(main())
