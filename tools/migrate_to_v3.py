#!/usr/bin/env python3
"""
tools/migrate_to_v3.py — jetson_project ディレクトリ再編 v3

全ファイルのパスをルートから3階層以内に収め、「研究の実機ログ」
「IROS投稿時点の凍結アーカイブ」「OCデモ」「共通コード」を分離する。
tools/migrate_to_v2.sh (bash) の後継。Windows/Linux両方で動かすため
Pythonで書く。ファイルは git mv で移動するだけで、削除・中身の書き換えは
一切行わない。

既定は dry-run（移動表を表示するだけ）。--apply で実際に git mv を実行する。

Usage:
  python tools/migrate_to_v3.py            # dry-run
  python tools/migrate_to_v3.py --apply    # 実行
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(*args, check=True):
    return subprocess.run(list(args), cwd=REPO_ROOT, check=check,
                           capture_output=True, text=True)


def git_ls_files():
    out = sh("git", "ls-files").stdout
    return [line for line in out.splitlines() if line]


def check_clean_tree():
    out = sh("git", "status", "--porcelain").stdout
    if out.strip():
        print("ERROR: 作業ツリーに未コミットの変更があります。先にcommitかstashしてください。",
              file=sys.stderr)
        sys.exit(1)


# =============================================================================
# 移動ルール（すべて静的）
# =============================================================================

# ルート直下の個別ファイル
ROOT_FILE_MOVES = {
    "OC_DEMO_WORKORDER.md": "oc_demo/docs/OC_DEMO_WORKORDER.md",
    "shot_8_screen1.png": "oc_demo/docs/shot_8_screen1.png",
    "oc_demo_package.tar.gz": "oc_demo/dist/oc_demo_package.tar.gz",
    "oc_demo_package_v3.tar.gz": "oc_demo/dist/oc_demo_package_v3.tar.gz",
}

# ディレクトリ丸ごとリネーム（内部のサブ構造はそのまま保持される）
DIR_RENAMES = {
    "IROS/deploy_results/modelA": "IROS/deploy_legacyA",
    "IROS/deploy_results/modelB": "IROS/deploy_legacyB",
    "IROS/deploy_results/modelC": "IROS/deploy_legacyC",
    "IROS/deploy_results/modelD": "IROS/deploy_legacyD",
    "IROS/deploy_results/DR": "IROS/deploy_DRfolder",
    "IROS/deploy_results/noDR": "IROS/deploy_noDRfolder",
    "IROS/collect_data_raw/raw_20260111": "IROS/sysid_20260111",
    "IROS/logs_verification": "IROS/verification",
    "IROS/analysis_results/figures_iros": "IROS/figures_paper",
    "IROS/legacy": "IROS/legacy_code",
    "models/IROS/pt_archive": "IROS/models_pt",
}

# 「ディレクトリ直下のファイルだけ」を対象ディレクトリへ移動（サブディレクトリは
# 対象にしない。サブディレクトリは上の DIR_RENAMES で個別に処理済みのはず）
DIRFILES_MOVES = {
    "IROS/collect_data_raw": "IROS/sysid_misc",
    "IROS/analysis_results": "IROS/figures_analysis",
    "IROS/figures": "IROS/figures_paper",
}

# セッションフォルダ全体をフラット化して data/ へ（サブディレクトリの区別は
# ファイル名に残っているので基底名だけで移動する）
SESSION_FLATTEN = {
    "results/RAL": "data/ral_20260803",
    "results/RAL_session1": "data/ral_20260731",
    "results/RAL_session2": "data/ral_20260731_b",
    "results/RAL_session3": "data/ral_20260731_c",
    "results/_quarantine_force_sensor_dead": "data/ral_quarantine",
}

# results/ 直下（サブディレクトリではない）の集計CSVファイル
RESULTS_ROOT_FILES_DEST = "out/ral"

# 移動しない（存在確認だけする）既知のパス
UNTOUCHED_DIRS = ["oc_demo", "midi", "models/RAL", "IROS/measured"]
UNTOUCHED_FILES = ["run_oc_demo.sh", "models/manifest.yaml", "IROS/README.md"]


def norm(p: str) -> str:
    return p.replace("\\", "/")


def build_plan(tracked_files: list[str]) -> list[tuple[str, str]]:
    """[(src, dst), ...] を返す。すべて git ls-files に基づく実在パスのみ対象。"""
    tracked = set(norm(f) for f in tracked_files)
    moves: list[tuple[str, str]] = []
    consumed: set[str] = set()

    def take(path: str) -> bool:
        if path in tracked and path not in consumed:
            consumed.add(path)
            return True
        return False

    # 1. ルート個別ファイル
    for src, dst in ROOT_FILE_MOVES.items():
        if take(src):
            moves.append((src, dst))

    # 2. ディレクトリ丸ごとリネーム（配下の全追跡ファイルを prefix 置換）
    for src_dir, dst_dir in DIR_RENAMES.items():
        prefix = src_dir + "/"
        for f in sorted(tracked):
            if f.startswith(prefix) and f not in consumed:
                rel = f[len(prefix):]
                moves.append((f, f"{dst_dir}/{rel}"))
                consumed.add(f)

    # 3. ディレクトリ直下ファイルのみ移動（サブディレクトリは除外＝既に2で消費済み）
    for src_dir, dst_dir in DIRFILES_MOVES.items():
        prefix = src_dir + "/"
        for f in sorted(tracked):
            if not f.startswith(prefix) or f in consumed:
                continue
            rel = f[len(prefix):]
            if "/" in rel:
                continue  # サブディレクトリ配下は対象外（2で処理済みのはず）
            moves.append((f, f"{dst_dir}/{rel}"))
            consumed.add(f)

    # 4. セッションフォルダのフラット化
    for src_dir, dst_dir in SESSION_FLATTEN.items():
        prefix = src_dir + "/"
        for f in sorted(tracked):
            if not f.startswith(prefix) or f in consumed:
                continue
            base = f.rsplit("/", 1)[-1]
            moves.append((f, f"{dst_dir}/{base}"))
            consumed.add(f)

    # 5. results/ 直下の集計CSV
    for f in sorted(tracked):
        if f in consumed:
            continue
        if f.startswith("results/") and "/" not in f[len("results/"):]:
            base = f.rsplit("/", 1)[-1]
            moves.append((f, f"{RESULTS_ROOT_FILES_DEST}/{base}"))
            consumed.add(f)

    return moves


def validate_plan(moves: list[tuple[str, str]], tracked_files: list[str]):
    tracked = set(norm(f) for f in tracked_files)
    errors = []

    # 5.1 全 src が実在する追跡ファイルか
    for src, _ in moves:
        if src not in tracked:
            errors.append(f"src が git 管理下に無い: {src}")

    # 5.2 dst の重複（衝突）が無いか
    dst_count = defaultdict(list)
    for src, dst in moves:
        dst_count[dst].append(src)
    for dst, srcs in dst_count.items():
        if len(srcs) > 1:
            errors.append(f"移動先が衝突: {dst}  <- {srcs}")

    # 5.3 dst が既存の（今回動かさない）追跡ファイルと衝突しないか
    moved_srcs = set(src for src, _ in moves)
    for _, dst in moves:
        if dst in tracked and dst not in moved_srcs:
            errors.append(f"移動先が既存の別ファイルと衝突: {dst}")

    # 5.4 CSV/JSON ペアが揃って同じ移動先ディレクトリに移動するか
    move_map = dict(moves)
    for src, dst in moves:
        if not src.endswith(".csv"):
            continue
        json_src = src[:-4] + ".json"
        if json_src not in tracked:
            continue  # 集計CSVなどJSONを持たないファイルは対象外
        if json_src not in move_map:
            errors.append(f"CSVとJSONのペアが揃っていない（JSON未計画）: {src} / {json_src}")
            continue
        csv_dst_dir = dst.rsplit("/", 1)[0]
        json_dst_dir = move_map[json_src].rsplit("/", 1)[0]
        if csv_dst_dir != json_dst_dir:
            errors.append(f"CSVとJSONの移動先ディレクトリが不一致: {src}->{dst} / "
                           f"{json_src}->{move_map[json_src]}")

    if errors:
        print("=== 検証エラー ===", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)


def print_plan(moves: list[tuple[str, str]]):
    by_dst_root = defaultdict(list)
    for src, dst in moves:
        root = dst.split("/", 1)[0]
        by_dst_root[root].append((src, dst))

    total = 0
    for root in sorted(by_dst_root):
        group = by_dst_root[root]
        total += len(group)
        print(f"\n### {root}/  ({len(group)} files)")
        # 代表例を数件だけ表示（全件は多すぎるため）
        for src, dst in group[:3]:
            print(f"  {src}  ->  {dst}")
        if len(group) > 3:
            print(f"  ... 他 {len(group) - 3} 件")
    print(f"\n合計: {total} files を移動予定")


def apply_plan(moves: list[tuple[str, str]]):
    # 移動先ディレクトリを作っておく（git mv はファイル単位だと
    # 移動先ディレクトリが無いと失敗するため）
    dst_dirs = sorted(set(dst.rsplit("/", 1)[0] for _, dst in moves))
    for d in dst_dirs:
        full = os.path.join(REPO_ROOT, *d.split("/"))
        os.makedirs(full, exist_ok=True)

    for i, (src, dst) in enumerate(moves, 1):
        r = subprocess.run(["git", "mv", src, dst], cwd=REPO_ROOT,
                            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[FAIL] git mv {src} {dst}\n{r.stderr}", file=sys.stderr)
            sys.exit(1)
        if i % 100 == 0 or i == len(moves):
            print(f"  ... {i}/{len(moves)}")

    # 空になった旧ディレクトリを掃除（ボトムアップ）
    candidate_dirs = set()
    for src, _ in moves:
        d = src.rsplit("/", 1)[0] if "/" in src else ""
        while d:
            candidate_dirs.add(d)
            d = d.rsplit("/", 1)[0] if "/" in d else ""
    for d in sorted(candidate_dirs, key=lambda p: -p.count("/")):
        full = os.path.join(REPO_ROOT, *d.split("/"))
        if os.path.isdir(full) and not os.listdir(full):
            os.rmdir(full)
            print(f"  (rmdir empty) {d}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="実際に git mv を実行する（既定はdry-run）")
    args = ap.parse_args()

    check_clean_tree()
    tracked = git_ls_files()
    moves = build_plan(tracked)
    validate_plan(moves, tracked)

    if not args.apply:
        print_plan(moves)
        print("\n(dry-run: 何も変更していません。実行するには --apply を付けてください)")
        return

    print(f"{len(moves)} files を git mv します...")
    apply_plan(moves)
    print("完了。git status で確認してください。")


if __name__ == "__main__":
    main()
