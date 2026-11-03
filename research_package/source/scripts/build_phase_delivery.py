from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "phasor_phase_delivery_20261023"
RUN_ID = datetime.now().strftime("%H%M%S")
STAGE = Path(f"C:/_phs_{RUN_ID}")
OUT_DIR = ROOT.parent

ACTIVE_REL = Path(
    "results/g1_phase_model_comparison_20261023/tctst_residual_v4/"
    "recovery_seed_20261023_attempt01/tctst_repaired_seed_20261023"
)
ACTIVE_SRC = ROOT / ACTIVE_REL


def latest_epoch(path: Path) -> int | None:
    if not path.exists():
        return None
    last = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
            if isinstance(item.get("epoch"), int):
                last = item["epoch"]
        except Exception:
            continue
    return last


def copy_file(src: Path, dst: Path, retry: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    attempts = 8 if retry else 3
    for attempt in range(attempts):
        try:
            before = src.stat()
            shutil.copyfile(src, dst)
            after = src.stat()
            stable = (
                dst.stat().st_size == after.st_size
                and before.st_size == after.st_size
                and before.st_mtime_ns == after.st_mtime_ns
            )
            if stable:
                shutil.copystat(src, dst)
                return
        except FileNotFoundError:
            if attempt + 1 == attempts:
                raise RuntimeError(f"Source disappeared while snapshotting: {src}")
        time.sleep(0.4)
    raise RuntimeError(f"Source kept changing while snapshotting: {src}")


def copy_tree(src: Path, dst: Path, exclude_active: bool = False) -> None:
    if not src.exists():
        return
    for current, dirs, files in os.walk(src):
        cur = Path(current)
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache"}]
        if exclude_active and (cur == ACTIVE_SRC or ACTIVE_SRC in cur.parents):
            dirs[:] = []
            continue
        rel = cur.relative_to(src)
        target_dir = dst / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name.endswith(".pyc"):
                continue
            copy_file(cur / name, target_dir / name)


def main() -> None:
    if STAGE.exists():
        raise SystemExit(f"Refusing to reuse staging directory: {STAGE}")
    STAGE.mkdir(parents=True)
    bundle_root = STAGE

    out = OUT_DIR / f"{BUNDLE_NAME}.zip"
    version = 2
    while out.exists():
        out = OUT_DIR / f"{BUNDLE_NAME}_v{version}.zip"
        version += 1

    snapshot_time = datetime.now().astimezone().isoformat(timespec="seconds")
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    active_epoch = latest_epoch(ACTIVE_SRC / "epochs.jsonl")

    # Capture the live training checkpoint first, between periodic saves.
    active_dst = bundle_root / ACTIVE_REL
    if ACTIVE_SRC.exists():
        active_dst.mkdir(parents=True, exist_ok=True)
        for name in ("partial_checkpoint.pt", "manifest.json"):
            src = ACTIVE_SRC / name
            if src.exists():
                copy_file(src, active_dst / name, retry=name == "partial_checkpoint.pt")
        log_src = ACTIVE_SRC / "epochs.jsonl"
        if log_src.exists():
            valid_lines = []
            for line in log_src.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    json.loads(line)
                    valid_lines.append(line)
                except Exception:
                    pass
            (active_dst / "epochs.jsonl").write_text(
                "".join(line + "\n" for line in valid_lines), encoding="utf-8"
            )

    for name in ("data", "docs", "models"):
        copy_tree(ROOT / name, bundle_root / name)
    copy_tree(ROOT / "results", bundle_root / "results", exclude_active=True)
    copy_file(ROOT / "fixed_clock_baseline.json", bundle_root / "fixed_clock_baseline.json")
    copy_file(ROOT / "progress_v4.md", bundle_root / "progress_v4.md")
    copy_file(ROOT / "LICENCE", bundle_root / "LICENCE")

    script_pattern = re.compile(r"(phase|phasor|g1_|tctst|pinn)", re.IGNORECASE)
    support_scripts = {
        "analyze.py", "collect_data.py", "csv_to_npz.py", "finalize.sh",
        "generate_final_report.py", "play.py", "play_fixed.py", "play_patched.py",
        "reproduce.sh", "run_rough_equalbudget.sh", "train.py",
        "visualize_terrain.py",
    }
    for src in sorted((ROOT / "scripts").iterdir()):
        if (
            src.is_file()
            and src.suffix.lower() in {".py", ".sh"}
            and (script_pattern.search(src.name) or src.name in support_scripts)
        ):
            copy_file(src, bundle_root / "scripts" / src.name)

    overlay_paths = (
        "src/tasks/velocity/phasor",
        "src/tasks/velocity/mdp/__init__.py",
        "src/tasks/velocity/mdp/observations.py",
        "src/tasks/velocity/mdp/rewards.py",
        "src/tasks/velocity/config/g1/__init__.py",
        "src/tasks/velocity/config/g1/env_cfgs.py",
        "src/assets/robots/unitree_g1/g1_constants.py",
        "src/runtime_compat.py",
    )
    overlay = bundle_root / "source_overlay"
    for rel in overlay_paths:
        src = ROOT / rel
        if src.is_file():
            copy_file(src, overlay / rel)
        elif src.is_dir():
            copy_tree(src, overlay / rel)
    (overlay / "BASE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
    copy_tree(ROOT / "tests", bundle_root / "tests")
    copy_tree(ROOT / "logs/rsl_rl/g1_velocity", bundle_root / "logs/rsl_rl/g1_velocity")

    status = {
        "snapshot_time_local": snapshot_time,
        "base_git_commit": commit,
        "active_tctst_seed": 20261023,
        "active_tctst_latest_epoch_in_log_at_snapshot_start": active_epoch,
        "active_tctst_planned_epochs": 100,
        "active_tctst_completed_at_snapshot": False,
        "active_tctst_snapshot_note": (
            "Point-in-time copy of the latest periodic checkpoint and complete "
            "epoch-log lines. Training continued in the original workspace."
        ),
        "model_selection_complete": False,
        "selected_checkpoint_fixed_clock_closed_loop_comparison_complete": False,
    }
    (bundle_root / "PACKAGE_STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    readme = f"""# 相位项目交付包

本包整理当前工作区中与机器人步态相位估计、PHASOR 接入、模型评测及固定 clock 对照直接相关的代码、数据和结果。

## 快照状态

- 快照时间：{snapshot_time}
- 基准仓库提交：{commit}
- 正在续跑的 TCTST seed 20261023：快照时日志到第 {active_epoch if active_epoch is not None else "未知"} / 100 轮；包内有该时点的周期检查点、训练日志和启动清单。
- 该训练在打包时尚未完成。后续模型选型、runtime parity 和使用最终选中检查点的 fixed-clock 闭环对照，不由这个快照代表。

## 内容

- results：相位数据、标签/门禁、外部模型比较、模型训练、控制闭环和历史尝试结果。
- data：G1 步行/跑步原始轨迹及派生数据。
- scripts：训练、评估、分析、恢复和闭环对照脚本。
- source_overlay：PHASOR 模块及接入 G1 任务所需的源码覆盖层；BASE_COMMIT.txt 标出基准提交。
- tests：相位数据、模型、runtime、门禁和控制实验测试。
- docs：预注册、实验计划和数据来源记录。
- logs：G1 velocity 下的 PHASOR 与 baseline 对照日志和模型。
- models：已有相位编码器与 PINN 模型文件。
- fixed_clock_baseline.json、progress_v4.md：固定时钟基线和项目进展记录。

本包聚焦相位研究内容，不包含 Git 历史、缓存、整套 Unitree 上游仓库或通用第三方仿真二进制。因此 source_overlay 需放到匹配 BASE_COMMIT.txt 的仓库中使用，不能单独作为可运行仓库。

PACKAGE_MANIFEST.tsv 列出压缩包内文件的相对路径和字节数。压缩包使用 ZIP64。
"""
    (bundle_root / "README_交付说明.md").write_text(readme, encoding="utf-8")

    manifest_path = bundle_root / "PACKAGE_MANIFEST.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["path", "bytes"])
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file() and path != manifest_path:
                writer.writerow([path.relative_to(bundle_root).as_posix(), path.stat().st_size])

    compressible = {
        ".py", ".sh", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml",
        ".csv", ".log", ".html", ".tsv", ".diff", ".xml",
    }
    files = sorted(path for path in bundle_root.rglob("*") if path.is_file())
    with zipfile.ZipFile(out, "w", allowZip64=True) as archive:
        for index, path in enumerate(files, 1):
            name = (Path(BUNDLE_NAME) / path.relative_to(bundle_root)).as_posix()
            compression = (
                zipfile.ZIP_DEFLATED
                if path.suffix.lower() in compressible
                else zipfile.ZIP_STORED
            )
            archive.write(
                path,
                name,
                compress_type=compression,
                compresslevel=1 if compression == zipfile.ZIP_DEFLATED else None,
            )
            if index % 200 == 0:
                print(f"ZIP {index}/{len(files)} files", flush=True)

    with zipfile.ZipFile(out, "r") as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        names = set(archive.namelist())
    required = (
        f"{BUNDLE_NAME}/README_交付说明.md",
        f"{BUNDLE_NAME}/PACKAGE_STATUS.json",
        f"{BUNDLE_NAME}/PACKAGE_MANIFEST.tsv",
        f"{BUNDLE_NAME}/data",
        f"{BUNDLE_NAME}/source_overlay/src/tasks/velocity/phasor/runtime.py",
    )
    missing = [
        item for item in required
        if item not in names and not any(name.startswith(item + "/") for name in names)
    ]
    if missing:
        raise RuntimeError(f"Missing required paths in ZIP: {missing}")

    print(json.dumps({
        "archive": str(out),
        "archive_bytes": out.stat().st_size,
        "file_count": len(files),
        "payload_bytes": sum(path.stat().st_size for path in files),
        "active_tctst_epoch_at_snapshot": active_epoch,
        "crc_check": "passed",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
