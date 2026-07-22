import argparse
import csv
import json
import re
from pathlib import Path
from cwite_paths import real_data_dir, real_run_path, real_output_path


DEFAULT_DIRS = [
    f"IPCW={real_run_path('ipcw')}",
    f"DeepHit={real_run_path('deephit')}",
    f"Powell={real_run_path('powell')}",
    f"CWITE={real_run_path('cwite')}",
]


DONE_RE = re.compile(
    r"SWEEP\s+\d+/\d+\s+DONE: val_low_mae=(?P<low>[0-9.]+)\s+"
    r"val_high_mae=(?P<high>[0-9.]+)\s+score=(?P<score>[0-9.]+)"
)
START_RE = re.compile(r"SWEEP\s+(?P<idx>\d+)/(?P<total>\d+)\s+START")
PROGRESS_RE = re.compile(
    r"Progress: completed=(?P<completed>\d+), failed=(?P<failed>\d+), "
    r"total_seen=(?P<seen>\d+)/(?P<total>\d+).*ETA=(?P<eta>[^;]+)"
)


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def finite_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def best_from_csv(out_dir):
    rows = read_csv_rows(out_dir / "sweep_results_partial.csv") or read_csv_rows(out_dir / "sweep_results.csv")
    ok_rows = [r for r in rows if r.get("status", "ok") == "ok" and finite_float(r.get("val_low_mae")) is not None]
    if not ok_rows:
        return None
    best = min(ok_rows, key=lambda r: float(r["val_low_mae"]))
    return {
        "method": best.get("method", out_dir.name),
        "val_low_mae": float(best["val_low_mae"]),
        "val_high_mae": float(best["val_high_mae"]),
        "val_score": float(best["val_score"]),
        "config": best,
        "completed": len(ok_rows),
        "failed": len([r for r in rows if r.get("status") == "failed"]),
        "source": str(out_dir / "sweep_results_partial.csv"),
    }


def infer_label(out_dir, method=None):
    name = Path(out_dir).name
    method = method or "Unknown"

    if name == "cwite" or method == "Proposed":
        variant = "CWITE"
    elif name in {"ipcw", "deephit", "powell"}:
        variant = name[0].upper() + name[1:]
    else:
        variant = method

    return variant


def infer_prediction_mode(summary):
    return "Final"


def infer_selection_group(summary):
    text = f"{summary.get('label', '')} {summary.get('out_dir', '')}".lower()
    if "overall" in text:
        return "Overall Selection"
    return "Not Overall Selection"


def best_from_log(out_dir):
    path = out_dir / "all_methods_feature_sweep.log.txt"
    if not path.exists():
        path = out_dir / "all_methods_argmax_feature_sweep.log.txt"
    if not path.exists():
        return None
    best = None
    progress = {}
    current_method = out_dir.name
    for line in path.read_text(errors="replace").splitlines():
        if "SWEEP" in line and "START:" in line:
            try:
                config = json.loads(line.split("START:", 1)[1].strip())
                current_method = config.get("method", current_method)
            except Exception:
                pass
            m = START_RE.search(line)
            if m:
                progress["seen"] = int(m.group("idx")) - 1
                progress["total"] = int(m.group("total"))
        m = DONE_RE.search(line)
        if m:
            candidate = {
                "method": current_method,
                "val_low_mae": float(m.group("low")),
                "val_high_mae": float(m.group("high")),
                "val_score": float(m.group("score")),
                "config": {},
                "source": str(path),
            }
            if best is None or candidate["val_low_mae"] < best["val_low_mae"]:
                best = candidate
        m = PROGRESS_RE.search(line)
        if m:
            progress.update(
                {
                    "completed": int(m.group("completed")),
                    "failed": int(m.group("failed")),
                    "seen": int(m.group("seen")),
                    "total": int(m.group("total")),
                    "eta": m.group("eta").strip(),
                }
            )
    if best is not None:
        best.update(progress)
    return best


def parse_out_dir_arg(raw):
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label, Path(path)
    return None, Path(raw)


def summarize_dir(raw_out_dir):
    explicit_label, out_dir = parse_out_dir_arg(raw_out_dir)
    summary = best_from_csv(out_dir) or best_from_log(out_dir)
    if summary is None:
        return {
            "method": out_dir.name,
            "label": explicit_label or infer_label(out_dir),
            "out_dir": str(out_dir),
            "status": "no results yet",
        }
    summary["out_dir"] = str(out_dir)
    summary["status"] = "ok"
    summary["label"] = explicit_label or infer_label(out_dir, summary.get("method"))
    best_config_path = out_dir / f"{summary['method'].lower()}_best_config.json"
    if best_config_path.exists():
        try:
            summary["best_config_file"] = json.loads(best_config_path.read_text())
        except Exception:
            pass
    return summary


def print_summary(summaries):
    ready = [s for s in summaries if s.get("status") == "ok"]
    waiting = [s for s in summaries if s.get("status") != "ok"]

    print("\n=== Feature Sweep Leaderboard: validation low-propensity MAE ===")
    if not ready:
        print("No completed configs found yet.")
    else:
        ranked = sorted(ready, key=lambda s: s["val_low_mae"])
        winner = ranked[0]
        for rank, s in enumerate(ranked, 1):
            gap = s["val_low_mae"] - winner["val_low_mae"]
            progress = ""
            if "completed" in s or "total" in s:
                progress = f" | completed={s.get('completed', '?')}/{s.get('total', '?')} failed={s.get('failed', 0)}"
            if "eta" in s:
                progress += f" ETA={s['eta']}"
            print(
                f"{rank}. {s['label']:30s} low={s['val_low_mae']:.3f} "
                f"high={s['val_high_mae']:.3f} score={s['val_score']:.3f} "
                f"gap={gap:+.3f}{progress}"
            )
        print(f"\nCurrent winner: {winner['label']} by low-bin validation MAE.")

        proposed_rows = [s for s in ready if "Proposed" in s["label"]]
        if proposed_rows:
            proposed = min(proposed_rows, key=lambda s: s["val_low_mae"])
            baselines = [s for s in ready if "Proposed" not in s["label"]]
            if baselines:
                best_base = min(baselines, key=lambda s: s["val_low_mae"])
                gap = proposed["val_low_mae"] - best_base["val_low_mae"]
                if gap < -0.25:
                    print(
                        f"Read: best Proposed variant ({proposed['label']}) is ahead of the best baseline "
                        f"({proposed['val_low_mae']:.3f} vs {best_base['label']} {best_base['val_low_mae']:.3f})."
                    )
                elif abs(gap) <= 0.25:
                    print(
                        f"Read: best Proposed variant ({proposed['label']}) is basically tied with the best baseline "
                        f"({proposed['val_low_mae']:.3f} vs {best_base['label']} {best_base['val_low_mae']:.3f})."
                    )
                elif gap >= 0.75:
                    print(
                        f"Read: best Proposed variant ({proposed['label']}) is struggling against {best_base['label']} "
                        f"({proposed['val_low_mae']:.3f} vs {best_base['val_low_mae']:.3f})."
                    )
                else:
                    print(
                        f"Read: best Proposed variant ({proposed['label']}) is behind but within striking distance of {best_base['label']} "
                        f"({proposed['val_low_mae']:.3f} vs {best_base['val_low_mae']:.3f})."
                    )

    if waiting:
        print("\nNo completed configs found for:")
        for s in waiting:
            print(f"- {s['label']}: {s['out_dir']}")


def print_grouped_summary(summaries):
    ready = [s for s in summaries if s.get("status") == "ok"]
    if not ready:
        return

    print("\n=== Grouped Leaderboards ===")
    group_order = [
        ("Overall Selection", "Final"),
        ("Not Overall Selection", "Final"),
    ]
    for selection_group, prediction_mode in group_order:
        rows = [
            s
            for s in ready
            if infer_selection_group(s) == selection_group
            and infer_prediction_mode(s) == prediction_mode
        ]
        if not rows:
            continue
        rank_metric = "val_score" if selection_group == "Overall Selection" else "val_low_mae"
        ranked = sorted(rows, key=lambda s: s[rank_metric])
        winner = ranked[0]
        metric_label = "score" if rank_metric == "val_score" else "low-bin MAE"
        print(f"\n--- {selection_group} | ranked by {metric_label} ---")
        for rank, s in enumerate(ranked, 1):
            gap = s[rank_metric] - winner[rank_metric]
            progress = ""
            if "completed" in s or "total" in s:
                progress = f" | completed={s.get('completed', '?')}/{s.get('total', '?')} failed={s.get('failed', 0)}"
            if "eta" in s:
                progress += f" ETA={s['eta']}"
            print(
                f"{rank}. {s['label']:34s} low={s['val_low_mae']:.3f} "
                f"high={s['val_high_mae']:.3f} score={s['val_score']:.3f} "
                f"gap={gap:+.3f}{progress}"
            )

        proposed_rows = [s for s in ranked if "CWITE" in s["label"]]
        baseline_rows = [s for s in ranked if "CWITE" not in s["label"]]
        if proposed_rows and baseline_rows:
            proposed = min(proposed_rows, key=lambda s: s[rank_metric])
            baseline = min(baseline_rows, key=lambda s: s[rank_metric])
            gap = proposed[rank_metric] - baseline[rank_metric]
            if gap < -0.25:
                read = "CWITE is ahead"
            elif abs(gap) <= 0.25:
                read = "CWITE is basically tied"
            elif gap >= 0.75:
                read = "CWITE is struggling"
            else:
                read = "CWITE is behind but within striking distance"
            print(
                f"Read: {read} by {metric_label} ({proposed['label']} {proposed[rank_metric]:.3f} "
                f"vs {baseline['label']} {baseline[rank_metric]:.3f})."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "out_dirs",
        nargs="*",
        default=DEFAULT_DIRS,
        help="Output dirs, optionally as Label=/path/to/out_dir.",
    )
    args = parser.parse_args()
    summaries = [summarize_dir(out_dir) for out_dir in args.out_dirs]
    print_summary(summaries)
    print_grouped_summary(summaries)


if __name__ == "__main__":
    main()
