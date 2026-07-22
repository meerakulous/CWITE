#!/usr/bin/env python3
"""Recover paper-relevant hyperparameters from CWITE experiment tarballs.

The scripts in this folder intentionally avoid importing project code from the
archives. They inspect text, Python, JSON, and CSV files inside a .tar.gz and
summarize both selected hyperparameters and the search grids needed for the
paper appendix.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import re
import shlex
import tarfile
from pathlib import Path
from typing import Any


METHOD_TXT_NAMES = {
    "ipcw": ("IPCW.txt", ["propensity_C", "num_layers", "hidden_size", "batch_size", "lr", "weight_decay"]),
    "deephit": ("deephit.txt", ["num_layers", "hidden_size", "batch_size", "lr", "weight_decay"]),
    "powell": ("powells.txt", ["num_layers", "hidden_size", "batch_size", "lr", "weight_decay"]),
    "standard_margin": ("powells.txt", ["num_layers", "hidden_size", "batch_size", "lr", "weight_decay"]),
    "oracle": ("oracle.txt", ["num_layers", "hidden_size", "batch_size", "lr", "weight_decay"]),
    "cwite": (
        "proposed.txt",
        [
            "propensity_C",
            "k",
            "num_layers",
            "hidden_size",
            "batch_size",
            "lr",
            "weight_decay",
            "lambda_loss",
            "weight_func",
        ],
    ),
}

SYNTHETIC_HYP_FILES = {
    "IPCW": ("ipcw", ["num_layers", "hidden_size", "batch_size", "lr", "weight_decay"]),
    "deephit": ("deephit", ["num_layers", "hidden_size", "batch_size", "lr", "weight_decay"]),
    "powells": ("powell", ["num_layers", "hidden_size", "batch_size", "lr", "weight_decay"]),
    "oracle": ("oracle", ["num_layers", "hidden_size", "batch_size", "lr", "weight_decay"]),
    "proposed_orig": ("cwite", ["num_layers", "hidden_size", "batch_size", "lr", "weight_decay", "lambda_loss", "weight_func"]),
}

REAL_TRAIN_COMMANDS = {
    "train_ipcw": "IPCW",
    "train_deephit": "Standard",
    "train_powell": "Standard Margin",
    "train_cwite": "CWITE",
}

DEFAULT_SEMISYNTH_LABELS = [
    "covariate_low_resource",
    "covariate_moderate_resource",
    "covariate_high_resource",
]

PAPER_FIELDS = [
    "method",
    "setting",
    "transform",
    "select_dim",
    "num_layers",
    "hidden_size",
    "batch_size",
    "lr",
    "weight_decay",
    "propensity_C",
    "k",
    "weight_func",
    "lambda_loss",
    "max_epochs",
    "patience",
    "seed",
]


def open_archive(path: str | Path) -> tarfile.TarFile:
    return tarfile.open(path, "r:gz")


def member_names(tf: tarfile.TarFile) -> list[str]:
    return [m.name for m in tf.getmembers() if m.isfile()]


def basename(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def read_member_text(tf: tarfile.TarFile, name: str) -> str:
    f = tf.extractfile(name)
    if f is None:
        return ""
    raw = f.read()
    return raw.decode("utf-8", errors="replace")


def find_by_basename(names: list[str], wanted: str) -> str | None:
    matches = [n for n in names if basename(n) == wanted and ".ipynb_checkpoints" not in n]
    if matches:
        return sorted(matches, key=len)[0]
    return None


def parse_literal_lines(text: str) -> list[Any]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(ast.literal_eval(line))
        except Exception:
            rows.append(line)
    return rows


def zip_fields(fields: list[str], values: Any) -> dict[str, Any]:
    if not isinstance(values, (list, tuple)):
        return {"raw": values}
    out: dict[str, Any] = {}
    for i, value in enumerate(values):
        key = fields[i] if i < len(fields) else f"extra_{i}"
        out[key] = value
    return out


def parse_python_search_metadata(text: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}

    seed_match = re.search(r"set_seed\((\d+)\)", text)
    if seed_match:
        meta["seed"] = int(seed_match.group(1))

    grid_match = re.search(r"for\s+grididx\s+in\s+range\((\d+)\)", text)
    if grid_match:
        meta["random_grid_trials"] = int(grid_match.group(1))

    epoch_match = re.search(r"for\s+epoch\s+in\s+range\((\d+)\)", text)
    if epoch_match:
        meta["max_epochs"] = int(epoch_match.group(1))

    patience_match = re.search(r"if\s+stop\s*==\s*(\d+)", text)
    if patience_match:
        meta["early_stopping_patience"] = int(patience_match.group(1))

    if "optim.Adam" in text:
        meta["optimizer"] = "Adam"

    arg_defaults = parse_argparse_defaults(text)
    if arg_defaults:
        meta["argparse_defaults"] = arg_defaults

    if "LogisticRegression" in text:
        c_match = re.search(r"for\s+\w+\s+in\s+\[([^\]]+)\]", text)
        if c_match and "LogisticRegression" in text[c_match.end() : c_match.end() + 300]:
            meta["propensity_C_grid"] = "[" + c_match.group(1).strip() + "]"

    choices = {
        "num_layers_grid": r"num_layers\s*=\s*np\.random\.choice\((\[[^\]]+\])",
        "hidden_size_grid": r"hidden_size\s*=\s*np\.random\.choice\((\[[^\]]+\])",
        "weight_decay_grid": r"wd\s*=\s*np\.random\.choice\((\[[^\]]+\])",
    }
    for key, pattern in choices.items():
        match = re.search(pattern, text)
        if match:
            meta[key] = match.group(1)

    batch_options_match = re.search(r"batch_options\s*=\s*(\{.*?\n\s*\})", text, re.S)
    if batch_options_match:
        meta["batch_size_lr_grid"] = " ".join(batch_options_match.group(1).split())

    lambda_match = re.search(r"lambda[^=\n]*=\s*np\.random\.choice\((\[[^\]]+\])", text)
    if lambda_match:
        meta["lambda_loss_grid"] = lambda_match.group(1)

    func_match = re.search(r"(?:func|weight_func)[^=\n]*=\s*np\.random\.choice\((\[[^\]]+\])", text)
    if func_match:
        meta["weight_func_grid"] = func_match.group(1)

    return meta


def parse_argparse_defaults(text: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    pattern = re.compile(r"add_argument\((.*?)\)", re.S)
    for match in pattern.finditer(text):
        call = " ".join(match.group(1).split())
        flag_match = re.search(r"[\"']--([A-Za-z0-9_-]+)[\"']", call)
        default_match = re.search(r"default\s*=\s*([^,\)]+)", call)
        if not flag_match or not default_match:
            continue
        key = flag_match.group(1).replace("-", "_")
        raw = default_match.group(1).strip()
        try:
            defaults[key] = ast.literal_eval(raw)
        except Exception:
            try:
                defaults[key] = float(raw) if "." in raw or "e" in raw.lower() else int(raw)
            except Exception:
                defaults[key] = raw
    return defaults


def recover_legacy_selected(tf: tarfile.TarFile, names: list[str], default_labels: list[str] | None = None) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for method, (txt_name, fields) in METHOD_TXT_NAMES.items():
        if method == "standard_margin":
            continue
        member = find_by_basename(names, txt_name)
        if not member:
            continue
        rows = parse_literal_lines(read_member_text(tf, member))
        labels = default_labels if default_labels and len(default_labels) == len(rows) else [f"run_{i + 1}" for i in range(len(rows))]
        selected[method] = {
            "source": member,
            "rows": [
                {"setting": labels[i], **zip_fields(fields, row)}
                for i, row in enumerate(rows)
            ],
        }
    if "powell" in selected:
        selected["standard_margin"] = {**selected["powell"], "alias_of": "powell"}
    return selected


def recover_synthetic_selected(tf: tarfile.TarFile, names: list[str]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    pattern = re.compile(r"(.+)_hyp_(\d+)_(\d+)\.txt$")
    for name in names:
        base = basename(name)
        match = pattern.match(base)
        if not match or ".ipynb_checkpoints" in name:
            continue
        stem, start, end = match.groups()
        if stem not in SYNTHETIC_HYP_FILES:
            continue
        method, fields = SYNTHETIC_HYP_FILES[stem]
        rows = parse_synthetic_hyp_rows(read_member_text(tf, name), fields, int(start), stem)
        if not rows:
            continue
        payload = selected.setdefault(method, {"source": [], "rows": []})
        payload["source"].append(name)
        payload["rows"].extend(rows)
    if "powell" in selected:
        selected["standard_margin"] = {
            "source": selected["powell"]["source"],
            "rows": [dict(row) for row in selected["powell"]["rows"]],
            "alias_of": "powell",
        }
    for payload in selected.values():
        payload["rows"].sort(key=lambda row: (row.get("setting", ""), row.get("count", -1)))
    return selected


def parse_synthetic_hyp_rows(text: str, fields: list[str], range_start: int, stem: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending_count: int | None = None
    pending_metric: float | None = None
    next_count = range_start
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        count_list_match = re.match(r"^(\d+)\s*,\s*(\[.*\])$", line)
        metric_match = re.match(r"^(\d+)\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)$", line, re.I)
        list_text: str | None = None
        count: int
        metric: float | None = None
        if count_list_match:
            count = int(count_list_match.group(1))
            list_text = count_list_match.group(2)
        elif metric_match:
            pending_count = int(metric_match.group(1))
            pending_metric = float(metric_match.group(2))
            continue
        elif line.startswith("["):
            count = pending_count if pending_count is not None else next_count
            metric = pending_metric
            list_text = line
        else:
            continue
        try:
            values = ast.literal_eval(list_text)
        except Exception:
            continue
        row = {
            "setting": synthetic_count_bin(count),
            "count": count,
            **zip_fields(fields, values),
        }
        if metric is not None:
            row["propensity_val_auroc"] = metric
        if stem == "proposed_orig":
            row["k"] = "elbow"
        rows.append(row)
        pending_count = None
        pending_metric = None
        next_count = count + 1
    return rows


def synthetic_count_bin(count: int) -> str:
    start = (count // 125) * 125
    end = start + 125
    return f"COUNT{start}-{end}"


def recover_source_metadata(tf: tarfile.TarFile, names: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    tracked_sources = {
        "IPCW.py",
        "deephit.py",
        "powells.py",
        "oracle.py",
        "proposed.py",
        "realdata_overall_sweep.py",
        "run_all_realdata_series.sh",
        "proposed_cluster_method_controls.py",
        "proposed_random_cluster_control.py",
        "proposed_k_sensitivity.py",
    }
    for name in names:
        base = basename(name)
        if not base.endswith(".py") or ".ipynb_checkpoints" in name:
            continue
        if "_hyp.py" not in base and base not in tracked_sources:
            continue
        text = read_member_text(tf, name)
        meta = parse_python_search_metadata(text) if base.endswith(".py") else {}
        metadata[name] = meta
    return metadata


def shell_commands_from_runner(text: str) -> dict[str, dict[str, Any]]:
    commands: dict[str, dict[str, Any]] = {}
    current_name = None
    current_parts: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("run_and_log "):
            if current_name and current_parts:
                commands[current_name] = parse_shell_args(" ".join(current_parts))
            parts = line.split()
            current_name = parts[1] if len(parts) > 1 else "unknown"
            current_parts = [line]
        elif current_name:
            current_parts.append(line.rstrip("\\").strip())
            if line == "":
                commands[current_name] = parse_shell_args(" ".join(current_parts))
                current_name = None
                current_parts = []
    if current_name and current_parts:
        commands[current_name] = parse_shell_args(" ".join(current_parts))
    return commands


def parse_shell_args(command: str) -> dict[str, Any]:
    command = re.sub(r'"\$\{RUN_ROOT\}/([^"]+)"', r"RUN_ROOT/\1", command)
    command = re.sub(r'"\$\{DATA_DIR\}"', "DATA_DIR", command)
    command = re.sub(r'"\$\{GPU\}"', "GPU", command)
    command = re.sub(r'"\$\{PYTHON\}"', "python", command)
    command = command.replace('"${PYTHON}"', "python")
    command = command.replace('"${GPU}"', "GPU")
    command = command.replace('"${DATA_DIR}"', "DATA_DIR")
    command = command.replace('"${RUN_ROOT}', "RUN_ROOT")
    command = command.replace("${RUN_ROOT}", "RUN_ROOT")
    command = command.replace('}"', "")
    command = command.replace("\\", " ")
    try:
        tokens = shlex.split(command)
    except ValueError:
        # Some runner paths contain quoted shell parameter expansions that are
        # awkward outside a shell. For recovery purposes, whitespace tokenizing
        # after stripping quotes is enough to recover flags and values.
        tokens = command.replace('"', "").replace("'", "").split()
    out: dict[str, Any] = {"raw": command}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--") or tok == "-gpu":
            key = tok.lstrip("-").replace("-", "_")
            vals: list[str] = []
            i += 1
            while i < len(tokens) and not tokens[i].startswith("-"):
                vals.append(tokens[i])
                i += 1
            out[key] = vals[0] if len(vals) == 1 else vals if vals else True
            continue
        i += 1
    return out


def recover_real(tf: tarfile.TarFile, names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "selected_configs": {},
        "sweep_results": {},
        "runner_commands": {},
        "source_metadata": recover_source_metadata(tf, names),
    }

    for name in names:
        base = basename(name)
        if base.endswith("_best_config.json") or re.match(r"cwite_k\d+_config\.json$", base):
            try:
                result["selected_configs"][name] = json.loads(read_member_text(tf, name))
            except json.JSONDecodeError:
                result["selected_configs"][name] = {"raw": read_member_text(tf, name)}
        elif base in {"sweep_results.csv", "sweep_results_partial.csv", "cluster_method_control_results.csv", "k_sensitivity_results.csv"}:
            text = read_member_text(tf, name)
            reader = csv.DictReader(io.StringIO(text))
            rows = list(reader)
            result["sweep_results"][name] = {
                "n_rows": len(rows),
                "best_rows": best_rows_from_csv(rows),
            }

    runner = find_by_basename(names, "run_all_realdata_series.sh")
    if runner:
        result["runner_commands"] = shell_commands_from_runner(read_member_text(tf, runner))

    return result


def best_rows_from_csv(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    by_method: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_method.setdefault(row.get("method", "unknown"), []).append(row)
    best = []
    for method, method_rows in by_method.items():
        sortable = []
        for row in method_rows:
            try:
                sortable.append((float(row.get("val_score", "nan")), row))
            except ValueError:
                pass
        if sortable:
            best.append({"method": method, **min(sortable, key=lambda item: item[0])[1]})
    return best


def flatten_for_paper(report: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if kind in {"synthetic", "semi_synthetic"}:
        for method, payload in report.get("selected_hyperparameters", {}).items():
            for row in payload.get("rows", []):
                out = {field: "" for field in PAPER_FIELDS}
                out.update(row)
                out["method"] = "Standard Margin" if method == "standard_margin" else method
                rows.append(out)
    elif kind == "real":
        selected = report.get("selected_configs", {})
        for path, cfg in selected.items():
            rows.append(real_row_from_selected_config(path, cfg, report))
        if not rows:
            rows.extend(real_rows_from_runner(report))
        order = {"IPCW": 0, "Standard": 1, "Standard Margin": 2, "CWITE": 3}
        rows.sort(key=lambda row: order.get(str(row.get("method", "")), 99))
    return rows


def real_method_label(raw_method: Any, path: str = "") -> str:
    method = str(raw_method or basename(path).split("_best_config")[0])
    mapping = {
        "Proposed": "CWITE",
        "proposed": "CWITE",
        "cwite": "CWITE",
        "DeepHit": "Standard",
        "deephit": "Standard",
        "Powell": "Standard Margin",
        "powell": "Standard Margin",
        "IPCW": "IPCW",
        "ipcw": "IPCW",
    }
    return mapping.get(method, method)


def real_row_from_selected_config(path: str, cfg: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    out = {field: "" for field in PAPER_FIELDS}
    method = real_method_label(cfg.get("method"), path)
    out.update(
        {
            "method": method,
            "setting": "real_data_selected",
            "transform": cfg.get("transform", ""),
            "select_dim": cfg.get("dim", "") if cfg.get("transform") == "select" else "",
            "num_layers": cfg.get("num_layers", ""),
            "hidden_size": cfg.get("hidden_size", ""),
            "batch_size": cfg.get("batch_size", ""),
            "lr": cfg.get("lr", ""),
            "weight_decay": cfg.get("weight_decay", cfg.get("wd", "")),
            "seed": cfg.get("seed", ""),
        }
    )

    if method == "CWITE":
        out.update(
            {
                "k": cfg.get("k", ""),
                "weight_func": cfg.get("weight_func", ""),
                "lambda_loss": cfg.get("lambda_loss", ""),
            }
        )
    return out


def real_rows_from_runner(report: dict[str, Any]) -> list[dict[str, Any]]:
    commands = report.get("runner_commands", {})
    if not commands:
        return []
    defaults = {}
    for path, meta in report.get("source_metadata", {}).items():
        if basename(path) == "realdata_overall_sweep.py":
            defaults = meta.get("argparse_defaults", {})
            break

    rows: list[dict[str, Any]] = []
    for command_name, method in REAL_TRAIN_COMMANDS.items():
        args = commands.get(command_name)
        if not args:
            continue
        out = {field: "" for field in PAPER_FIELDS}
        out.update(
            {
                "method": method,
                "setting": "real_data_feature_sweep",
                "transform": join_values(args.get("transforms", defaults.get("transforms", ""))),
                "select_dim": join_values(args.get("select_dims", defaults.get("select_dims", ""))),
                "num_layers": args.get("num_layers", defaults.get("num_layers", "")),
                "hidden_size": args.get("hidden_size", defaults.get("hidden_size", "")),
                "batch_size": args.get("batch_size", defaults.get("batch_size", "")),
                "lr": args.get("lr", defaults.get("lr", "")),
                "weight_decay": args.get("wd", defaults.get("wd", "")),
                "k": join_values(args.get("k", "")),
                "weight_func": join_values(args.get("weight_func", "")),
                "lambda_loss": join_values(args.get("lambda_loss", "")),
                "max_epochs": args.get("max_epochs", defaults.get("max_epochs", "")),
                "patience": args.get("patience", defaults.get("patience", "")),
                "seed": args.get("seed", defaults.get("seed", "")),
            }
        )
        rows.append(out)
    return rows


def join_values(value: Any) -> Any:
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return value


def write_outputs(report: dict[str, Any], out_dir: Path, prefix: str, kind: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{prefix}_hyperparameters.json"
    md_path = out_dir / f"{prefix}_hyperparameters.md"
    csv_path = out_dir / f"{prefix}_paper_table_rows.csv"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    paper_rows = flatten_for_paper(report, kind)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in paper_rows:
            writer.writerow(row)

    md_path.write_text(render_markdown(report, paper_rows, kind) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


def render_markdown(report: dict[str, Any], paper_rows: list[dict[str, Any]], kind: str) -> str:
    lines = [f"# Hyperparameter Recovery: {kind}", ""]
    if report.get("archive"):
        lines += [f"Archive: `{report['archive']}`", ""]

    if paper_rows:
        lines += ["## Paper Table Rows", ""]
        header = ["method", "setting", "transform", "select_dim", "num_layers", "hidden_size", "batch_size", "lr", "weight_decay", "propensity_C", "k", "weight_func", "lambda_loss"]
        if len(paper_rows) > 100:
            lines += [
                f"{len(paper_rows)} rows were recovered. The complete table is in the companion CSV; this preview shows the first 40 rows.",
                "",
            ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        preview_rows = paper_rows[:40] if len(paper_rows) > 100 else paper_rows
        for row in preview_rows:
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in header) + " |")
        lines.append("")
    else:
        lines += [
            "## Paper Table Rows",
            "",
            "No selected hyperparameter rows were found in this archive. If this is a code-only archive, rerun on the result/output tarball containing `*_best_config.json`, `sweep_results.csv`, or legacy `*.txt` files.",
            "",
        ]

    lines += ["## Recovered Search Metadata", ""]
    source_meta = report.get("source_metadata", {})
    if source_meta:
        for path, meta in sorted(source_meta.items()):
            lines.append(f"### `{path}`")
            if meta:
                for key, value in sorted(meta.items()):
                    lines.append(f"- `{key}`: `{value}`")
            else:
                lines.append("- No grid metadata pattern detected.")
            lines.append("")

    if report.get("runner_commands"):
        lines += ["## Real-Data Runner Commands", ""]
        for name, args in sorted(report["runner_commands"].items()):
            lines.append(f"### `{name}`")
            for key, value in sorted(args.items()):
                if key != "raw":
                    lines.append(f"- `{key}`: `{value}`")
            lines.append("")

    if report.get("selected_configs"):
        lines += ["## Selected JSON Configs", ""]
        for path, cfg in sorted(report["selected_configs"].items()):
            lines.append(f"### `{path}`")
            for key, value in sorted(cfg.items()):
                lines.append(f"- `{key}`: `{value}`")
            lines.append("")

    if kind == "synthetic" and report.get("selected_hyperparameters"):
        lines += ["## Synthetic Row Counts", ""]
        for method, payload in sorted(report["selected_hyperparameters"].items()):
            rows = payload.get("rows", [])
            by_setting: dict[str, int] = {}
            for row in rows:
                by_setting[row.get("setting", "unknown")] = by_setting.get(row.get("setting", "unknown"), 0) + 1
            counts = ", ".join(f"{setting}: {count}" for setting, count in sorted(by_setting.items()))
            lines.append(f"- `{method}`: {len(rows)} rows ({counts})")
        lines.append("")

    return "\n".join(lines)


def recover_archive(archive: str, kind: str, out_dir: str | Path, prefix: str | None = None) -> dict[str, Any]:
    prefix = prefix or Path(archive).name.replace(".tar.gz", "").replace(".tgz", "").replace(".targ.z", "")
    with open_archive(archive) as tf:
        names = member_names(tf)
        if kind == "real":
            report = recover_real(tf, names)
        elif kind == "synthetic":
            report = {
                "selected_hyperparameters": recover_synthetic_selected(tf, names),
                "source_metadata": recover_source_metadata(tf, names),
            }
        else:
            labels = DEFAULT_SEMISYNTH_LABELS if kind == "semi_synthetic" else None
            report = {
                "selected_hyperparameters": recover_legacy_selected(tf, names, labels),
                "source_metadata": recover_source_metadata(tf, names),
            }
        report["archive"] = archive
        report["kind"] = kind
        report["n_files_seen"] = len(names)

    write_outputs(report, Path(out_dir), prefix, kind)
    return report


def main(kind: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", help="Path to .tar.gz/.tgz archive")
    parser.add_argument("--out-dir", default="hyperparameter_recovery_outputs")
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()
    recover_archive(args.archive, kind=kind, out_dir=args.out_dir, prefix=args.prefix)
