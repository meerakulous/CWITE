from pathlib import Path
import os


DATA_ROOT = Path(os.environ.get("CWITE_REAL_DATA_DIR", os.environ.get("CWITE_DATA_ROOT", "../data/real_labor"))).expanduser()
RUN_ROOT = Path(os.environ.get("CWITE_REAL_RUN_ROOT", os.environ.get("CWITE_OUTPUT_ROOT", "../outputs/real_data"))).expanduser()


def real_data_dir():
    return str(DATA_ROOT)


def real_run_path(*parts):
    path = RUN_ROOT.joinpath(*parts)
    if parts and Path(parts[-1]).suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    return str(path)


def real_output_path(*parts):
    return real_run_path(*parts)
