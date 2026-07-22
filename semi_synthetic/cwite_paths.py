from pathlib import Path
import os


DATA_ROOT = Path(os.environ.get("CWITE_DATA_ROOT", "../data")).expanduser()
OUTPUT_ROOT = Path(os.environ.get("CWITE_OUTPUT_ROOT", "../outputs")).expanduser()


def data_path(*parts):
    return str(DATA_ROOT.joinpath(*parts))


def output_path(*parts):
    path = OUTPUT_ROOT.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def output_dir(*parts):
    path = OUTPUT_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
