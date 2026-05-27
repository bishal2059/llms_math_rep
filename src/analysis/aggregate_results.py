from pathlib import Path
import pandas as pd


def aggregate_csvs(folder: str | Path, pattern: str = "*.csv") -> pd.DataFrame:
    folder = Path(folder)
    frames = []
    for p in sorted(folder.glob(pattern)):
        df = pd.read_csv(p)
        df["source_file"] = p.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)