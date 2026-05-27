from pathlib import Path
import pandas as pd


def to_latex_table(df: pd.DataFrame, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(df.to_latex(index=False), encoding="utf-8")