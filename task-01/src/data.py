import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

COLUMNS = (
    [f"X{i}" for i in range(1, 17)]
    + [f"Y{i}" for i in range(1, 17)]
    + [f"P{i}" for i in range(1, 17)]
    + ["Powerall"]
)

def load_dataset(scenario: str) -> pd.DataFrame:
    path = DATA_DIR / f"{scenario}_Data.csv"
    df = pd.read_csv(path, header=None)
    df.columns = COLUMNS

    return df