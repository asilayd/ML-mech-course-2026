import numpy as np
import pandas as pd


def validate_dataset(df: pd.DataFrame) -> dict:
    """Проверка корректности исходного датасета WEC."""

    coord_cols = (
        [f"X{i}" for i in range(1, 17)]
        + [f"Y{i}" for i in range(1, 17)]
    )

    power_cols = [f"P{i}" for i in range(1, 17)]

    expected_cols = coord_cols + power_cols + ["Powerall"]

    if list(df.columns) != expected_cols:
        raise ValueError("Некорректная структура столбцов")

    coords = df[coord_cols]
    power = df[power_cols]
    total_power = df["Powerall"]

    power_error = (power.sum(axis=1) - total_power).abs()

    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "infinite_values": int(np.isinf(df.to_numpy()).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_layouts": int(coords.duplicated().sum()),

        "min_coordinate": coords.min().min(),
        "max_coordinate": coords.max().max(),
        "invalid_coordinates": int(
            ((coords < 0) | (coords > 566)).sum().sum()
        ),

        "negative_individual_power": int(
            (power < 0).sum().sum()
        ),
        "negative_total_power": int(
            (total_power < 0).sum()
        ),

        "mean_power_error_W": power_error.mean(),
        "max_power_error_W": power_error.max(),
        "median_power_error_W": power_error.median(),
        "p95_power_error_W": power_error.quantile(0.95),
        "p99_power_error_W": power_error.quantile(0.99),
        "p999_power_error_W": power_error.quantile(0.999),

        "power_error_above_001_W": int((power_error > 0.01).sum()),
        "power_error_above_1_W": int((power_error > 1).sum()),
        "power_error_above_100_W": int((power_error > 100).sum()),
        "power_error_above_1000_W": int((power_error > 1000).sum()),

        "max_relative_power_error_pct": (
            power_error / total_power.abs().replace(0, np.nan) * 100
        ).max(),
    }