"""
make_dataset.py

Purpose
-------
Creates a synthetic dataset of bonds with exactly 4 static features:
- bond_type
- currency
- time_to_maturity_days
- coupon_type

It assigns labels A/B/C/D using deterministic business rules.

Labels (FINAL rules used in this script)
----------------------------------------
A = USD + floating coupon + Agency bonds
B = USD + fixed coupon + Agency bond + time_to_maturity_days <= 365
C = CAD + Agency bond (coupon_type can be fixed OR floating; no differentiation)
D = USD + fixed coupon + Agency bond + time_to_maturity_days > 365

Output
------
Writes: <project_root>/data/bonds.csv
Columns: bond_type, currency, time_to_maturity_days, coupon_type, label
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def assign_label(currency: str, coupon_type: str, bond_type: str, ttm_days: int) -> str:
    """
    Deterministically assigns label A/B/C/D based on the rules above.

    Tie-break at 365:
    - ttm_days <= 365 -> B
    - ttm_days > 365  -> D
    """
    currency = currency.upper().strip()
    coupon_type = coupon_type.lower().strip()
    bond_type = bond_type.lower().strip()

    # This POC is for Agency universe only.
    if bond_type != "agency":
        raise ValueError(f"Unsupported bond_type={bond_type}. Only 'Agency' is allowed in this POC.")

    # C: CAD agency bonds (no differentiation on coupon type)
    if currency == "CAD":
        return "C"

    # USD rules
    if currency == "USD" and coupon_type == "floating":
        return "A"

    if currency == "USD" and coupon_type == "fixed":
        return "B" if ttm_days <= 365 else "D"

    # If any unexpected values appear, fail loudly (no silent random mapping).
    raise ValueError(
        f"Unsupported combination: currency={currency}, coupon_type={coupon_type}, bond_type={bond_type}, ttm_days={ttm_days}"
    )


def main(n_rows: int = 5000, seed: int = 42) -> None:
    """
    Generates a dataset with controlled randomness.

    Design choices
    -------------
    - bond_type: always Agency (to match your defined label universe)
    - currency: USD or CAD
    - coupon_type:
        - if CAD: fixed OR floating allowed (both map to label C)
        - if USD: fixed OR floating allowed (maps to A/B/D depending on ttm)
    - time_to_maturity_days: integer range ~[1..2000], skewed to include many around 365 boundary
    """
    rng = np.random.default_rng(seed)

    # Always Agency
    bond_types = np.array(["Agency"] * n_rows)

    # Currency split
    currencies = rng.choice(["USD", "CAD"], size=n_rows, p=[0.82, 0.18], replace=True)

    # Coupon type generation
    # (CAD can be fixed or floating; USD can be fixed or floating)
    coupon_types = rng.choice(["fixed", "floating"], size=n_rows, p=[0.75, 0.25], replace=True)

    # Time to maturity: bias near 365
    near_boundary = rng.integers(300, 430, size=int(n_rows * 0.45))
    broad = rng.integers(1, 2000, size=n_rows - len(near_boundary))
    ttm_days = np.concatenate([near_boundary, broad])
    rng.shuffle(ttm_days)

    # Labels
    labels = [
        assign_label(
            currency=currencies[i],
            coupon_type=coupon_types[i],
            bond_type=bond_types[i],
            ttm_days=int(ttm_days[i]),
        )
        for i in range(n_rows)
    ]

    df = pd.DataFrame(
        {
            "bond_type": bond_types,
            "currency": currencies,
            "time_to_maturity_days": ttm_days.astype(int),
            "coupon_type": coupon_types,
            "label": labels,
        }
    )

    # Save to project-root/data/bonds.csv regardless of working directory
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    out_path = data_dir / "bonds.csv"
    df.to_csv(out_path, index=False)

    print(f"✅ Wrote dataset: {out_path}")
    print("Label counts:")
    print(df["label"].value_counts())
    print("\nSample rows:")
    print(df.head(8))


if __name__ == "__main__":
    main()
