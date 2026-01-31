"""
predict.py

Purpose
-------
Loads the trained model (<project_root>/models/model.joblib) and predicts a label A/B/C/D
for a single bond input using exactly 4 static features:
- bond_type
- currency
- time_to_maturity_days
- coupon_type

Design choice
-------------
1) Validate the input is within the supported demo domain.
2) If outside domain -> print "UNKNOWN".
3) If within domain -> feed the input to the trained ML pipeline and print the prediction.

Usage (terminal)
----------------
python src/predict.py --bond_type Agency --currency CAD --time_to_maturity_days 800 --coupon_type floating
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


SUPPORTED_CURRENCIES = {"USD", "CAD"}
SUPPORTED_BOND_TYPE = "AGENCY"
SUPPORTED_COUPON_TYPES = {"fixed", "floating"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict trading book label A/B/C/D from 4 static features."
    )
    parser.add_argument("--bond_type", required=True, type=str, help="Agency")
    parser.add_argument("--currency", required=True, type=str, help="USD or CAD")
    parser.add_argument("--time_to_maturity_days", required=True, type=int, help="Integer days to maturity (>= 0)")
    parser.add_argument("--coupon_type", required=True, type=str, help="fixed or floating")
    return parser.parse_args()


def is_supported_input(bond_type: str, currency: str, coupon_type: str, ttm_days: int) -> bool:
    bond_type_n = bond_type.strip().upper()
    currency_n = currency.strip().upper()
    coupon_type_n = coupon_type.strip().lower()

    if bond_type_n != SUPPORTED_BOND_TYPE:
        return False
    if currency_n not in SUPPORTED_CURRENCIES:
        return False
    if coupon_type_n not in SUPPORTED_COUPON_TYPES:
        return False
    if ttm_days < 0:
        return False
    return True


def main() -> None:
    args = parse_args()

    # Normalize inputs
    bond_type = args.bond_type.strip()
    currency = args.currency.strip().upper()
    coupon_type = args.coupon_type.strip().lower()
    ttm_days = int(args.time_to_maturity_days)

    # Minimal domain gate (no guessing outside supported values)
    if not is_supported_input(bond_type, currency, coupon_type, ttm_days):
        print("UNKNOWN")
        return

    # Robust path: always load model from project root /models/model.joblib
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python src/train_evaluate.py to create it."
        )

    model = joblib.load(model_path)

    X = pd.DataFrame(
        [{
            "bond_type": bond_type,
            "currency": currency,
            "time_to_maturity_days": ttm_days,
            "coupon_type": coupon_type,
        }]
    )

    pred = model.predict(X)[0]
    print(pred)


if __name__ == "__main__":
    main()
