# Trading Book Identifier (Supervised ML)




https://github.com/user-attachments/assets/990a0470-cf08-482d-a1ed-2cb898e85aa0



This repository demonstrates a **supervised machine learning classification** proof-of-concept that predicts a **trading book label** (A/B/C/D) from **exactly 4 static features**:

- `bond_type` (categorical)
- `currency` (categorical)
- `time_to_maturity_days` (numeric)
- `coupon_type` (categorical)

The dataset is **synthetic** (no confidential data) but shaped to resemble real inputs. The label logic is deterministic, and the ML model learns the mapping from examples.

---

## Bigger picture (production context)

In real trading operations, trades can occasionally be booked to the wrong trading book, creating downstream operational breaks and risk. This repository demonstrates how a supervised ML classifier can flag potentially mis-booked trades and suggest the most likely correct book based on static trade features. It can also be used as a pre-booking assist tool, proposing an appropriate book when the correct mapping is unclear.

---

## Problem Statement

Build a supervised ML classifier that identifies a trading book label based on 4 static features and evaluates performance using:

- **Accuracy**
- **Precision**
- **Recall**


---

## Labels (A/B/C/D)

This repo uses the following label mapping (as implemented in `make_dataset.py`):

- **A** = USD + floating coupon + Agency  
- **B** = USD + fixed coupon + Agency + `time_to_maturity_days <= 365`  
- **C** = CAD + Agency (any coupon type, any maturity)
- **D** = USD + fixed coupon + Agency + `time_to_maturity_days > 365`  

---

## Repository Structure

trading-book-identifier-ml/
├─ README.md
├─ requirements.txt

├─ data/
│ ├─ bonds.csv # created by make_dataset.py
├─ models/
├─ model.joblib # created by train_evaluate.py
└─ src/
├─ make_dataset.py
├─ train_evaluate.py
└─ predict.py


---
### 1) Install Python 3.11 (Homebrew)

>brew install python@3.11
### 2) Create and activate virtual environment
> /opt/homebrew/bin/python3.11 -m venv .venv

> source .venv/bin/activate 

### 3) Install dependencies
> python -m pip install --upgrade pip

> python -m pip install -r requirements.txt

---
### Running the Project

## Step 1 — Generate dataset
>python src/make_dataset.py

## Step 2 — Train + evaluate model
>python src/train_evaluate.py

## Step 3 — Predict label for a new input
> python src/predict.py --bond_type Agency --currency CAD --time_to_maturity_days 800 --coupon_type floating

Outputs a single label: A / B / C / D 


predict.py validates the input domain before calling the model.

If currency not in {USD, CAD} or bond_type not Agency or coupon_type not in {fixed, floating} or time_to_maturity_days < 0

"UNKNOWN" is returned
