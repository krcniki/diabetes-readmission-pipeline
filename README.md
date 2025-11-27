# ML_Fall2025_Project

## 📌 Project Overview
An extensive analysis of hospital readmission data to predict if a diabetic patient will be readmitted within 30 days. This project solves a heavy **Class Imbalance** problem (only 10% positive class) using custom feature engineering and automated hyperparameter tuning.

## 🏗 Architecture
This project uses a modular production-style structure:
- **`src/`**: Contains helper modules for preprocessing and evaluation.
- **`tune.py`**: Dedicated script for Hyperparameter Optimization (RandomizedSearch).
- **`train.py`**: Main training pipeline with SMOTE and dynamic parameter loading.

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run optimization: `python tune.py` (Saves best params to json)
3. Train models: `python train.py` (Loads json and logs results)