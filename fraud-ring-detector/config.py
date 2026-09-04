import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RANDOM_SEED = 42

# Thresholds
LAYER1_ML_THRESHOLD = 0.8
LAYER1_RULE_THRESHOLD = 50

# Data Generation Settings
N_ACCOUNTS = 2000
N_TRANSACTIONS = 50000
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"
NUM_RINGS = 8
