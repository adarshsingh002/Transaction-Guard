import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RANDOM_SEED = 42

# Thresholds
LAYER1_ML_THRESHOLD = 0.8
LAYER1_RULE_THRESHOLD = 50
