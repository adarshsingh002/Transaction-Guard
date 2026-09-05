import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

# Layer 1 Rule Thresholds
RULE_CARD_TESTING_VELOCITY = 5
RULE_CARD_TESTING_AMOUNT = 50.0
RULE_HIGH_DEVIATION_ZSCORE = 4.0
RULE_WEIGHTS = {
    "card_testing_flag": 1.0,
    "high_deviation_flag": 0.8,
    "odd_hour_new_category_flag": 0.6
}

# Layer 1 ML Settings
TEST_SIZE = 0.2
LAYER1_RULE_WEIGHT = 0.4
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "models", "layer1_model.joblib")

# Layer 2 Graph Settings
REG_TIME_WINDOW_MINUTES = 360
EDGE_WEIGHT_SHARED_DEVICE = 1.0
EDGE_WEIGHT_SHARED_PAYMENT = 1.0
EDGE_WEIGHT_SHARED_SUBNET = 1.0
EDGE_WEIGHT_REG_PROXIMITY = 0.1

# Layer 2 Clustering & Scoring
LOUVAIN_MIN_COMPONENT_SIZE = 6
CLUSTER_SCORE_WEIGHTS = {
    'age_homogeneity': 0.2,
    'synchrony': 0.3,
    'identifier_density': 0.2,
    'kyc_lack': 0.15,
    'financial_behavior': 0.15
}

# Fusion Engine Thresholds
THRESHOLD_BLOCK = 0.85
THRESHOLD_REVIEW = 0.5
