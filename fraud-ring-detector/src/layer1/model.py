"""Layer 1 ML model."""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from schemas import Layer1Output, TopFeature

def prepare_data(df: pd.DataFrame):
    """
    Prepares the features dataframe for ML training/prediction.
    Selects numerical and boolean features, converting them as needed.
    """
    df_prep = df.copy()
    
    # Extract binary rule flags if they exist
    if 'triggered_rules' in df_prep.columns:
        df_prep['card_testing_flag'] = df_prep['triggered_rules'].apply(lambda x: 'card_testing_flag' in x).astype(int)
        df_prep['high_deviation_flag'] = df_prep['triggered_rules'].apply(lambda x: 'high_deviation_flag' in x).astype(int)
        df_prep['odd_hour_new_category_flag'] = df_prep['triggered_rules'].apply(lambda x: 'odd_hour_new_category_flag' in x).astype(int)
    else:
        df_prep['card_testing_flag'] = 0
        df_prep['high_deviation_flag'] = 0
        df_prep['odd_hour_new_category_flag'] = 0
    
    feature_cols = [
        'velocity_1min', 'velocity_1hr', 'velocity_24hr',
        'amount_zscore', 'is_new_merchant_category', 'geo_device_consistency',
        'is_unusual_hour', 'failure_retry_ratio',
        'card_testing_flag', 'high_deviation_flag', 'odd_hour_new_category_flag'
    ]
    
    for col in ['is_new_merchant_category', 'geo_device_consistency', 'is_unusual_hour']:
        if col in df_prep.columns:
            df_prep[col] = df_prep[col].astype(int)
            
    X = df_prep[feature_cols].copy()
    y = df_prep['label_fraud'].astype(int) if 'label_fraud' in df_prep.columns else None
    
    return X, y, feature_cols

def train(df_features: pd.DataFrame):
    """Trains the layer 1 machine learning model."""
    X, y, feature_cols = prepare_data(df_features)
    
    if y is None:
        raise ValueError("Training data must contain 'label_fraud' column.")
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_SEED
    )
    
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=config.RANDOM_SEED,
        eval_metric='logloss'
    )
    
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump((model, feature_cols), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")
    
    return model, X_test, y_test

def combine_layer1_score(rule_score: float, ml_probability: float) -> float:
    """
    Blends rule engine score and ML probability.
    - If rule_score >= 1.0 (hard-override), force score >= 0.9.
    - Otherwise, weighted average based on LAYER1_RULE_WEIGHT.
    """
    if rule_score >= 1.0:
        return max(0.9, ml_probability)
    
    return float(rule_score * config.LAYER1_RULE_WEIGHT + ml_probability * (1 - config.LAYER1_RULE_WEIGHT))

def predict(df_features: pd.DataFrame) -> list[Layer1Output]:
    """Generates predictions using the layer 1 model and formats output."""
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model not found at {config.MODEL_SAVE_PATH}")
        
    model, feature_cols = joblib.load(config.MODEL_SAVE_PATH)
    X, _, _ = prepare_data(df_features)
    
    ml_probs = model.predict_proba(X)[:, 1]
    importances = model.feature_importances_
    
    feature_means = X.mean()
    feature_stds = X.std().replace(0, 1) # avoid div by zero
    
    outputs = []
    
    for idx, (original_index, row) in enumerate(df_features.iterrows()):
        txn_id = row['txn_id']
        rule_score = row.get('rule_score', 0.0)
        ml_prob = ml_probs[idx]
        
        layer1_score = combine_layer1_score(rule_score, ml_prob)
        triggered_rules = row.get('triggered_rules', [])
        
        # Heuristic for feature contribution: global importance * local deviation
        x_row = X.iloc[idx]
        local_importance = importances * np.abs((x_row - feature_means) / feature_stds)
        
        # Get top 2 features
        top_idx = np.argsort(local_importance)[-2:][::-1]
        
        top_features = []
        for idx in top_idx:
            feat_name = feature_cols[idx]
            val = float(x_row[feat_name])
            imp_val = local_importance.iloc[idx]
            
            if imp_val > 0.5:
                contrib = "high"
            elif imp_val > 0.1:
                contrib = "medium"
            else:
                contrib = "low"
                
            top_features.append(TopFeature(
                feature=feat_name,
                value=val,
                contribution=contrib
            ))
            
        outputs.append(Layer1Output(
            txn_id=txn_id,
            layer1_score=round(layer1_score, 4),
            triggered_rules=triggered_rules,
            top_features=top_features
        ))
        
    return outputs
