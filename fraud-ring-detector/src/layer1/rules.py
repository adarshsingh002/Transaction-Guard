import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def rule_card_testing(row):
    """
    card_testing_flag: velocity_1min > RULE_CARD_TESTING_VELOCITY AND amount < RULE_CARD_TESTING_AMOUNT
    """
    triggered = (row['velocity_1min'] > config.RULE_CARD_TESTING_VELOCITY) and (row['amount'] < config.RULE_CARD_TESTING_AMOUNT)
    return 'card_testing_flag', bool(triggered)

def rule_high_deviation(row):
    """
    high_deviation_flag: amount_zscore > RULE_HIGH_DEVIATION_ZSCORE AND new geo (geo_device_consistency == False)
    """
    triggered = (row['amount_zscore'] > config.RULE_HIGH_DEVIATION_ZSCORE) and (not row['geo_device_consistency'])
    return 'high_deviation_flag', bool(triggered)

def rule_odd_hour_new_category(row):
    """
    odd_hour_new_category_flag: is_unusual_hour AND is_new_merchant_category
    """
    triggered = row['is_unusual_hour'] and row['is_new_merchant_category']
    return 'odd_hour_new_category_flag', bool(triggered)

def evaluate_row(row, rules, weights):
    score = 0.0
    triggered = []
    has_hard_override = False
    
    for rule_func in rules:
        flag_name, is_triggered = rule_func(row)
        if is_triggered:
            triggered.append(flag_name)
            weight = weights.get(flag_name, 0.0)
            score += weight
            if weight >= 1.0:
                has_hard_override = True
                
    if has_hard_override:
        final_score = 1.0
    else:
        final_score = min(1.0, score)
        
    return pd.Series({'rule_score': final_score, 'triggered_rules': triggered})

def evaluate_rules(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates all rules on the features dataframe.
    Returns the dataframe with two new columns: 'rule_score' and 'triggered_rules'.
    """
    rules = [rule_card_testing, rule_high_deviation, rule_odd_hour_new_category]
    weights = config.RULE_WEIGHTS
    
    if len(features_df) == 0:
        features_df['rule_score'] = []
        features_df['triggered_rules'] = []
        return features_df

    # Apply rules per row as specified by requirements
    res = features_df.apply(lambda row: evaluate_row(row, rules, weights), axis=1)
    
    features_df['rule_score'] = res['rule_score']
    features_df['triggered_rules'] = res['triggered_rules']
    
    return features_df
