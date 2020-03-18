"""import os
os.chdir('/home/itush/git/staking_bot') # TODO: drop these lines when done with debugging. """
import xgboost
from get_data import extract_training_data, extract_prediction_data, get_data
from build_features import create_features

FEATURES = ['win_amount_ratio', 'reputation_score', 'confidence']


def create_trained_model():
    training_data = extract_training_data(get_data(), verbose=False)
    training_set = create_features(training_data, training_data)
    xgb_model = xgboost.XGBClassifier(objective="binary:logistic", random_state=6, eval_metric='aucpr')
    X = training_set[FEATURES]
    y = training_set['winningOutcome']
    xgb_model.fit(X, y)
    return xgb_model


def predict_current_proposals(trained_model):
    data = get_data()
    training_data = extract_training_data(data, verbose=False)
    x = create_features(training_data, extract_prediction_data(data, verbose=False))
    x['prediction'] = [prediction[0] for prediction in trained_model.predict_proba(x[FEATURES])]
    return x


def transmit_predictions(prediction):
    pass

