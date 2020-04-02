import xgboost
from get_data import extract_training_data, extract_prediction_data, get_data
from build_features import create_features
import requests
import sys

FEATURES = ['win_amount_ratio', 'reputation_score', 'confidence']


def create_trained_model(data=None):
    if data is None:
        data = get_data()
    training_data = extract_training_data(data, verbose=False)
    training_set = create_features(training_data, training_data)
    xgb_model = xgboost.XGBClassifier(objective="binary:logistic", random_state=6, eval_metric='aucpr')
    X = training_set[FEATURES]
    y = training_set['winningOutcome']
    xgb_model.fit(X, y)
    return xgb_model


def predict_current_proposals(trained_model, data=None):
    if data is None:
        data = get_data()
    training_data = extract_training_data(data, verbose=False)
    x = create_features(training_data, extract_prediction_data(data, verbose=False))
    x['prediction'] = [prediction[0] for prediction in trained_model.predict_proba(x[FEATURES])]
    return x


def transmit_text(text, webhook_url):
    print(f"Transmitting: '{text}'")
    transmit = requests.post(webhook_url, json={"text": text})
    print(transmit.status_code, transmit.reason)


def transmit_predictions(webhook_url):
    data = get_data()
    print("Creating trained model...")
    model = create_trained_model(data)
    print("predicting on current proposals..")
    predictions = predict_current_proposals(model, data)
    predictions['prediction'] = round(predictions['prediction'], 2)
    preboosted = predictions[predictions.stage == 'PreBoosted']
    queued = predictions[predictions.stage == 'Queued']
    preboosted.apply(lambda row:
                     transmit_text(f"Proposal PreBoosted at {row.preBoostedAt} "
                                   f"titled '{row.title}' has a {row.prediction} chance of passing.",
                                   webhook_url), axis=1)
    queued.apply(lambda row:
                     transmit_text(f"~~EXPERIMENTAL~~ Queued proposal created at {row.createdAt} "
                                   f"titled '{row.title}' currently has a {row.prediction} chance of passing.",
                                   webhook_url), axis=1)


def main(argv):
    transmit_predictions(argv)


if __name__ == "__main__":
    main(sys.argv[1])
