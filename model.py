import xgboost
from get_data import extract_training_data, extract_prediction_data, get_data
from build_features import create_features
import requests

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


def transmit_text(text):
    slack_webhook_url = "https://hooks.slack.com/services/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"
    print(f"Transmitting: '{text}'")
    transmit = requests.post(slack_webhook_url, json={"text": text})
    print(transmit.status_code, transmit.reason)


def transmit_predictions():
    data = get_data()
    print("Creating trained model...")
    model = create_trained_model(data)
    print("predicting on current proposals..")
    predictions = predict_current_proposals(model, data)
    predictions['prediction'] = round(predictions['prediction'], 2)
    preboosted = predictions[predictions.stage == 'PreBoosted']
    queued = predictions[predictions.stage == 'Queued']

    preboosted.apply(lambda row:
                     transmit_text(f"PreBoosted proposal: {row.title} has a {row.prediction} chance of passing."),
                     axis=1)
    transmit_text("Queued proposal predictions are highly experimental!!! but,")
    queued.apply(lambda row:
                     transmit_text(f"Queued proposal: {row.title} currently has a {row.prediction} chance of passing."),
                     axis=1)


if __name__ == "__main__":
    transmit_predictions()
