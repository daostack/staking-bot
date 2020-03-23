
import numpy as np
import pandas as pd
from get_data import get_proposal_data


def _assign_ID_and_quantity(metric):
    if metric == 'stakes':
        ID = 'staker'
        quantity = 'amount'
    elif metric == 'votes':
        ID = 'voter'
        quantity = 'reputation'
    else:
        raise Exception(f"parameter metric must be either 'stakes' or 'votes' not '{metric}' !")
    return ID, quantity


def _build_metric_df(metric, data, verbose=False):
    ID, quantity = _assign_ID_and_quantity(metric)
    ret = pd.DataFrame(columns=[quantity, 'createdAt', 'outcome', ID])
    for i in data.index:
        try:
            proposal_df = get_proposal_data(i, which=metric, data=data)
        except Exception as exc:
            if verbose:
                print(exc)
            proposal_df = pd.DataFrame(columns=[quantity,'createdAt','outcome',ID])
        proposal_df['proposal'] = data.loc[i, 'title']
        proposal_df['winningOutcome'] = data.loc[i, 'winningOutcome']
        ret = pd.concat([ret, proposal_df], sort=False)
    return ret


def _build_IDs_df(metric, data):
    ID, quantity = _assign_ID_and_quantity(metric)
    metric_df = _build_metric_df(metric, data=data, verbose=False)
    winning_metric = metric_df[metric_df.outcome==metric_df.winningOutcome]
    losing_metric = metric_df[metric_df.outcome!=metric_df.winningOutcome]
    winnings = winning_metric.groupby(ID).agg({quantity: ['sum','count']})
    losings = losing_metric.groupby(ID).agg({quantity: ['sum','count']})
    IDs_df = winnings.join(losings, on=ID, how='outer', rsuffix='_lost', lsuffix='_won')
    IDs_df = IDs_df.replace(np.nan, 1) # treat no mistakes as 1 mistake
    IDs_df[f'win_count_ratio_{metric}'] = IDs_df[f'{quantity}_won', 'count'] / IDs_df[f'{quantity}_lost', 'count']
    IDs_df[f'win_{quantity}_ratio'] = IDs_df[f'{quantity}_won', 'sum'] / IDs_df[f'{quantity}_lost', 'sum']
    IDs_df[f'{metric}_count_score'] = IDs_df[f'{quantity}_won', 'count'] * IDs_df[f'win_count_ratio_{metric}']
    IDs_df[f'{quantity}_score'] = IDs_df[f'{quantity}_won', 'sum'] * IDs_df[f'win_{quantity}_ratio']
    return IDs_df


def _assign_score(row, metric, metricer, quantity, multiply_score_by_quantity, ID_list, score, scores_df):
    proposal_score = 0
    for ID in ID_list:
        try:
            ID_proposal_score = scores_df.loc[ID, score].values[0]
        except:
            ID_proposal_score = 0
        if ID_proposal_score != 0:

            quantity_value = 0
            proposal_IDs_list = row[metric]
            for proposal_id in proposal_IDs_list:
                if proposal_id[metricer] == ID:
                    quantity_value = quantity_value + int(proposal_id[quantity])
            if multiply_score_by_quantity:
                ID_proposal_score = ID_proposal_score * quantity_value
        proposal_score = proposal_score + ID_proposal_score
    return proposal_score


def _calc_proposal_ID_score(row, score, scores_df, multiply_score_by_quantity):
    if score not in scores_df.columns:
        raise Exception(f"sorry but {score} must be a column in your scores_df")
    metric = scores_df.index.name.replace("er", "es")
    metricer, quantity = _assign_ID_and_quantity(metric)
    metric_column = metricer
    if metric == 'votes':
        metric_column = 'preboost_voter'
    ID_lists = {'for': row[f'for_{metric_column}s'],
                'against': row[f'against_{metric_column}s']}
    ret = {}
    for for_or_against, ID_list in ID_lists.items():
        if not isinstance(ID_list, np.ndarray):
            return 0
        ret[for_or_against] = \
            _assign_score(row, metric, metricer, quantity, multiply_score_by_quantity, ID_list, score, scores_df)
    return ret['for'] - ret['against']


def _apply_score(df, scores, data, multiply_score_by_quantity=True):
    for score in scores:
        df[score] = df.apply(lambda row: _calc_proposal_ID_score(row, score, data, multiply_score_by_quantity), axis=1)


def create_features(training_set, test_set):
    stakers = _build_IDs_df(data=training_set, metric='stakes')
    voters = _build_IDs_df(data=training_set, metric='votes')
    staking_scores = ['win_amount_ratio']
    voting_scores = ['reputation_score']
    _apply_score(test_set, staking_scores, stakers)
    _apply_score(test_set, voting_scores, voters, multiply_score_by_quantity=False)
    return test_set

