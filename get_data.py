import requests
import numpy as np
import pandas as pd


def _run_query(query):
    '''
    simple query function found here: https://gist.github.com/gbaman/b3137e18c739e0cf98539bf4ec4366ad
    '''
    request = requests.post('https://api.thegraph.com/subgraphs/name/daostack/v36_8',
                            json={'query': query})
    if request.status_code == 200:
        return request.json()
    else:
        raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))


query = """{ 
    dao(id: "0x294f999356ed03347c7a23bcbcf8d33fa41dc830"){
        name  
        proposals(first: 1000){
            id
            title
            createdAt
            closingAt
            boostedAt
            preBoostedAt
            proposer
            quietEndingPeriodBeganAt
            votingMachine
            votesFor
            votesAgainst
            fulltext
            description
            winningOutcome
            totalRepWhenCreated
            totalRepWhenExecuted
            executedAt
            stage
            confidence
            confidenceThreshold
            contributionReward{
                ethReward
                reputationReward
                nativeTokenReward
                externalTokenReward
            }
            votes{
                id
                createdAt
                reputation
                voter
                outcome
            }
            stakesFor
            stakesAgainst
            stakes{
                createdAt
                staker
                outcome
                amount
            }
        }
    }
}"""


def _make_time_readable(data):
    time_columns = ['boostedAt', 'closingAt', 'createdAt', 'preBoostedAt', 'quietEndingPeriodBeganAt', 'executedAt']
    for time_column in time_columns:
        data[time_column] = pd.to_datetime(data[time_column], unit='s')
    return data


def _convert_number_columns(data):
    number_columns = ['confidence', 'confidenceThreshold', 'stakesAgainst', 'stakesFor', 'totalRepWhenCreated',
                      'totalRepWhenExecuted', 'votesAgainst', 'votesFor', 'ethReward', 'reputationReward',
                      'nativeTokenReward', 'externalTokenReward']
    for number_column in number_columns:
        data[number_column] = data[number_column].astype('float')
    return data


def get_proposal_data(i, which, data):
    if which == 'votes':
        number_cols = ['reputation']
    elif which == 'stakes':
        number_cols = ['amount']
    else:
        raise Exception(f"parameter 'which' must be either 'stakes' or 'votes', not '{which}'!")

    ret = pd.DataFrame(data.loc[i, which])
    if ret.empty:
        raise Exception(f"Woops, no {which} data at line {i}!")
    ret = _make_time_readable(ret, ['createdAt'])
    ret = _convert_number_columns(ret, number_cols)
    return ret


def _get_preboost_votes_data(i, boosted, data):
    try:
        votes = get_proposal_data(i=i, which='votes', data=data)
    except Exception as exc:
        print(exc)
        return [0,0,0,0]
    if boosted:
        votes = votes[votes.createdAt <= data.loc[i, 'preBoostedAt']].copy()
    forVotes = votes[votes.outcome=='Pass']
    againstVotes = votes[votes.outcome=='Fail']
    forVoters = forVotes.voter.unique()
    againstVoters = againstVotes.voter.unique()
    forVotesRep = forVotes.reputation.sum()
    againstVotesRep = againstVotes.reputation.sum()
    return [forVoters, againstVoters, forVotesRep, againstVotesRep]


def _get_stakes_data(i, data):
    try:
        stakes = get_proposal_data(i=i, which='stakes', data=data)
    except Exception as exc:
        print(exc)
        return [0,0,None,None]
    against_stakers_df = stakes[stakes.outcome=='Fail']
    for_stakers_df = stakes[stakes.outcome=='Pass']
    against_stakers_count = len(against_stakers_df)
    for_stakers_count = len(for_stakers_df)
    for_stakers = for_stakers_df.staker.unique()
    against_stakers = against_stakers_df.staker.unique()
    return [for_stakers_count, against_stakers_count, for_stakers, against_stakers]


def _extract_data(data, boosted=True):
    newVotingCols = ['for_preboost_voters','against_preboost_voters','preboost_for_rep','preboost_against_rep']
    newTokenCols = ['ethReward', 'reputationReward', 'nativeTokenReward', 'externalTokenReward']
    newStakesCols = ['for_stakers_count', 'against_stakers_count', 'for_stakers', 'against_stakers']
    data = data.copy()
    for col in newVotingCols + newTokenCols + newStakesCols:
        data[col] = np.nan
    for i, row in data.iterrows():
        data.loc[i, newVotingCols] = _get_preboost_votes_data(i, boosted, data)
        data.loc[i, newStakesCols] = _get_stakes_data(i, data)
        contributionReward = data.loc[i, 'contributionReward']
        if contributionReward:
            data.loc[i, newTokenCols] = contributionReward.values()
    return data


def _wrangle_data(data):
    data = _make_time_readable(data)
    data = _extract_data(data)
    data = _convert_number_columns(data)
    data.dropna(subset=['preBoostedAt'], inplace=True)  # we will only be looking at preboosted proposals
    # expired means failed even if the votes are net positive:
    data.loc[data.stage == 'ExpiredInQueue', 'winningOutcome'] = 'Fail'
    return data




def get_data_for_training():
    result = _run_query(query)
    proposals = result['data']['dao']['proposals']
    df = pd.DataFrame(data=proposals)
    df = df[(df.stage == 'Executed') | (df.stage == 'ExpiredInQueue')]
    df = _wrangle_data(df)
    return df



def create_features(train_data, test_data):
    return test_data_with_features

def create_and_train_model(data_with_features):

    return trained_model

def get_data_for_prediction():

    return data


def predict(trained_model, data_for_prediction):
    return prediction


def transmit_predictions(prediction):

    return slack_message