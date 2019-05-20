import http.client
import json
from Team import *
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from sklearn.utils import shuffle
import numpy as np

print('Type in first team(please remember to type the full name with FC, AFC etc.)')
team1 = input()
print('type in second team')
team2 = input()
connection = http.client.HTTPConnection('api.football-data.org')
headers = {'X-Auth-Token': 'YOUR API TOKEN '}
connection.request('GET', '/v2/competitions/PL/standings', None, headers)
response = json.loads(connection.getresponse().read().decode())

dataConnection = http.client.HTTPConnection('api.football-data.org')
dataConnection.request('GET', '/v2/competitions/PL/matches', None, headers)
matches = json.loads(dataConnection.getresponse().read().decode())

tableRaw = response['standings']
generalTeamStanding = tableRaw[0]['table']
homeTeamStanding = tableRaw[1]['table']
awayTeamStanding = tableRaw[2]['table']
currentMatchday = matches['matches'][0]['season']['currentMatchday']


def get_stats(team_name, team_standing):  # getting team stats for convenient usage later on
    for i in range(len(team_standing)):
        team_position = team_standing[i]
        team_name_raw = team_position['team']
        if team_name_raw['name'] == team_name:
            return TeamStats(team_position['won'],
                             team_position['draw'],
                             team_position['lost'],
                             team_position['goalsFor'],
                             team_position['goalsAgainst'])


teams = []


for i in range(20):  # filling teams with teams... duh
    teamName = generalTeamStanding[i]['team']['name']
    teams.append(teamName)


homeTeams = []
awayTeams = []
homeTeamsGoals = []
awayTeamsGoals = []

teams = pd.get_dummies(teams)

for i in range(currentMatchday * 10):  # getting previously played matches
    home_team_score = matches['matches'][i]['score']['fullTime']['homeTeam']
    away_team_score = matches['matches'][i]['score']['fullTime']['awayTeam']
    if home_team_score is None or away_team_score is None:
        pass
    else:
        homeTeams.append(matches['matches'][i]['homeTeam']['name'])
        awayTeams.append(matches['matches'][i]['awayTeam']['name'])
        homeTeamsGoals.append(home_team_score)
        awayTeamsGoals.append(away_team_score)


homeGeneralWins = []
homeGeneralDraws = []
homeGeneralLosses = []
homeGoalsFor = []
homeGoalsAgainst = []
homeVenueWins = []
homeVenueDraws = []
homeVenueLosses = []
homeVenueGoalsFor = []
homeVenueGoalsAgainst = []
awayGeneralWins = []
awayGeneralDraws = []
awayGeneralLosses = []
awayGoalsFor = []
awayGoalsAgainst = []
awayVenueWins = []
awayVenueDraws = []
awayVenueLosses = []
awayVenueGoalsFor = []
awayVenueGoalsAgainst = []
for i in range(len(homeTeams)):  # getting data for the df
    homeTeamToDF = Team(homeTeams[i], get_stats(homeTeams[i], generalTeamStanding), get_stats(homeTeams[i], homeTeamStanding))
    awayTeamToDF = Team(awayTeams[i], get_stats(homeTeams[i], generalTeamStanding), get_stats(awayTeams[i], awayTeamStanding))
    homeGeneralWins.append(homeTeamToDF.stats.wins)
    homeGeneralDraws.append(homeTeamToDF.stats.draws)
    homeGeneralLosses.append(homeTeamToDF.stats.losses)
    homeGoalsFor.append(homeTeamToDF.stats.goals_for)
    homeGoalsAgainst.append(homeTeamToDF.stats.goals_against)
    homeVenueWins.append(homeTeamToDF.venue_stats.wins)
    homeVenueDraws.append(homeTeamToDF.venue_stats.draws)
    homeVenueLosses.append(homeTeamToDF.venue_stats.losses)
    homeVenueGoalsFor.append(homeTeamToDF.venue_stats.goals_for)
    homeVenueGoalsAgainst.append(homeTeamToDF.venue_stats.goals_against)
    awayGeneralWins.append(awayTeamToDF.stats.wins)
    awayGeneralDraws.append(awayTeamToDF.stats.draws)
    awayGeneralLosses.append(awayTeamToDF.stats.losses)
    awayGoalsFor.append(awayTeamToDF.stats.goals_for)
    awayGoalsAgainst.append(awayTeamToDF.stats.goals_against)
    awayVenueWins.append(awayTeamToDF.venue_stats.wins)
    awayVenueDraws.append(awayTeamToDF.venue_stats.draws)
    awayVenueLosses.append(awayTeamToDF.venue_stats.losses)
    awayVenueGoalsFor.append(awayTeamToDF.venue_stats.goals_for)
    awayVenueGoalsAgainst.append(awayTeamToDF.venue_stats.goals_against)

values = []
for i in range(20):
    values.append(i)
    values[i] += 1
keys = list(teams.columns.values)
teams = dict(zip(keys, values))

for i in range(len(homeTeams)):  # converting data to labels
    homeTeams[i] = teams[homeTeams[i]]
    awayTeams[i] = teams[awayTeams[i]]

winners = []
winnersNames = []
for i in range(len(homeTeams)):  # getting winners from previously played matches
    if homeTeamsGoals[i] > awayTeamsGoals[i]:
        winners.append(homeTeams[i])
        winnersNames.append(homeTeams[i])
    elif homeTeamsGoals[i] < awayTeamsGoals[i]:
        winners.append(awayTeams[i])
        winnersNames.append(awayTeams[i])
    else:
        draw = 0
        winners.append(draw)
        winnersNames.append('Draw')

df = pd.DataFrame({'Home Team': homeTeams,
                   'Away Team': awayTeams,
                   'Home Goals': homeTeamsGoals,
                   'Away Goals': awayTeamsGoals,
                   'Winner': winners,
                   'Winner Name': winnersNames,
                   'homeGeneralWins': homeGeneralWins,
                   'homeGeneralDraws': homeGeneralDraws,
                   'homeGeneralLosses': homeGeneralLosses,
                   'homeGoalsFor': homeGoalsFor,
                   'homeGoalsAgainst': homeGoalsAgainst,
                   'homeVenueWins': homeVenueWins,
                   'homeVenueDraws': homeVenueDraws,
                   'homeVenueLosses': homeVenueLosses,
                   'homeVenueGoalsFor': homeVenueGoalsFor,
                   'homeVenueGoalsAgainst': homeVenueGoalsAgainst,
                   'awayGeneralWins': awayGeneralWins,
                   'awayGeneralDraws': awayGeneralDraws,
                   'awayGeneralLosses': awayGeneralLosses,
                   'awayGoalsFor': awayGoalsFor,
                   'awayGoalsAgainst': awayGoalsAgainst,
                   'awayVenueWins': awayVenueWins,
                   'awayVenueDraws': awayVenueDraws,
                   'awayVenueLosses': awayVenueLosses,
                   'awayVenueGoalsFor': awayVenueGoalsFor,
                   'awayVenueGoalsAgainst': awayVenueGoalsAgainst,
                   })
print()


def preprocess_features(data):  # processing features for the NN
    selected_features = data[['Home Team',
                              'Away Team',
                              'homeGeneralWins',
                              'homeGeneralDraws',
                              'homeGeneralLosses',
                              'homeGoalsFor',
                              'homeGoalsAgainst',
                              'homeVenueWins',
                              'homeVenueDraws',
                              'homeVenueLosses',
                              'homeVenueGoalsFor',
                              'homeVenueGoalsAgainst',
                              'awayGeneralWins',
                              'awayGeneralDraws',
                              'awayGeneralLosses',
                              'awayGoalsFor',
                              'awayGoalsAgainst',
                              'awayVenueWins',
                              'awayVenueDraws',
                              'awayVenueLosses',
                              'awayVenueGoalsFor',
                              'awayVenueGoalsAgainst']]
    return selected_features


def preprocess_targets(data):  # processing targets
    output_targets = data['Winner']
    return output_targets


df = shuffle(df)  # shuffle dataframe so it's not the same all the time

training_examples = preprocess_features(df.head(len(df.index) - 10))
training_targets = preprocess_targets(df.head(len(df.index) - 10))

validation_examples = preprocess_features(df.tail(10))
validation_targets = preprocess_targets(df.tail(10))

model = keras.Sequential([
    keras.layers.Dense(22),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(21, activation=tf.nn.softmax)
])


model.compile(optimizer='RMSprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_examples.values, training_targets.values, batch_size=4, epochs=150)

team1 = Team(team1, get_stats(team1, generalTeamStanding), get_stats(team1, homeTeamStanding))
team2 = Team(team2, get_stats(team2, generalTeamStanding), get_stats(team2, awayTeamStanding))

model_input = pd.DataFrame({'team1 name': teams[team1.name],
                            'team2 name': teams[team2.name],
                            'team1 wins': team1.stats.wins,
                            'team1 draws': team1.stats.draws,
                            'team1 losses': team1.stats.losses,
                            'team1 goals_for': team1.stats.goals_for,
                            'team1 goals_against': team1.stats.goals_against,
                            'team1 venue_wins': team1.venue_stats.wins,
                            'team1 venue_draws': team1.venue_stats.draws,
                            'team1 venue losses': team1.venue_stats.losses,
                            'team1 venue goals_for': team1.venue_stats.goals_for,
                            'team1 venue goals_against': team1.venue_stats.goals_against,
                            'team2 wins': team2.stats.wins,
                            'team2 draws': team2.stats.draws,
                            'team2 losses': team2.stats.losses,
                            'team2 goals_for': team2.stats.goals_for,
                            'team2 goals_against': team2.stats.goals_against,
                            'team2 away wins': team2.venue_stats.wins,
                            'team2 away draws': team2.venue_stats.draws,
                            'team2 away losses': team2.venue_stats.losses,
                            'team2 away goals_for': team2.venue_stats.goals_for,
                            'team2 away goals_against': team2.venue_stats.goals_against}, index=[0])


prediction = model.predict(model_input)

result = np.argmax(prediction)
print('siema')
if result == 0:
    print('Draw')
else:
    print(keys[result-1])
    print('is going to win')
