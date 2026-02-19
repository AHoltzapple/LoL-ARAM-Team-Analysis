# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 18:36:00 2025

@author: Arshlao25
"""

import json
import time
import requests
import pandas as pd

apikey = '[INPUT API KEY]'

#%%

def retrieve_champ_info():
    champs_r = requests.get('http://ddragon.leagueoflegends.com/cdn/15.2.1/data/en_US/champion.json')
    champs = json.loads(champs_r.text)['data']
    champion_data = []
    for champ in champs.values():
        roles = champ['tags']
        if len(roles) > 1:
            primary_role, secondary_role = roles[0], roles[1]
        else:
            primary_role, secondary_role = roles[0], None
        champ_info = {
            'name': champ['name'],
            'id': champ['id'],
            'key': champ['key'],
            'primary_role': primary_role,
            'secondary_role': secondary_role}
        champ_info.update(champ['info'])
        champ_info.update(champ['stats'])
        champion_data.append(champ_info)
    champions_df = pd.DataFrame(champion_data)
    return champions_df

champ_info = retrieve_champ_info()
champ_info.to_csv('champ_info.csv', index = False)

#%%

def retrieve_puuid(apikey, gamename, tag):
    acct_r = requests.get(
        'https://americas.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{}/{}?api_key={}'.format(gamename, tag, apikey))
    if acct_r.status_code != 200:
        return print('Received error: ' + str(acct_r.status_code))
    response = json.loads(acct_r.text)
    puuid = response['puuid']
    return puuid

# puuid = retrieve_puuid(apikey, 'Erythro25', 'NA1')


#%%

from datetime import datetime

def epoch_to_date(epoch_ms):
    """Converts Riot API epoch time (in milliseconds) to a human-readable date."""
    return datetime.utcfromtimestamp(epoch_ms).strftime('%Y-%m-%d %H:%M:%S')

def date_to_epoch(date_str):
    """Converts a date string (YYYY-MM-DD HH:MM:SS) to Riot API epoch time in milliseconds."""
    return int(datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').timestamp())

# Example usage
patch_date = "2025-01-23 05:00:00"
patch_epoch = date_to_epoch(patch_date)


#%%

def retrieve_matches(apikey, puuid, sample=20, starttime=''):
    if starttime != '':
        starttime = 'startTime={}&'.format(starttime)
    matchlist = []    
    matches = requests.get('https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{}/ids?{}queue=450&count={}&api_key={}'.format(puuid, starttime, sample, apikey))
    while matches.status_code == 429:
        print('Hit call limit, waiting 30s')
        time.sleep(30)
        matches = requests.get('https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{}/ids?{}queue=450&count={}&api_key={}'.format(puuid, starttime, sample, apikey))
    if matches.status_code != 200:
        ValueError('Received error: ' + str(matches.status_code))
    matchlist.extend(json.loads(matches.content))
    print('{} matches retrieved'.format(len(matchlist)))
    return matchlist

#%%

def retrieve_single_match(apikey, matchid):
    indiv_match_r = requests.get('https://americas.api.riotgames.com/lol/match/v5/matches/{}?api_key={}'.format(matchid, apikey))
    while indiv_match_r.status_code == 429:
        print('Hit call limit, waiting 30s')
        time.sleep(30)
        indiv_match_r = requests.get('https://americas.api.riotgames.com/lol/match/v5/matches/{}?api_key={}'.format(matchid, apikey))
    if indiv_match_r.status_code == 200:
        match_json = json.loads(indiv_match_r.text)
    else: raise ValueError('Invalid response code {}'.format(str(indiv_match_r.status_code)))
    return match_json


#%%

def parse_match_data(match_data):
    game_status = match_data["info"]["endOfGameResult"]
    game_mode = match_data["info"]["gameMode"]
    platform_id = match_data["info"]["platformId"]
    game_duration = match_data['info']['gameDuration']
    if game_status != "GameComplete" or game_mode != "ARAM" or platform_id != "NA1" or game_duration < 480:
        return None
    team_levels = {100: [], 200: []}
    player_data = []
    for participant in match_data["info"]["participants"]:
        champ_level = participant["champLevel"]
        team_id = participant["teamId"]
        player_info = {
            "puuid": participant["puuid"],
            "championId": participant["championId"],
            "championName": participant["championName"],
            "teamId": participant["teamId"]
            }
        player_data.append(player_info)
        team_levels[team_id].append(champ_level)
    for team_id, levels in team_levels.items():
        if levels:
            level_range = max(levels) - min(levels)
            if level_range > 3:
                return None
    team_results = {team["teamId"]: team["win"] for team in match_data["info"]["teams"]}
    return {
        "team1_result": team_results.get(100),  # Team 1 (100)
        "team2_result": team_results.get(200),  # Team 2 (200)
        "game_duration": game_duration,
        "players": player_data}

#%%

def save_match_data(match_dict, filename="matches.json"):
    """Saves match data to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(match_dict, f, indent=4)
    print(f"Saved {len(match_dict)} matches to {filename}")

# save_match_data(filtered_matches)

def load_match_data(filename="matches.json"):
    """Loads match data from a JSON file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            match_dict = json.load(f)
        print(f"Loaded {len(match_dict)} matches from {filename}")
        return match_dict
    except FileNotFoundError:
        print("No previous match data found. Starting fresh.")
        return {}

# filtered_matches = load_match_data()


def save_puuid_progress(puuids, index, filename="puuid_progress_personalkey.json"):
    """Saves PUUID list and index to a JSON file."""
    data = {
        "puuids": puuids,
        "index": index
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Progress saved. Next start index: {index}")

# save_puuid_progress(puuids, index)


def load_puuid_progress(filename="puuid_progress_personalkey.json"):
    """Loads PUUID list and index from a JSON file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Progress loaded. Resuming from index {data['index']}")
        return data["puuids"], data["index"]
    except FileNotFoundError:
        print("No saved progress found. Starting fresh.")
        return [], 0  # Start from scratch if no progress file exists

# puuids, index = load_puuid_progress()




#%%

# ### Initial collection start parameters:
# puuids = []
# filtered_matches = {}

# puuid1 = retrieve_puuid(apikey, '[USERNAME]', 'NA1')
# puuids.append(puuid1)

# index = 0
# max_matches = 20000

# ### Resuming data collection parameters:
# # filtered_matches = load_match_data()
# # puuids, index = load_puuid_progress()
    
# puuids_set = set(puuids)
# match_set = set(filtered_matches.keys())

# while len(filtered_matches) < max_matches and index < len(puuids):
#     save = False
#     puuid = puuids[index]  
#     puuids_set.add(puuid)
#     matches = retrieve_matches(apikey, puuid, sample=20, starttime=patch_epoch) # call 1

#     for matchid in matches:
#         if matchid in match_set:
#             continue
#         match_data = retrieve_single_match(apikey, matchid) # call 2
#         match_info = parse_match_data(match_data)
#         if not match_info:
#             continue
#         for player in match_info['players']:
#             if player['puuid'] not in puuids_set:
#                 puuids.append(player['puuid'])
#         filtered_matches[matchid] = match_info
#         match_set.add(matchid)
#         if len(filtered_matches) % 100 == 0:
#             save = True
#     print('Total Match Data: ' + str(len(filtered_matches)) + '\n')
#     index += 1
#     if save:
#         save_puuid_progress(puuids, index)
#         save_match_data(filtered_matches)
#         save = False

