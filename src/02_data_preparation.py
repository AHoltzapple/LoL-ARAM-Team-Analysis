#%% IMPORTS

import json
import pandas as pd

#%% JSON FILE PARSE

# Load JSON file
file_path = "matches.json"
with open(file_path, "r") as f:
    json_data = json.load(f)

# Process data
match_data = []
for match_id, match_details in json_data.items():
    game_duration = match_details["game_duration"]
    team1_result = match_details["team1_result"]
    team2_result = match_details["team2_result"]
    for i, player in enumerate(match_details["players"]):
        team = 1 if i < 5 else 2
        win = team1_result if team == 1 else team2_result
        match_data.append({
            "match_id": match_id,
            "team": team,
            "champion_id": player["championId"],
            "champion_name": player["championName"],
            "game_duration": game_duration,
            "win": win})
 
# Convert to DataFrame
df_matches = pd.DataFrame(match_data)

# match_counts = df_matches["match_id"].value_counts()
# invalid_matches = match_counts[match_counts != 10]

# print(f"Total matches not having exactly 10 players: {len(invalid_matches)}")
# print(invalid_matches)

# Display the first few rows
print(df_matches.head())
print('\n')
print(df_matches.info(verbose=True))


#%% MERGE CHAMP INFO

# Load champion data
champ_info_df = pd.read_csv("champ_info.csv")

# Load match data
df_matches = pd.read_csv('matches_df.csv')

df_matches["champion_id"] = df_matches["champion_id"].astype(int)
champ_info_df["champion_id"] = champ_info_df["key"].astype(int)

df_merged = df_matches.merge(champ_info_df, on="champion_id", how="left")

print(df_merged.info(verbose=True))


#%% DROP STAT COLUMNS

cols = ['match_id', 'team', 'game_duration',
       'win', 'primary_role', 'attack', 'defense', 'magic', 'difficulty']

df_merged = df_merged[cols]

df_merged = pd.get_dummies(df_merged, columns=["primary_role"], drop_first=False)

print(df_merged.info(verbose=True))

#%% AGGREGATIONS & CLEARNING

# Define champion stats to aggregate
champion_stats = ["attack", "defense", "magic", 'difficulty']

# Define roles and ranged column (sum/count)
role_columns = [
    "primary_role_Assassin", "primary_role_Fighter", "primary_role_Mage",
    "primary_role_Marksman", "primary_role_Support", "primary_role_Tank"]

# Define aggregation functions
agg_funcs = {col: ["mean"] for col in champion_stats}
agg_funcs.update({col: "sum" for col in role_columns})  # Sum for roles & ranged
agg_funcs["game_duration"] = "mean"  # Retain game duration
agg_funcs['win'] = 'max'

# Aggregate the dataset by match and team
df_team = df_merged.groupby(["match_id", "team"]).agg(agg_funcs).reset_index()

# Flatten MultiIndex columns
df_team.columns = ['_'.join(col).strip('_') for col in df_team.columns]

df_team = df_team.rename(columns={'game_duration_mean':'game_duration','win_max':'win'})

print(df_team.head())
print(df_team.info(verbose=True))

df_team.to_csv('cleaned_data.csv', index=False)
