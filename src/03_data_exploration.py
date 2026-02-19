import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
 
#%% BRING IN CLEAN DATA

df = pd.read_csv('cleaned_data.csv')

#%% Match Duration Outliers

# Plot the distribution of match durations
plt.figure(figsize=(10, 6))
plt.hist(df['game_duration'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Match Duration (seconds)')
plt.ylabel('Frequency')
plt.title('Distribution of Match Durations')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Compute IQR for match duration
Q1 = df['game_duration'].quantile(0.25)
Q3 = df['game_duration'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['game_duration'] < lower_bound) | (df['game_duration'] > upper_bound)]

# Remove outlier matches based on the upper bound
df_filtered = df[df['game_duration'] <= upper_bound]

# Plot the distribution of match durations
plt.figure(figsize=(10, 6))
plt.hist(df_filtered['game_duration'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Match Duration (seconds)')
plt.ylabel('Frequency')
plt.title('Distribution of Match Durations')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print(
df_filtered['game_duration'].max(),
df_filtered['game_duration'].min(),
df_filtered['game_duration'].mean())
# team and duration no longer needed:
# df_filtered = df_filtered.drop(['team','game_duration'],axis=1)

df_filtered.to_csv('filtered_df.csv',index=False)



#%% Role distribution visualizations

plt.figure(figsize=(12, 6))
df_primary_roles = df_filtered[[col for col in df_filtered.columns if col.startswith('primary_role_')]]
df_primary_roles.sum().sort_values().plot(kind='bar', color='royalblue', edgecolor='black', alpha=0.7)
plt.xlabel('Primary Role')
plt.ylabel('Total Count in Matches')
plt.title('Distribution of Primary Roles in ARAM Matches')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Identify primary role columns
role_columns = [col for col in df_filtered.columns if col.startswith('primary_role_')]

# Count matches that contain at least one of each role
matches_per_role = {role: (df_filtered[role] > 0).sum() for role in role_columns}

# Sort roles by match count in ascending order (so the highest is on the right)
sorted_roles = sorted(matches_per_role.items(), key=lambda x: x[1])

# Unpack sorted dictionary
sorted_role_names, sorted_role_counts = zip(*sorted_roles)

# Create a bar chart to visualize the counts
plt.figure(figsize=(12, 6))
bars = plt.bar(sorted_role_names, sorted_role_counts, color='royalblue', edgecolor='black', alpha=0.7)

# Add labels on top of bars
for bar, count in zip(bars, sorted_role_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(count), 
             ha='center', va='bottom', fontsize=12)

plt.xlabel('Primary Role')
plt.ylabel('Total Matches')
plt.title('Number of Teams Containing At Least One of Each Role')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

# Print the exact counts in sorted order
for role, count in sorted_roles:
    print(f"Total matches with at least one {role.replace('primary_role_', '')}: {count}")



#%%

# roles = [col for col in df_filtered.columns if 'role' in col]

# # Identify rare role counts (where a specific role count appears in fewer than 100 matches)
# roles_to_filter = []

# # Count occurrences of each role count across all role columns
# role_counts = {}

# for role in roles:
#     role_counts[role] = df_filtered[role].value_counts()

# # Convert to DataFrame for easier analysis
# role_count_df = pd.DataFrame(role_counts).fillna(0).astype(int).T
# role_count_df = role_count_df.reset_index()
# role_count_df = role_count_df.rename(columns={'index':'role'})

# for role in role_count_df['role']:
#     role_rows = role_count_df.loc[role_count_df['role']==role]
#     for col in role_rows.columns[1:]:
#         if role_rows[col].iloc[0] < 50 and role_rows[col].iloc[0] > 0:
#             roles_to_filter.append((role, col))

# matches_to_filter = []

# for rolecount in roles_to_filter:
#     role, count = rolecount[0], rolecount[1]
#     matches = df_filtered.loc[df_filtered[role]==count]['match_id'].unique().tolist()
#     matches_to_filter = matches_to_filter + matches
    
# df_filtered = df_filtered.loc[~df_filtered['match_id'].isin(matches_to_filter)]
# df_filtered["win"] = df_filtered["win"].astype(int)

# print(role_count_df)

# df_filtered.to_csv('final_df.csv',index=False)

# print(df_filtered.info(verbose=True))


#%%

# Load dataset (replace with actual path if needed)
# df_filtered = pd.read_csv('filtered_data.csv')

roles = [col for col in df_filtered.columns if 'role' in col]

### Step 1: Identify rare role counts

# Create a dictionary to count occurrences of each role count across all matches
role_counts = {role: df_filtered[role].value_counts() for role in roles}

# Convert to DataFrame for easier manipulation
role_count_df = pd.DataFrame(role_counts).fillna(0).astype(int).T.reset_index().rename(columns={'index': 'role'})

# List to track teams that need to be removed
teams_to_remove_v1 = []

# Loop through role count data to find rare cases (less than 50 matches)
for role in role_count_df['role']:
    role_rows = role_count_df.loc[role_count_df['role'] == role]
    for col in role_rows.columns[1:]:  # Exclude role name column
        if role_rows[col].iloc[0] < 50 and role_rows[col].iloc[0] > 0:
            teams_to_remove_v1 += df_filtered.loc[df_filtered[role] == int(col), ['match_id', 'team']].values.tolist()

# Convert list to DataFrame for easy filtering
teams_to_remove_v1_df = pd.DataFrame(teams_to_remove_v1, columns=['match_id', 'team'])

# Remove only the rare teams, keeping their opponent team in the dataset
df_filtered_v1 = df_filtered.merge(teams_to_remove_v1_df, on=['match_id', 'team'], how='left', indicator=True)
df_filtered_v1 = df_filtered_v1[df_filtered_v1['_merge'] == 'left_only'].drop(columns=['_merge'])

print(f"Version 1: Removed {len(teams_to_remove_v1)} team entries due to rare role counts.")

### Step 2: Remove teams with more than 3 of any role

# Identify teams where any role count exceeds 3
teams_to_remove_v2 = df_filtered.loc[df_filtered[roles].max(axis=1) > 3, ['match_id', 'team']].values.tolist()

# Convert to DataFrame for filtering
teams_to_remove_v2_df = pd.DataFrame(teams_to_remove_v2, columns=['match_id', 'team'])

# Remove only the teams with extreme stacking, keeping their opponents
df_filtered_v2 = df_filtered.merge(teams_to_remove_v2_df, on=['match_id', 'team'], how='left', indicator=True)
df_filtered_v2 = df_filtered_v2[df_filtered_v2['_merge'] == 'left_only'].drop(columns=['_merge'])

print(f"Version 2: Removed {len(teams_to_remove_v2)} team entries due to excessive role stacking.")

# Save results
df_filtered_v1.to_csv('final_df_v1.csv', index=False)
df_filtered_v2.to_csv('final_df_v2.csv', index=False)

# Print dataset info
print("Version 1 dataset info:")
print(df_filtered_v1.info(verbose=True))

print("Version 2 dataset info:")
print(df_filtered_v2.info(verbose=True))

df_filtered_v2.drop(['game_duration','team'],axis=1, inplace=True)
df_filtered_v2.to_csv('final_df.csv',index=False)
df_filtered = df_filtered_v2


#%%

# Separate winning and losing teams
win_group = df_filtered[df_filtered['win'] == 1]
loss_group = df_filtered[df_filtered['win'] == 0]

# Loop through all primary roles and generate charts for total wins and win rates
# Dictionary to store win rates by role count
win_rates_by_role_count = []

# Loop through each role and calculate win rate for each role count (0 to 5)
for role in roles:
    # Clean up the role name for display (removes "primary_role_" prefix)
    clean_role_name = role.replace("primary_role_", "").replace("_sum", "")

    # Aggregate total matches and wins for different counts of the role in a team
    role_stats = df_filtered.groupby(role)['win'].agg(['count', 'sum'])
    role_stats['win_rate'] = role_stats['sum'] / role_stats['count']
    
    # Store the results in a structured format
    for count in role_stats.index:
        win_rates_by_role_count.append({
            "Role": clean_role_name,
            "Role Count": count,
            "Total Matches": role_stats.loc[count, 'count'],
            "Wins": role_stats.loc[count, 'sum'],
            "Win Rate": role_stats.loc[count, 'win_rate']
        })

    # Plot win rate for the role
    plt.figure(figsize=(10, 6))
    bars = plt.bar(role_stats.index, role_stats['win_rate'], color='royalblue', edgecolor='black', alpha=0.7)
    plt.xlabel(f'Number of {clean_role_name}s in Team')
    plt.ylabel('Win Rate')
    plt.title(f'Win Rate by Number of {clean_role_name}s in Team')
    plt.xticks(range(4))
    plt.ylim(0, 1)
    plt.axhline(0.5, color='black', linestyle='--', alpha=0.7, label="Baseline (50%)")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2%}', ha='center', va='bottom')

    plt.show()

# Convert list to DataFrame for further analysis
win_rates_by_role_count_df = pd.DataFrame(win_rates_by_role_count)


#%%

# Extract average stat columns
avg_stat_columns = [col for col in df_filtered.columns if col.endswith('_mean')]

# Ensure only 4 plots are selected for the 2x2 grid (modify if necessary)
selected_stats = avg_stat_columns[:4]  # Adjust this if you want specific stats

# Compute averages for winning vs. losing teams
win_loss_averages = df_filtered.groupby('win')[selected_stats].mean()

# Print the averages for manual table creation
print("Average Champion Statistics for Wins vs. Losses:")
print(win_loss_averages.to_string())  # Ensures full table prints properly

# Create a 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Adjust figure size as needed
fig.suptitle("Comparison of Champion Statistics for Wins vs. Losses", fontsize=14)

# Loop through selected stats and plot
for ax, stat in zip(axes.flat, selected_stats):
    sns.boxplot(data=df_filtered, x='win', y=stat, palette=['red', 'lightblue'], ax=ax)
    ax.set_xlabel('Win (0 = Loss, 1 = Win)')
    ax.set_ylabel(stat.replace('_mean', ' (Avg)'))
    ax.set_title(stat.replace('_mean', ''))

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
plt.show()


# Print the averages for manual table creation
print("Average Champion Statistics for Wins vs. Losses:")
print(win_loss_averages.to_string())  # Ensures full table prints properly

#%%

# Compute correlation matrix
correlation_matrix = df_filtered.drop('match_id', axis=1).corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


