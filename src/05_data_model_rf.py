# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 12:25:54 2025

@author: Arshlao25
"""
# Import necessary packages for analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('final_df.csv')
df = df.drop('match_id', axis=1)
df["win"] = df["win"].astype(int)

# Define role columns
roles = [col for col in df.columns if 'role' in col]

#%%

# Initialize Random Forest with basic parameters
rf_base = RandomForestClassifier(n_estimators=500, random_state=25, min_samples_leaf=50)

X = df.drop('win',axis=1) # Features
y = df["win"]  # Target

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25, stratify=y)

# Train on full dataset
rf_base.fit(X_train, y_train)

# Predict on test set
rf_preds = rf_base.predict(X_test)
rf_probs = rf_base.predict_proba(X_test)[:, 1]  # Probabilities for AUC

# Evaluate performance
rf_accuracy = accuracy_score(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf_probs)
rf_conf_matrix = confusion_matrix(y_test, rf_preds)

print(f"\nRandom Forest Base Model Accuracy: {rf_accuracy:.4f}")
print(f"Random Forest AUC-ROC: {rf_auc:.4f}")
print("\nConfusion Matrix:\n", rf_conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, rf_preds))


# Get feature importances
feature_importances = rf_base.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(np.array(X_train.columns)[sorted_idx], feature_importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()

#%%

# Initialize Random Forest with basic parameters
rf_base = RandomForestClassifier(n_estimators=500, random_state=25, min_samples_leaf=100)

X = df.drop(['win', 'defense_mean','magic_mean',
             'attack_mean','difficulty_mean'],axis=1) # Features
y = df["win"]  # Target

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25, stratify=y)

# Train on full dataset
rf_base.fit(X_train, y_train)

# Predict on test set
rf_preds = rf_base.predict(X_test)
rf_probs = rf_base.predict_proba(X_test)[:, 1]  # Probabilities for AUC

# Evaluate performance
rf_accuracy = accuracy_score(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf_probs)
rf_conf_matrix = confusion_matrix(y_test, rf_preds)

print(f"\nRandom Forest Base Model Accuracy: {rf_accuracy:.4f}")
print(f"Random Forest AUC-ROC: {rf_auc:.4f}")
print("\nConfusion Matrix:\n", rf_conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, rf_preds))


# Get feature importances
feature_importances = rf_base.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(np.array(X_train.columns)[sorted_idx], feature_importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()



#%%

# Extract a single tree from the Random Forest
single_tree = rf_base.estimators_[0]  # First tree in the forest
tree = single_tree.tree_

# Extract probabilities for each leaf node
leaf_nodes = np.where(tree.children_left == -1)[0]  # Terminal nodes
leaf_probs = {}

for leaf in leaf_nodes:
    class_counts = tree.value[leaf][0]  # Class distribution
    win_prob = class_counts[1] / sum(class_counts)  # Win probability
    leaf_probs[leaf] = win_prob

# Identify highest and lowest probability nodes
highest_prob_leaf = max(leaf_probs, key=leaf_probs.get)
lowest_prob_leaf = min(leaf_probs, key=leaf_probs.get)

print(f"\nHighest Win Probability: {leaf_probs[highest_prob_leaf]:.3f} (Leaf {highest_prob_leaf})")
print(f"Lowest Win Probability: {leaf_probs[lowest_prob_leaf]:.3f} (Leaf {lowest_prob_leaf})")

def get_decision_path(tree, node_id, path=[]):
    """ Recursively find the path from the root to a given leaf node using actual feature names. """
    if node_id == 0:  # Root node
        return path[::-1]  # Reverse order to start from root
    
    # Find the parent node
    parent_id = np.where((tree.children_left == node_id) | (tree.children_right == node_id))[0][0]
    
    # Determine if left or right split
    feature_name = X_train.columns[tree.feature[parent_id]]  # Get actual feature name
    if tree.children_left[parent_id] == node_id:
        decision = f"{feature_name} <= {tree.threshold[parent_id]:.2f}"
    else:
        decision = f"{feature_name} > {tree.threshold[parent_id]:.2f}"
    
    return get_decision_path(tree, parent_id, path + [decision])

# Get decision paths for highest & lowest probability leaves
high_prob_path = get_decision_path(tree, highest_prob_leaf)
low_prob_path = get_decision_path(tree, lowest_prob_leaf)

print("\nDecision Path for Highest Win Probability:")
print(" → ".join(high_prob_path))

print("\nDecision Path for Lowest Win Probability:")
print(" → ".join(low_prob_path))

#%%

# Load champion information dataset
champ_info = pd.read_csv('champ_info.csv')

# Extract unique role names
unique_roles = champ_info['primary_role'].unique()

# Convert roles into one-hot encoded format dynamically
for role in unique_roles:
    champ_info[f'primary_role_{role}'] = (champ_info['primary_role'] == role).astype(int)

champ_info = champ_info[['name',"primary_role_Assassin", "primary_role_Fighter", "primary_role_Mage",
            "primary_role_Marksman", "primary_role_Support", "primary_role_Tank"]]


def predict_ARAM_win_chance_rf(champ_names):  
    role_counts = {
    "primary_role_Assassin_sum": 0,
    "primary_role_Fighter_sum": 0,
    "primary_role_Mage_sum": 0,
    "primary_role_Marksman_sum": 0,
    "primary_role_Support_sum": 0,
    "primary_role_Tank_sum": 0}

    # Map champion names to roles
    for champ in champ_names:
        roles = champ_info.loc[champ_info["name"] == champ, [
            "primary_role_Assassin", "primary_role_Fighter", "primary_role_Mage",
            "primary_role_Marksman", "primary_role_Support", "primary_role_Tank"]]
        role_counts = {key: role_counts[key] + roles.iloc[0][key.replace("_sum", "")] for key in role_counts}
    
    input_df = pd.DataFrame([role_counts])
    win_prob = rf_base.predict_proba(input_df)[:, 1][0]

    return print(f"""\nTeam Champions: {', '.join(champ_names)}
Predicted Win Probability: {win_prob:.2%}\n""")

# Example champion input
champ_names = ["Galio", "Garen", "Jinx", "Ryze", "Katarina"]

# Get raw team feature values
predict_ARAM_win_chance_rf(champ_names)
