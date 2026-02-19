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

# Load dataset
df = pd.read_csv('final_df.csv')
df = df.drop('match_id', axis=1)
df["win"] = df["win"].astype(int)

# Define role columns
roles = [col for col in df.columns if 'role' in col]

#%% BASE MODEL ALL FEATURES

features = df.drop('win', axis=1).columns.tolist()

df_scaled = df.copy()

# Apply StandardScaler to all numeric columns (including role counts & quadratic terms)
scaler = StandardScaler()
df_scaled[features] = scaler.fit_transform(df[features])

X = df_scaled[features] # Features
y = df_scaled["win"]  # Target

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25, stratify=y)

# Initialize and train logistic regression model
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]  # Probability of class 1 (win)


# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Print results
print('\nBaseline Logistic Regression Results\n')
print('Features Included: ')
print(df_scaled.columns)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Calculate log-likelihood
log_likelihood = np.sum(y_test * np.log(y_prob) + (1 - y_test) * np.log(1 - y_prob))
print('Log-Likelihood:', log_likelihood)

# Log-likelihood of the null model
y_mean = np.mean(y_test)
null_log_likelihood = np.sum(y_test * np.log(y_mean) + (1 - y_test) * np.log(1 - y_mean))
print('Null Log-Likelihood:', null_log_likelihood)

# Calculate the R-squared
mcfadden_r_squared = 1 - (log_likelihood / null_log_likelihood)
print("McFadden's R-squared:", mcfadden_r_squared)

# Check for independence in errors.
# Easily checked using the Durbin-Watson test, included in the statsmodels package
# Ideally want a value close to 2.0

residuals = y_test - y_pred
dw_test = durbin_watson(residuals)
print(f'Durbin-Watson statistic: {dw_test}\n')


#%% RUN VIF

# # Function to iteratively remove features with high VIF
# def reduce_vif(df, threshold=10):
#     removed_features = []  # Store removed features
#     dropped = True
#     df_features = df.drop('win',axis=1)
#     while dropped:
#         vif_df = pd.DataFrame()
#         vif_df["Feature"] = df_features.columns
#         vif_df["VIF"] = [variance_inflation_factor(df_features.values, i) for i in range(df_features.shape[1])]
        
#         max_vif = vif_df["VIF"].max()
#         if max_vif > threshold:
#             feature_to_drop = vif_df.sort_values("VIF", ascending=False).iloc[0]["Feature"]
#             print(f"Removing {feature_to_drop} (VIF={max_vif:.2f})")
#             df_features = df_features.drop(columns=[feature_to_drop])
#             removed_features.append(feature_to_drop)
#         else:
#             dropped = False

#     print("\nFinal Features Retained:", df_features.columns.tolist())
#     print("Features Removed Due to High VIF:", removed_features)
    
#     reduced_df = df.drop(removed_features,axis=1)    
#     return reduced_df, removed_features

# # Example Usage:
# vif_reduced, removed_vars = reduce_vif(df)  # Assuming X is your feature matrix (dataframe)


df_features = df.drop(['win','attack_mean','magic_mean'],axis=1)
vif_df = pd.DataFrame()
vif_df["Feature"] = df_features.columns
vif_df["VIF"] = [variance_inflation_factor(df_features.values, i) for i in range(df_features.shape[1])]
print(None)
print(vif_df.sort_values('VIF', ascending=False))


#%% RUN VIF REDUCED MODEL

features = df_features.columns.tolist()

df_scaled = df_features.copy()

# Apply StandardScaler to all numeric columns (including role counts & quadratic terms)
scaler = StandardScaler()
df_scaled[features] = scaler.fit_transform(df_features[features])

X = df_scaled[features] # Features
y = df["win"]  # Target

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25, stratify=y)

# Initialize and train logistic regression model
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]  # Probability of class 1 (win)


# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Print results
print('\nVIF-Reduced Logistic Regression Results\n')
print('Features Included: ')
print(df_scaled.columns)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Calculate log-likelihood
log_likelihood = np.sum(y_test * np.log(y_prob) + (1 - y_test) * np.log(1 - y_prob))
print('Log-Likelihood:', log_likelihood)

# Log-likelihood of the null model
y_mean = np.mean(y_test)
null_log_likelihood = np.sum(y_test * np.log(y_mean) + (1 - y_test) * np.log(1 - y_mean))
print('Null Log-Likelihood:', null_log_likelihood)

# Calculate the R-squared
mcfadden_r_squared = 1 - (log_likelihood / null_log_likelihood)
print("McFadden's R-squared:", mcfadden_r_squared)

# Check for independence in errors.
# Easily checked using the Durbin-Watson test, included in the statsmodels package
# Ideally want a value close to 2.0

residuals = y_test - y_pred
dw_test = durbin_watson(residuals)
print(f'Durbin-Watson statistic: {dw_test}\n')



# EQUATION
intercept = log_reg.intercept_[0]
coefficients = log_reg.coef_[0]

# Get feature names
feature_names = X.columns if hasattr(X, 'columns') else [f'x{i+1}' for i in range(len(coefficients))]
feature_names = feature_names.sort_values()

# Create the equation string
equation = f"logit(win) = {intercept:.3f}"
for coef, feature in zip(coefficients, feature_names):
    equation += f" + ({coef:.3f} * {feature})"

# Display the equation
print("Logistic Regression Equation:")
print(equation)

odds_changes = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Change in Odds': np.exp(coefficients)
})

# Display the change in odds for each feature
print(odds_changes)




#%% RUN RFE 1

# Set the model
logreg = LogisticRegression(max_iter=1000, solver='lbfgs')

# Stratified cross validation takes a set of "folds" of data, and tests each one against a model trained using the others
# This sets the number of groups or "folds" to use
cv = StratifiedKFold(n_splits=5)

# Apply RFECV with StratifiedKFold as the cross-validation method
rfecv = RFECV(estimator=logreg, step=1, cv=cv, scoring='accuracy')
rfecv.fit(X, y)

# Display selected features and optimal number of features
selected_features = X.columns[rfecv.support_]
removed_features = X.columns[~rfecv.support_]
print("\nSelected features:", selected_features)
print("\nRemoved features:", removed_features)
print("\nOptimal number of features:", rfecv.n_features_, " out of ", len(X.columns))




#%%

df_scaled = df_features[selected_features]

# Apply StandardScaler to all numeric columns (including role counts & quadratic terms)
scaler = StandardScaler()
df_scaled[selected_features] = scaler.fit_transform(df_scaled[selected_features])


X = df_scaled[selected_features] # Features
y = df["win"]  # Target

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25, stratify=y)

# Initialize and train logistic regression model
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]  # Probability of class 1 (win)


# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Print results
print('\nRFE-Reduced Logistic Regression Results\n')
print('Features Included: ')
print(df_scaled.columns)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Calculate log-likelihood
log_likelihood = np.sum(y_test * np.log(y_prob) + (1 - y_test) * np.log(1 - y_prob))
print('Log-Likelihood:', log_likelihood)

# Log-likelihood of the null model
y_mean = np.mean(y_test)
null_log_likelihood = np.sum(y_test * np.log(y_mean) + (1 - y_test) * np.log(1 - y_mean))
print('Null Log-Likelihood:', null_log_likelihood)

# Calculate the R-squared
mcfadden_r_squared = 1 - (log_likelihood / null_log_likelihood)
print("McFadden's R-squared:", mcfadden_r_squared)

# Check for independence in errors.
# Easily checked using the Durbin-Watson test, included in the statsmodels package
# Ideally want a value close to 2.0

residuals = y_test - y_pred
dw_test = durbin_watson(residuals)
print(f'Durbin-Watson statistic: {dw_test}\n')


# EQUATION
intercept = log_reg.intercept_[0]
coefficients = log_reg.coef_[0]

# Get feature names
feature_names = X.columns if hasattr(X, 'columns') else [f'x{i+1}' for i in range(len(coefficients))]
feature_names = feature_names.sort_values()

# Create the equation string
equation = f"logit(win) = {intercept:.3f}"
for coef, feature in zip(coefficients, feature_names):
    equation += f" + ({coef:.3f} * {feature})"

# Display the equation
print("Logistic Regression Equation:")
print(equation)

odds_changes = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Change in Odds': np.exp(coefficients)
})
print('\n')
# Display the change in odds for each feature
print(odds_changes)
print('\n')


#%%

# Assuming already fitted a StandardScaler earlier
# Example: scaler = StandardScaler().fit(X_train)
# Ensure you have access to this fitted scaler
# Replace with actual scaler object used during model training

# Example raw feature values for a new team (before scaling)
raw_team = np.array([[2.5, 0, 2, 3]])  # Use a 2D array for sklearn transform

# Standardize the raw values using the fitted scaler
scaled_team = scaler.transform(raw_team)  # This will return a scaled array

# Coefficients from the trained logistic regression model (same order as features)
coefficients = np.array([-0.0676, -0.1534, 0.0990, 0.0478])  # Ensure ordering matches training

# Calculate log-odds
log_odds = np.dot(scaled_team, coefficients)  # Matrix multiplication

# Convert log-odds to probability using sigmoid function
win_probability = expit(log_odds)[0]  # Extract scalar value

# Print results
print('\nAvg Difficulty: 2.5')
print('Assassins: 0')
print('Mages: 2')
print('Tanks: 3')
print(f"Predicted Win Probability: {win_probability:.4f}\n")

#%%

# Load champion information dataset
champ_info = pd.read_csv('champ_info.csv')

# Extract unique role names
unique_roles = champ_info['primary_role'].unique()

# Convert roles into one-hot encoded format dynamically
for role in unique_roles:
    champ_info[f'primary_role_{role}'] = (champ_info['primary_role'] == role).astype(int)

champ_info = champ_info[['name','difficulty','primary_role_Assassin','primary_role_Mage',
                        'primary_role_Tank']]



def predict_ARAM_win_chance(champ_names):
    team_data = champ_info[champ_info['name'].isin(champ_names)]
    
    # Compute team features
    difficulty_mean = team_data['difficulty'].mean()
    role_sums = team_data.loc[:, ['primary_role_Assassin','primary_role_Mage',
                            'primary_role_Tank']].sum()
    
    # Create feature array and scale
    team_features = np.array([difficulty_mean] + role_sums.tolist()).reshape(1, -1)
    scaled_team_features = scaler.transform(team_features)
    
    # Coefficients from trained logistic regression model
    coefficients = np.array([-0.0676, -0.1534, 0.0990, 0.0478]) 
    
    # Compute log-odds and convert to win probability
    log_odds = np.dot(scaled_team_features, coefficients)
    win_probability = expit(log_odds)[0]
    
    return print(f"""\nTeam Champions: {', '.join(champion_team)}
Predicted Win Probability: {win_probability:.4f}\n""")

# Example champion input
champion_team = ["Galio", "Cho'Gath", "Morgana", "Ziggs", "Fiddlesticks"]

# Get raw team feature values
raw_team_features = predict_ARAM_win_chance(champion_team)
