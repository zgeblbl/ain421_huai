# %% [markdown]
# # Fuzzy Logic Project 1 - Final

# %% [markdown]
# ## Özge Bülbül 2220765008

# %%
import os
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score, classification_report,roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzz

# %%
# Here I took random csvs but used random seed 42 to get the same randomized dataset each time.

random.seed(42)

no_sepsis_path = "sepsis_dataset-2/dataset/no_sepsis"
sepsis_path = "sepsis_dataset-2/dataset/sepsis"
num_files_to_pick = 750

def select_random_files(folder_path, num_files):
    files = os.listdir(folder_path)
    selected_files = random.sample(files, num_files)
    return [os.path.join(folder_path, file) for file in selected_files]

# %%
selected_no_sepsis = select_random_files(no_sepsis_path, num_files_to_pick)
selected_sepsis = select_random_files(sepsis_path, num_files_to_pick)

selected_files = selected_no_sepsis + selected_sepsis

print(f"Selected {len(selected_no_sepsis)} 'no_sepsis' files and {len(selected_sepsis)} 'sepsis' files.")

# %%
selected_no_sepsis

# %%
# This function below takes all csvs and merges their mean values so from 1500 csvs it returns a 1500 row, merged dataset.
def load_and_merge_csv_with_mean(file_list):
    data_frames = []
    for file in file_list:
        df = pd.read_csv(file)
        mean_df = df.mean().to_frame().T
        data_frames.append(mean_df)
    return pd.concat(data_frames, ignore_index=True)

# %%
merged_data = load_and_merge_csv_with_mean(selected_files)
print(f"Merged data shape: {merged_data.shape}")

# %%
merged_data.head()

# %%
merged_data['sepsis_icd'] = merged_data['sepsis_icd'].round().astype(int)   # I am turning sepsis_icd back into 0-1 values because it turned to float like 0.0 after taking mean

print(merged_data['sepsis_icd'].unique())

# %%
# I shuffled the dataset
merged_data = merged_data.sample(frac=1).reset_index(drop=True)

# %%
merged_data

# %% [markdown]
# Below, I did some visualization to detect correlations etc.

# %%
correlation_matrix = merged_data.corr()

plt.figure(figsize=(18, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix of Inputs and Output")
plt.show()

# %%
input_columns = ["heart_rate", "bp_systolic", "bp_diastolic", "map", "resp", "temp", "spo2", "fio2", "wbc", "bun", "bilirubin", "creatinine", "lactate", "platelets", "ph", "pco2", "po2", "bicarbonate", "hemoglobin", "hematocrit", "potassium", "chloride", "gcs", "age"]
output_column = 'sepsis_icd'

input_output_corr = merged_data[input_columns + [output_column]].corr()

plt.figure(figsize=(18, 12))
sns.heatmap(input_output_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Between Inputs and Output (Sepsis)")
plt.show()

# %%
plt.figure(figsize=(14, 10))
for i, column in enumerate(input_columns, 1):
    plt.subplot(6, 4, i)
    sns.histplot(merged_data[column], kde=True, color='hotpink', bins=30)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(14, 10))
for i, column in enumerate(input_columns, 1):
    plt.subplot(6,4, i)
    sns.boxplot(x=merged_data[column], color='mediumseagreen')
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
plt.tight_layout()
plt.show()

# %%
stats = merged_data[input_columns].agg(['min', 'max', 'std']).transpose()
stats.columns = ['Min', 'Max', 'Standard Deviation']

print("Statistical Summary of Parameters:")
print(stats)

# Count the number of positive (sepsis=1) and negative (sepsis=0) instances
positive_instances = merged_data[merged_data['sepsis_icd'] == 1].shape[0]
negative_instances = merged_data[merged_data['sepsis_icd'] == 0].shape[0]

print("\nNumber of Positive and Negative Instances:")
print(f"Positive Instances (Sepsis=1): {positive_instances}")
print(f"Negative Instances (Sepsis=0): {negative_instances}")

# %% [markdown]
# Below, I did missing value handling.

# %%
missing_values = merged_data.isnull().sum()
print(missing_values[missing_values > 0])

# %%
low_missing_cols = merged_data.columns[(merged_data.isnull().mean() < 0.2) & (merged_data.isnull().mean() > 0)]
low_missing_cols

# %%
for col in low_missing_cols:
    merged_data[col].fillna(merged_data[col].median(), inplace=True)

# Drop high missingness columns (>50%)
high_missing_cols = merged_data.columns[merged_data.isnull().mean() > 0.4]
print(high_missing_cols)
merged_data.drop(columns=high_missing_cols, inplace=True)

# Verify remaining nulls and columns
print("Remaining null values:")
print(merged_data.isnull().sum())
print("\nRemaining columns:")
print(merged_data.columns)

# %%
moderate_missing_cols = ['lactate', 'ph', 'pco2', 'po2']
for col in moderate_missing_cols:
    merged_data[col].fillna(merged_data[col].median(), inplace=True)

# Verify final dataset
print("Remaining null values:")
print(merged_data.isnull().sum())
print("\nFinal columns:")
print(merged_data.columns)

# %%
merged_data.to_csv('merged_data.csv', index=False)

# %%
correlation_matrix = merged_data.corr()

correlation_with_target = correlation_matrix['sepsis_icd'].sort_values(ascending=False)
print(correlation_with_target)

# %%
threshold = 0.22   # I picked this threshold to continue on my analysis with only the most relevant features

selected_features = correlation_with_target[abs(correlation_with_target) > threshold].index.tolist()

print("\nSelected features based on correlation with target:")
print(selected_features)

# %%
# New dataset with only the selected features
selected_data = merged_data[selected_features]

# Shape of the selected dataset
print("\nShape of the dataset with selected features:")
print(selected_data.shape)

# %%
correlation_matrix_filtered = selected_data.corr()
correlation_with_target = correlation_matrix_filtered['sepsis_icd'].sort_values(ascending=False)
print(correlation_with_target)

# %%
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_filtered, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# %%
plt.figure(figsize=(14, 10))
for i, column in enumerate(selected_data, 1):
    plt.subplot(3, 2, i)
    sns.histplot(selected_data[column], kde=True, color='hotpink', bins=30)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(14, 10))
for i, column in enumerate(selected_data, 1):
    plt.subplot(3,2, i)
    sns.boxplot(x=selected_data[column], color='mediumseagreen')
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
plt.tight_layout()
plt.show()

# %%
print("Available columns in filtered dataset:")
print(correlation_matrix_filtered.columns)

# %%
input_columns_filtered = [col for col in input_columns if col in correlation_matrix_filtered.columns]
print("Filtered input columns:")
print(input_columns_filtered)

# %% [markdown]
# Below, while defining x,y input and target features, i scaled all values.

# %%
scaler = MinMaxScaler()
X = scaler.fit_transform(selected_data[input_columns_filtered])
y = selected_data[output_column].values
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# %%
stats = selected_data.agg(['min', 'max', 'std']).transpose()
stats.columns = ['Min', 'Max', 'Standard Deviation']

print("Statistical Summary of Parameters:")
print(stats)

# Count the number of positive (sepsis=1) and negative (sepsis=0) instances
positive_instances = selected_data[selected_data['sepsis_icd'] == 1].shape[0]
negative_instances = selected_data[selected_data['sepsis_icd'] == 0].shape[0]

print("\nNumber of Positive and Negative Instances:")
print(f"Positive Instances (Sepsis=1): {positive_instances}")
print(f"Negative Instances (Sepsis=0): {negative_instances}")

# %%
selected_data

# %% [markdown]
# ## ANFIS Components
# - <b>Fuzzy Inputs and Membership Functions</b><br>
# Define membership functions for the input variables.
# 
# - <b>Fuzzy Rules</b><br>
# Create fuzzy if-then rules based my features' correlation ("If heart rate is high, then sepsis is likely").
# 
# - <b>Fuzzy Inference</b><br>
# Use the inference system.
# 
# - <b>Defuzzification</b><br>
# Compute crisp output from the fuzzy inference system.
# 
# - <b>Learning</b><br>
# Use gradient descent or another optimization method to tune parameters of the membership functions.
# 
# 

# %%
selected_data = selected_data.drop(columns=['sirs', 'qsofa'])

# %%
selected_data.to_csv('selected_data.csv', index=False)

# %%
import torch
import torch.nn as nn
import torch.optim as optim

# %%
selected_data

# %%
# Gaussian Membership Function
def gaussian_mf(x, mean, sigma):
    return torch.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

# %%
# Compute Rule Activations for a Batch
def batch_rule_activation(inputs, rules, fuzzy_sets):
    batch_size = inputs.size(0)
    num_rules = len(rules)
    rule_activations = torch.ones(batch_size, num_rules, dtype=torch.float32)

    for rule_idx, rule in enumerate(rules):
        for feature, fuzzy_set_name in rule['conditions'].items():
            feature_idx = feature_index_map[feature]
            fuzzy_set_details = next(
                item for item in fuzzy_sets[feature] if item['name'] == fuzzy_set_name
            )
            mean = torch.tensor(fuzzy_set_details['mean'], dtype=torch.float32)
            sigma = torch.tensor(fuzzy_set_details['sigma'], dtype=torch.float32)
            input_values = inputs[:, feature_idx]
            mf_output = gaussian_mf(input_values, mean, sigma)
            rule_activations[:, rule_idx] *= mf_output

    return rule_activations

# %%
# ANFIS Model
class ANFIS(nn.Module):
    def __init__(self, num_rules):
        super(ANFIS, self).__init__()
        self.output_weights = nn.Parameter(torch.randn(num_rules))

    def forward(self, rule_activations):
        weighted_outputs = rule_activations * self.output_weights
        total_activation = torch.sum(rule_activations, dim=1)
        total_activation[total_activation == 0] = 1e-6  # division by zero
        return torch.sum(weighted_outputs, dim=1) / total_activation

# %%
# Training Function
def train_anfis(X, y, rules, fuzzy_sets, model, learning_rate, epochs, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    dataset_size = len(X)
    for epoch in range(epochs):
        total_loss = 0
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        correct_predictions = 0

        indices = torch.randperm(dataset_size)
        X = X[indices]
        y = y[indices]
        all_targets = []
        all_predictions = []

        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_inputs = X[start_idx:end_idx]
            batch_targets = y[start_idx:end_idx]

            rule_activations = batch_rule_activation(batch_inputs, rules, fuzzy_sets)

            predictions = model(rule_activations)
            loss = criterion(predictions, batch_targets)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_targets.extend(batch_targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().detach().numpy())

            batch_preds = torch.round(predictions)  # Threshold = 0.5
            true_positive += ((batch_preds == 1) & (batch_targets == 1)).sum().item()
            false_positive += ((batch_preds == 1) & (batch_targets == 0)).sum().item()
            true_negative += ((batch_preds == 0) & (batch_targets == 0)).sum().item()
            false_negative += ((batch_preds == 0) & (batch_targets == 1)).sum().item()


            correct_predictions += (torch.round(predictions) == batch_targets).sum().item()

        accuracy = (true_positive + true_negative) / dataset_size * 100
        TPR = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        FPR = false_positive / (false_positive + true_negative) if (false_positive + true_negative) > 0 else 0
        TNR = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
        FNR = false_negative / (false_negative + true_positive) if (false_negative + true_positive) > 0 else 0
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        F1 = 2 * (precision * TPR) / (precision + TPR) if (precision + TPR) > 0 else 0
        ROC_AUC = roc_auc_score(all_targets, all_predictions)

        print(f"Epoch {epoch + 1}")
        print(f"Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"TPR (Recall): {TPR:.2f}, FPR: {FPR:.2f}, TNR (Specificity): {TNR:.2f}, FNR: {FNR:.2f}")
        print(f"Precision: {precision:.2f}, F1-Score: {F1:.2f}, ROC AUC: {ROC_AUC:.2f}")

# %%
# generate fuzzy sets using percentiles
def generate_fuzzy_sets(data):
    fuzzy_sets = {}
    for feature in data.columns:
        feature_values = data[feature].values
        low_mean = np.percentile(feature_values, 25)
        normal_mean = np.percentile(feature_values, 50)
        high_mean = np.percentile(feature_values, 75)
        sigma = (np.max(feature_values) - np.min(feature_values)) / 6
        fuzzy_sets[feature] = [
            {'name': 'Low', 'mean': low_mean, 'sigma': sigma},
            {'name': 'Normal', 'mean': normal_mean, 'sigma': sigma},
            {'name': 'High', 'mean': high_mean, 'sigma': sigma},
        ]
    return fuzzy_sets


# %% [markdown]
# correlations to create rules: <br>
# resp           0.282415<br>
# heart_rate     0.248462<br>
# ph            -0.220355<br>
# bp_systolic   -0.265758<br>
# bicarbonate   -0.288430

# %%
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

feature_index_map = {
    'heart_rate': 0,    # Feature 1 -> index 0
    'bp_systolic': 1,   # Feature 2 -> index 1
    'resp': 2,          # Feature 3 -> index 2
    'ph': 3,            # Feature 4 -> index 3
    'bicarbonate': 4    # Feature 5 -> index 4
}
fuzzy_sets = generate_fuzzy_sets(selected_data)

rules = [
    {'conditions': {'heart_rate': 'High', 'resp': 'High'}, 'output': [1]},
    {'conditions': {'heart_rate': 'Low', 'bp_systolic': 'High', 'resp': 'Low'}, 'output': [0]},
    {'conditions': {'heart_rate': 'Normal', 'bp_systolic': 'Normal', 'resp': 'Normal'}, 'output': [1]},
    {'conditions': {'heart_rate': 'Low', 'resp': 'Low'}, 'output': [0]},
    {'conditions': {'bp_systolic': 'Low', 'bicarbonate': 'Low'}, 'output': [1]},
    {'conditions': {'bp_systolic': 'High', 'bicarbonate': 'Normal'}, 'output': [0]},
    {'conditions': {'ph': 'High', 'bicarbonate': 'Normal'}, 'output': [0]},
    {'conditions': {'heart_rate': 'Normal', 'bp_systolic': 'High', 'ph': 'Low'}, 'output': [0]},
]


# %%
print(X.shape)

# %%
num_rules = len(rules)
anfis_model = ANFIS(num_rules)

train_anfis(X, y, rules, fuzzy_sets, anfis_model, learning_rate=0.01, epochs=100, batch_size=32)

predictions = []
with torch.no_grad():
    for i in range(0, len(X), 32):
        batch_inputs = X[i:i+32]
        rule_activations = batch_rule_activation(batch_inputs, rules, fuzzy_sets)
        batch_predictions = anfis_model(rule_activations)
        predictions.extend(batch_predictions.tolist())

print(predictions)

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# below i made sure the types are matching
y_test = np.array(y_test)
log_reg_preds = np.array(log_reg_preds)
rf_preds = np.array(rf_preds)

log_reg_accuracy = accuracy_score(y_test, log_reg_preds)
rf_accuracy = accuracy_score(y_test, rf_preds)

print("Logistic Regression Performance:")
print(f"Accuracy: {log_reg_accuracy}")
print(classification_report(y_test, log_reg_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, log_reg_preds))

print("\nRandom Forest Performance:")
print(f"Accuracy: {rf_accuracy}")
print(classification_report(y_test, rf_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

# %% [markdown]
# In conclusion the accuracies can be ranked best to worst in this order: random forest, logistic regression, anfis. I will further explain my thoughts on this in my report.


