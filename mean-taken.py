# %% [markdown]
# # Fuzzy Logic Project 1

# %% [markdown]
# ## Özge Bülbül 2220765008

# %%
import os
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score

# %%
# Here I took random csvs but used random seed 42 to get the same randomized dataset each time.

random.seed(42)

no_sepsis_path = "sepsis_dataset-2/dataset/no_sepsis"
sepsis_path = "sepsis_dataset-2/dataset/sepsis"
num_files_to_pick = 225

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
# This function below takes all csvs and merges their mean values so from 450 csvs it returns a 450 row, merged dataset.
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
merged_data.head()

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

# %% [markdown]
# I reported my findings from these plots above in my report.

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
critical_columns = ['sirs', 'qsofa', 'sepsis_icd']  # Since these three features are output features, they are crucial for the model, therefore I dropped their null rows.
merged_data = merged_data.dropna(subset=critical_columns)
print(f"Data shape after dropping rows with null critical columns: {merged_data.shape}")
print(merged_data.isnull().sum())

# %%
correlation_matrix = merged_data.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

plt.figure(figsize=(18, 12))

sns.heatmap(
    correlation_matrix, 
    mask=mask, 
    cmap="coolwarm", 
    annot=True, 
    fmt=".2f", 
    cbar_kws={"shrink": .8},
    vmin=-1, 
    vmax=1
)

plt.title("Triangular Correlation Heatmap", fontsize=16)
plt.show()

# %% [markdown]
# Below, I first printed the features with missing values less than 20% and filled them with median. Then I dropped the features with more than 50% nulls. Finally I decided to fill the rest of the null values with again, median.

# %%
# Columns with <20% missing values
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
correlation_matrix = merged_data.corr()

correlation_with_target = correlation_matrix['sepsis_icd'].sort_values(ascending=False)
print(correlation_with_target)

# %%
threshold = 0.22    # I picked this threshold to continue on my analysis with only the most relevant features

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
#!pip install scikit-fuzzy

# %% [markdown]
# Below, I printed mean of each selected input feature. Using the mean, min and max, I defined my fuzzy sets and visualized them.

# %%
selected_data['heart_rate'].mean()

# %%
heart_rate = np.arange(48, 141, 1)

low_heart_rate = fuzz.trimf(heart_rate, [48, 80, 110])
high_heart_rate = fuzz.trimf(heart_rate, [70, 110, 140])

plt.plot(heart_rate, low_heart_rate, label="Low")
plt.plot(heart_rate, high_heart_rate, label="High")
plt.title("Fuzzy Sets for Heart Rate")
plt.xlabel("Heart Rate")
plt.ylabel("Membership Degree")
plt.legend()
plt.show()

# %%
selected_data['resp'].mean()

# %%
resp = np.arange(10, 46, 1)
low_resp = fuzz.trimf(resp, [10, 16, 22])
high_resp = fuzz.trimf(resp, [18, 31, 45])

plt.plot(resp, low_resp, label="Low")
plt.plot(resp, high_resp, label="High")
plt.title("Fuzzy Sets for resp")
plt.xlabel("resp")
plt.ylabel("Membership Degree")
plt.legend()
plt.show()

# %%
selected_data['bp_systolic'].mean()

# %%
bp_systolic = np.arange(70, 191, 1)

low_bp_systolic = fuzz.trimf(bp_systolic, [70, 100, 130])
high_bp_systolic = fuzz.trimf(bp_systolic, [90, 140, 190])

plt.plot(bp_systolic, low_bp_systolic, label="Low")
plt.plot(bp_systolic, high_bp_systolic, label="High")
plt.title("Fuzzy Sets for Systolic Blood Pressure (High Ambiguity)")
plt.xlabel("Systolic Blood Pressure (mmHg)")
plt.ylabel("Membership Degree")
plt.legend()
plt.show()

# %%
selected_data['po2'].mean()

# %%
po2 = np.arange(30, 471, 1)

low_po2 = fuzz.trimf(po2, [30, 90, 150])
high_po2 = fuzz.trimf(po2, [100, 285, 470])

plt.plot(po2, low_po2, label="Low")
plt.plot(po2, high_po2, label="High")
plt.title("Fuzzy Sets for po2 (High Ambiguity)")
plt.xlabel("po2")
plt.ylabel("Membership Degree")
plt.legend()
plt.show()

# %%
selected_data['bicarbonate'].mean()

# %%
bicarbonate = np.arange(5, 45, 1)

low_bicarbonate = fuzz.trimf(bicarbonate, [5, 17, 28])
high_bicarbonate = fuzz.trimf(bicarbonate, [16, 30, 45])

plt.plot(bicarbonate, low_bicarbonate, label="Low")
plt.plot(bicarbonate, high_bicarbonate, label="High")
plt.title("Fuzzy Sets for Bicarbonate (High Ambiguity)")
plt.xlabel("Bicarbonate (mmol/L)")
plt.ylabel("Membership Degree")
plt.legend()
plt.show()

# %%
heart_rate = ctrl.Antecedent(np.arange(48, 141, 1), 'heart_rate')
resp = ctrl.Antecedent(np.arange(10, 46, 1), 'resp')
bicarbonate = ctrl.Antecedent(np.arange(5, 45, 1), 'bicarbonate')
po2 = ctrl.Antecedent(np.arange(30, 471, 1), 'po2')
bp_systolic = ctrl.Antecedent(np.arange(70, 191, 1), 'bp_systolic')

# %%
heart_rate['low'] = fuzz.trimf(heart_rate.universe, [48, 80, 110])
heart_rate['high'] = fuzz.trimf(heart_rate.universe, [70, 110, 140])

resp['low'] = fuzz.trimf(resp.universe, [10, 16, 22])
resp['high'] = fuzz.trimf(resp.universe, [18, 31, 45])

bicarbonate['low'] = fuzz.trimf(bicarbonate.universe, [5, 17, 28])
bicarbonate['high'] = fuzz.trimf(bicarbonate.universe, [16, 30, 45])

po2['low'] = fuzz.trimf(po2.universe, [30, 90, 150])
po2['high'] = fuzz.trimf(po2.universe, [100, 285, 470])

bp_systolic['low'] = fuzz.trimf(bp_systolic.universe, [70, 100, 130])
bp_systolic['high'] = fuzz.trimf(bp_systolic.universe, [90, 140, 190])


# %%
sepsis = ctrl.Consequent(np.arange(0, 2, 1), 'sepsis')
sepsis['no'] = fuzz.trimf(sepsis.universe, [0, 0, 1])
sepsis['yes'] = fuzz.trimf(sepsis.universe, [0, 1, 1])

# %% [markdown]
# ## Experiment 1

# %% [markdown]
# Here I tried 4 very basic rules, using the positively correlated features in two rules and negaatives on the other two.

# %%
rule1 = ctrl.Rule(resp['high'] | heart_rate['high'], sepsis['yes'])
rule2 = ctrl.Rule(bicarbonate['high'] | po2['high'] | bp_systolic['high'], sepsis['no'])
rule3 = ctrl.Rule(resp['low'] | heart_rate['low'], sepsis['no'])
rule4 = ctrl.Rule(bp_systolic['low'] | po2['low'] | bicarbonate['low'], sepsis['yes'])

# %%
sepsis_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
sepsis_simulation = ctrl.ControlSystemSimulation(sepsis_control)

# %%
selected_data['predicted_sepsis'] = 0
selected_data['predicted_score'] = 0.0

# %%
for idx, row in selected_data.iterrows():
    sepsis_simulation.input['resp'] = row['resp']
    sepsis_simulation.input['heart_rate'] = row['heart_rate']
    sepsis_simulation.input['bp_systolic'] = row['bp_systolic']
    sepsis_simulation.input['po2'] = row['po2']
    sepsis_simulation.input['bicarbonate'] = row['bicarbonate']
    sepsis_simulation.compute()
    selected_data.at[idx, 'predicted_score'] = sepsis_simulation.output.get('sepsis', 0)
    selected_data.at[idx, 'predicted_sepsis'] = 1 if sepsis_simulation.output.get('sepsis', 0) >= 0.5 else 0

predicted_score = sepsis_simulation.output.get('sepsis', 0)

# %%
selected_data

# %%
y_true = selected_data['sepsis_icd']
y_pred = selected_data['predicted_sepsis']
y_scores = selected_data['predicted_score']

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
tnr = tn / (tn + fp)
fnr = fn / (fn + tp)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

fpr_values, tpr_values, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr_values, tpr_values)

print(f"Confusion Matrix:\n{cm}")
print(f"TPR (Sensitivity): {tpr:.2f}")
print(f"FPR (Fall-out): {fpr:.2f}")
print(f"TNR (Specificity): {tnr:.2f}")
print(f"FNR (Miss Rate): {fnr:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity/TPR): {tpr:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr_values, tpr_values, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend()
plt.show()

# %% [markdown]
# ## Experiment 2

# %% [markdown]
# In this experiment I used the positively and negatively correlated features together.

# %%
rule1 = ctrl.Rule(resp['high'] & bp_systolic['low'] & po2['low'], sepsis['yes'])
rule2 = ctrl.Rule(bicarbonate['high'] & heart_rate['low'], sepsis['no'])
rule3 = ctrl.Rule(bp_systolic['high'] & po2['high'] & resp['low'], sepsis['no'])
rule4 = ctrl.Rule(bicarbonate['low'] & heart_rate['high'], sepsis['yes'])

# %%
sepsis_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
sepsis_simulation = ctrl.ControlSystemSimulation(sepsis_control)

# %%
selected_data['predicted_sepsis'] = 0
selected_data['predicted_score'] = 0.0

# %%
for idx, row in selected_data.iterrows():
    sepsis_simulation.input['resp'] = row['resp']
    sepsis_simulation.input['heart_rate'] = row['heart_rate']
    sepsis_simulation.input['bp_systolic'] = row['bp_systolic']
    sepsis_simulation.input['po2'] = row['po2']
    sepsis_simulation.input['bicarbonate'] = row['bicarbonate']
    sepsis_simulation.compute()
    selected_data.at[idx, 'predicted_score'] = sepsis_simulation.output.get('sepsis', 0)
    selected_data.at[idx, 'predicted_sepsis'] = 1 if sepsis_simulation.output.get('sepsis', 0) >= 0.5 else 0

predicted_score = sepsis_simulation.output.get('sepsis', 0)


# %%
selected_data

# %%
y_true = selected_data['sepsis_icd']
y_pred = selected_data['predicted_sepsis']
y_scores = selected_data['predicted_score']

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
tnr = tn / (tn + fp)
fnr = fn / (fn + tp)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

fpr_values, tpr_values, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr_values, tpr_values)

print(f"Confusion Matrix:\n{cm}")
print(f"TPR (Sensitivity): {tpr:.2f}")
print(f"FPR (Fall-out): {fpr:.2f}")
print(f"TNR (Specificity): {tnr:.2f}")
print(f"FNR (Miss Rate): {fnr:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity/TPR): {tpr:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr_values, tpr_values, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend()
plt.show()

# %% [markdown]
# ## Experiment 3
# In this experiment I tried out some contradicting values. For instance, both heart rate and resp are positively correlated, I defined a rule that when heart rate is high but resp is low, sepsis is no because resp has a higher positive correlation than heart rate.
# 

# %%
rule1 = ctrl.Rule(resp['high'] & bicarbonate['low'], sepsis['yes'])
rule2 = ctrl.Rule(bp_systolic['high'] & heart_rate['low'], sepsis['no'])
rule3 = ctrl.Rule(bp_systolic['low'] & bicarbonate['high'], sepsis['no'])  # because bicarbonate has higher negative correlation
rule4 = ctrl.Rule(po2['low'] & bicarbonate['high'], sepsis['no'])   # because bicarbonate has higher negative correlation
rule5 = ctrl.Rule(bp_systolic['high'] & bicarbonate['low'], sepsis['yes'])
rule6 = ctrl.Rule(resp['low'] & heart_rate['high'], sepsis['no'])  # because resp has higher correlation
rule7 = ctrl.Rule(resp['high'] & heart_rate['low'], sepsis['yes'])

# %%
sepsis_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
sepsis_simulation = ctrl.ControlSystemSimulation(sepsis_control)

# %%
selected_data['predicted_sepsis'] = 0
selected_data['predicted_score'] = 0.0

# %%
for idx, row in selected_data.iterrows():
    sepsis_simulation.input['resp'] = row['resp']
    sepsis_simulation.input['heart_rate'] = row['heart_rate']
    sepsis_simulation.input['bp_systolic'] = row['bp_systolic']
    sepsis_simulation.input['po2'] = row['po2']
    sepsis_simulation.input['bicarbonate'] = row['bicarbonate']
    sepsis_simulation.compute()
    selected_data.at[idx, 'predicted_score'] = sepsis_simulation.output.get('sepsis', 0)
    selected_data.at[idx, 'predicted_sepsis'] = 1 if sepsis_simulation.output.get('sepsis', 0) >= 0.5 else 0

predicted_score = sepsis_simulation.output.get('sepsis', 0)


# %%
selected_data

# %%
y_true = selected_data['sepsis_icd']
y_pred = selected_data['predicted_sepsis']
y_scores = selected_data['predicted_score']

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
tnr = tn / (tn + fp)
fnr = fn / (fn + tp)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

fpr_values, tpr_values, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr_values, tpr_values)

print(f"Confusion Matrix:\n{cm}")
print(f"TPR (Sensitivity): {tpr:.2f}")
print(f"FPR (Fall-out): {fpr:.2f}")
print(f"TNR (Specificity): {tnr:.2f}")
print(f"FNR (Miss Rate): {fnr:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity/TPR): {tpr:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr_values, tpr_values, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend()
plt.show()

# %% [markdown]
# ## Experiment 4
# In this last experiment I expanded experiment 3's rule system to see whether the performance will improve or not.
# 

# %%
rule1 = ctrl.Rule(resp['high'] & bp_systolic['low'], sepsis['yes'])
rule2 = ctrl.Rule(bicarbonate['high'] & heart_rate['low'], sepsis['no'])
rule3 = ctrl.Rule(bp_systolic['low'] & resp['high'], sepsis['yes'])
rule4 = ctrl.Rule(bicarbonate['low'] & heart_rate['high'], sepsis['yes'])
rule5 = ctrl.Rule(bp_systolic['low'] & bicarbonate['high'], sepsis['no'])  # because bicarbonate has higher negative correlation
rule6 = ctrl.Rule(bp_systolic['high'] & bicarbonate['low'], sepsis['yes'])
rule7 = ctrl.Rule(po2['high'] & bicarbonate['low'], sepsis['yes'])
rule8 = ctrl.Rule(po2['low'] & bicarbonate['high'], sepsis['no'])
rule9 = ctrl.Rule(resp['low'] & heart_rate['high'], sepsis['no'])  # because resp has higher correlation
rule10 = ctrl.Rule(resp['high'] & heart_rate['low'], sepsis['yes'])

# %%
sepsis_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10])
sepsis_simulation = ctrl.ControlSystemSimulation(sepsis_control)

# %%
selected_data['predicted_sepsis'] = 0
selected_data['predicted_score'] = 0.0

# %%
for idx, row in selected_data.iterrows():
    sepsis_simulation.input['resp'] = row['resp']
    sepsis_simulation.input['heart_rate'] = row['heart_rate']
    sepsis_simulation.input['bp_systolic'] = row['bp_systolic']
    sepsis_simulation.input['po2'] = row['po2']
    sepsis_simulation.input['bicarbonate'] = row['bicarbonate']
    sepsis_simulation.compute()
    selected_data.at[idx, 'predicted_score'] = sepsis_simulation.output.get('sepsis', 0)
    selected_data.at[idx, 'predicted_sepsis'] = 1 if sepsis_simulation.output.get('sepsis', 0) >= 0.5 else 0

predicted_score = sepsis_simulation.output.get('sepsis', 0)


# %%
selected_data

# %%
y_true = selected_data['sepsis_icd']
y_pred = selected_data['predicted_sepsis']
y_scores = selected_data['predicted_score']

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
tnr = tn / (tn + fp)
fnr = fn / (fn + tp)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

fpr_values, tpr_values, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr_values, tpr_values)

print(f"Confusion Matrix:\n{cm}")
print(f"TPR (Sensitivity): {tpr:.2f}")
print(f"FPR (Fall-out): {fpr:.2f}")
print(f"TNR (Specificity): {tnr:.2f}")
print(f"FNR (Miss Rate): {fnr:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity/TPR): {tpr:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr_values, tpr_values, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend()
plt.show()

# %% [markdown]
# Experiment 1: <br>
# Confusion Matrix:
# <br>[[150  75]
# <br> [ 83 142]]
# <br>TPR (Sensitivity): 0.63
# <br>FPR (Fall-out): 0.33
# <br>TNR (Specificity): 0.67
# <br>FNR (Miss Rate): 0.37
# <br>Accuracy: 0.65
# <br>Precision: 0.65
# <br>Recall (Sensitivity/TPR): 0.63
# <br>F1 Score: 0.64
# <br>ROC AUC: 0.73
# <br>
# <br>Experiment 2: <br>
# Confusion Matrix:
# <br>[[165  60]
# <br> [ 91 134]]
# <br>TPR (Sensitivity): 0.60
# <br>FPR (Fall-out): 0.27
# <br>TNR (Specificity): 0.73
# <br>FNR (Miss Rate): 0.40
# <br>Accuracy: 0.66
# <br>Precision: 0.69
# <br>Recall (Sensitivity/TPR): 0.60
# <br>F1 Score: 0.64
# <br>ROC AUC: 0.73
# <br>
# <br>
# Experiment 3: <br>
# Confusion Matrix:
# <br>[[184  41]
# <br> [125 100]]
# <br>TPR (Sensitivity): 0.44
# <br>FPR (Fall-out): 0.18
# <br>TNR (Specificity): 0.82
# <br>FNR (Miss Rate): 0.56
# <br>Accuracy: 0.63
# <br>Precision: 0.71
# <br>Recall (Sensitivity/TPR): 0.44
# <br>F1 Score: 0.55
# <br>ROC AUC: 0.67
# 
# Experiment 4: <br>
# Confusion Matrix:
# <br>[[163  62]
# <br> [ 89 136]]
# <br>TPR (Sensitivity): 0.60
# <br>FPR (Fall-out): 0.28
# <br>TNR (Specificity): 0.72
# <br>FNR (Miss Rate): 0.40
# <br>Accuracy: 0.66
# <br>Precision: 0.69
# <br>Recall (Sensitivity/TPR): 0.60
# <br>F1 Score: 0.64
# <br>ROC AUC: 0.70
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# %% [markdown]
# After my comparisons, I picked experiment 2 and explained my reasoning in my report.


