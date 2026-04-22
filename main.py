import pandas as pd

df = pd.read_csv("churn.csv")

# 🔥 1. Fix TotalCharges (very important)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 🔥 2. Handle missing values
df.fillna(0, inplace=True)

# 🔥 3. Convert Churn to binary
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# 🔥 4. Drop useless column
df.drop('customerID', axis=1, inplace=True)

print(df.head())
print("\nCleaned Successfully ✅")

# 🔥 Tenure Group (time-based feature)
def tenure_group(x):
    if x <= 12:
        return "0-1 Year"
    elif x <= 24:
        return "1-2 Years"
    elif x <= 48:
        return "2-4 Years"
    else:
        return "4+ Years"

df['TenureGroup'] = df['tenure'].apply(tenure_group)

# 🔥 Average Monthly Spend
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)

# 🔥 Customer Value
df['CustomerValue'] = df['tenure'] * df['MonthlyCharges']

print("\nFeature Engineering Done 🚀")

print(df[['tenure', 'TenureGroup', 'AvgMonthlySpend', 'CustomerValue']].head())

# 🔥 Convert categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)

print("\nCategorical Encoding Done ✅")
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 🔥 Split features & target
X = df.drop('Churn', axis=1)
y = df['Churn']

# 🔥 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModel Training Done ✅")

# 🔥 Predict class
y_pred = model.predict(X_test)

# 🔥 Predict probability
y_prob = model.predict_proba(X_test)[:, 1]

print("\nPredictions Done ✅")


# Add probability to full dataset
df['Churn_Probability'] = model.predict_proba(X)[:, 1]

# 🔥 Risk Segmentation
def risk_level(p):
    if p > 0.7:
        return "High Risk"
    elif p > 0.3:
        return "Medium Risk"
    else:
        return "Low Risk"

df['Risk_Level'] = df['Churn_Probability'].apply(risk_level)

print("\nRisk Segmentation Done 🚀")
print(df[['Churn', 'Churn_Probability', 'Risk_Level']].head())

df.to_csv("churn_final.csv", index=False)
print("\nFile Saved as churn_final.csv ✅")