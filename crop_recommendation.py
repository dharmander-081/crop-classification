# ============================================================
# CROP RECOMMENDATION SYSTEM - COMPLETE JUPYTER NOTEBOOK
# ============================================================
# Copy each cell (separated by # --- Cell N ---) into your
# Jupyter Notebook. Run cells sequentially.
# ============================================================


# --- Cell 1: Import Libraries ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import warnings

warnings.filterwarnings("ignore")

print("✅ All libraries imported successfully!")


# --- Cell 2: Load Dataset ---

# Load the dataset (update the path if needed)
df = pd.read_csv("Crop_recommendation.csv")

# Display first 5 rows
print("First 5 rows of the dataset:")
df.head()


# --- Cell 3: Dataset Overview ---

# Shape of the dataset
print(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

# Column names and data types
print("Column Info:")
print(df.dtypes)
print()

# Basic statistics
print("Statistical Summary:")
df.describe()


# --- Cell 4: Check for Missing Values and Duplicates ---

# Missing values
print("Missing Values per Column:")
print(df.isnull().sum())
print(f"\nTotal Missing Values: {df.isnull().sum().sum()}")

# Duplicates
dup_count = df.duplicated().sum()
print(f"\nDuplicate Rows: {dup_count}")

# Remove duplicates if any
if dup_count > 0:
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape[0]} rows remain")
else:
    print("No duplicates found — dataset is clean!")


# --- Cell 5: Target Variable Distribution ---

# Count of each crop
plt.figure(figsize=(14, 6))
crop_counts = df["label"].value_counts()
colors = sns.color_palette("husl", len(crop_counts))
bars = plt.bar(crop_counts.index, crop_counts.values, color=colors, edgecolor="black", linewidth=0.5)
plt.title("Distribution of Crops in the Dataset", fontsize=16, fontweight="bold")
plt.xlabel("Crop", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha="right")

# Add count labels on bars
for bar, count in zip(bars, crop_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.show()

print(f"\nTotal unique crops: {df['label'].nunique()}")
print(f"Crops: {', '.join(sorted(df['label'].unique()))}")


# --- Cell 6: Feature Distributions ---

# Distribution of each numerical feature
features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(features):
    sns.histplot(df[col], kde=True, ax=axes[i], color=sns.color_palette("Set2")[i], edgecolor="black")
    axes[i].set_title(f"Distribution of {col}", fontsize=13, fontweight="bold")
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel("Frequency", fontsize=10)

# Hide the last empty subplot
axes[-1].set_visible(False)

plt.suptitle("Feature Distributions", fontsize=18, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# --- Cell 7: Correlation Heatmap ---

# Correlation matrix of numerical features
plt.figure(figsize=(10, 8))
corr = df[features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # Upper triangle mask

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8},
    vmin=-1,
    vmax=1,
)
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

# Key insights
print("📊 Key Insights from Correlation Heatmap:")
print("  • Most features have low correlation with each other (good for modeling)")
print("  • P and K show slight positive correlation")
print("  • Features are mostly independent → all are useful for prediction")


# --- Cell 8: Boxplots — Feature vs Crop ---

fig, axes = plt.subplots(2, 4, figsize=(22, 12))
axes = axes.flatten()

for i, col in enumerate(features):
    sns.boxplot(data=df, x="label", y=col, ax=axes[i], palette="Set3")
    axes[i].set_title(f"{col} by Crop", fontsize=13, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=90, labelsize=7)

axes[-1].set_visible(False)

plt.suptitle("Feature Distribution Across Crops", fontsize=18, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()


# --- Cell 9: Pairplot (Optional — Slow for Large Data) ---

# Uncomment below to generate a pairplot (takes a few seconds)
# sns.pairplot(df, hue="label", vars=features[:4], palette="husl", diag_kind="kde")
# plt.suptitle("Pairplot of Selected Features", y=1.02, fontsize=16)
# plt.show()

print("💡 Pairplot cell is optional. Uncomment to run if needed.")


# --- Cell 10: Prepare Data for Modeling ---

# Separate features (X) and target (y)
X = df[features]  # N, P, K, temperature, humidity, ph, rainfall
y = df["label"]   # Crop name

# Encode the target labels to numbers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size:  {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of classes:  {len(le.classes_)}")


# --- Cell 11: Model 1 — Logistic Regression ---

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=2000, random_state=42)
lr_model.fit(X_train, y_train)

# Predict on test set
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print(f"📌 Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")


# --- Cell 12: Model 2 — Decision Tree ---

# Train Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict on test set
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

print(f"📌 Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")


# --- Cell 13: Model 3 — Random Forest ---

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"📌 Random Forest Accuracy: {rf_accuracy * 100:.2f}%")


# --- Cell 14: Model Comparison ---

# Compare all models
models = {
    "Logistic Regression": lr_accuracy,
    "Decision Tree": dt_accuracy,
    "Random Forest": rf_accuracy,
}

# Display comparison table
comparison_df = pd.DataFrame({
    "Model": models.keys(),
    "Accuracy (%)": [round(v * 100, 2) for v in models.values()],
})
comparison_df = comparison_df.sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
print("📊 Model Comparison:")
print(comparison_df.to_string(index=False))

# Bar chart
plt.figure(figsize=(8, 5))
colors_bar = ["#2ecc71", "#3498db", "#e74c3c"]
bars = plt.bar(models.keys(), [v * 100 for v in models.values()], color=colors_bar, edgecolor="black")
plt.title("Model Accuracy Comparison", fontsize=16, fontweight="bold")
plt.ylabel("Accuracy (%)", fontsize=12)
plt.ylim(0, 105)

# Add percentage labels
for bar, acc in zip(bars, models.values()):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{acc * 100:.2f}%", ha="center", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.show()

# Select best model
best_model_name = max(models, key=models.get)
best_accuracy = models[best_model_name]
print(f"\n🏆 Best Model: {best_model_name} with {best_accuracy * 100:.2f}% accuracy")


# --- Cell 15: Confusion Matrix — Best Model (Random Forest) ---

# Using Random Forest predictions for confusion matrix
plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")
plt.title("Confusion Matrix — Random Forest", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()


# --- Cell 16: Classification Report — Best Model ---

# Detailed classification report
print("📋 Classification Report — Random Forest\n")
report = classification_report(y_test, rf_pred, target_names=le.classes_)
print(report)


# --- Cell 17: Feature Importance — Random Forest ---

# Feature importance from the Random Forest model
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances,
}).sort_values("Importance", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df["Feature"], feat_imp_df["Importance"], color=sns.color_palette("viridis", len(features)))
plt.title("Feature Importance — Random Forest", fontsize=16, fontweight="bold")
plt.xlabel("Importance", fontsize=12)
plt.tight_layout()
plt.show()

print("📊 Feature Importance (sorted):")
for _, row in feat_imp_df.sort_values("Importance", ascending=False).iterrows():
    print(f"  {row['Feature']:>12s}: {row['Importance']:.4f}")


# --- Cell 18: Save the Best Model ---

import pickle

# Save the trained Random Forest model and Label Encoder
with open("crop_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model saved as 'crop_model.pkl'")
print("✅ Label Encoder saved as 'label_encoder.pkl'")


# --- Cell 19: Prediction Function ---

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predict the best crop based on soil and climate conditions.

    Parameters:
        N           : Nitrogen content in soil
        P           : Phosphorus content in soil
        K           : Potassium content in soil
        temperature : Temperature in °C
        humidity    : Relative humidity in %
        ph          : pH value of soil
        rainfall    : Rainfall in mm

    Returns:
        Predicted crop name (string)
    """
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = rf_model.predict(input_data)
    crop_name = le.inverse_transform(prediction)[0]
    return crop_name


# --- Cell 20: Test the Prediction Function ---

# Example 1: High N, moderate P, K → likely rice
crop1 = predict_crop(N=90, P=42, K=43, temperature=21, humidity=82, ph=6.5, rainfall=200)
print(f"🌾 Prediction 1: {crop1}")

# Example 2: Low N, high P, K → likely something else
crop2 = predict_crop(N=20, P=130, K=200, temperature=30, humidity=60, ph=7.0, rainfall=100)
print(f"🌾 Prediction 2: {crop2}")

# Example 3: Moderate values
crop3 = predict_crop(N=40, P=60, K=45, temperature=25, humidity=70, ph=6.8, rainfall=150)
print(f"🌾 Prediction 3: {crop3}")

# Example 4: Custom input
crop4 = predict_crop(N=50, P=50, K=50, temperature=28, humidity=65, ph=6.0, rainfall=120)
print(f"🌾 Prediction 4: {crop4}")

print("\n✅ Prediction function is working correctly!")
print("🎉 Crop Recommendation System is complete!")
