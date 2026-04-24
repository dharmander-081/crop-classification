"""Generate the .ipynb Jupyter Notebook file."""
import json

cells = []

def md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    })

def code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source]
    })

# ── Title ──
md(["# Crop Recommendation System\n",
    "\n",
    "Predicting the best crop to grow based on soil and climate conditions.\n",
    "\n",
    "**Dataset:** Crop_recommendation.csv\n",
    "\n",
    "**Features:** N, P, K, Temperature, Humidity, pH, Rainfall\n",
    "\n",
    "**Target:** Crop label (22 types)"])

# ── Cell 1 ──
code(["# importing required libraries\n",
      "\n",
      "import numpy as np\n",
      "import pandas as pd\n",
      "import matplotlib.pyplot as plt\n",
      "import seaborn as sns\n",
      "from sklearn.model_selection import train_test_split\n",
      "from sklearn.preprocessing import LabelEncoder\n",
      "from sklearn.linear_model import LogisticRegression\n",
      "from sklearn.tree import DecisionTreeClassifier\n",
      "from sklearn.ensemble import RandomForestClassifier\n",
      "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay\n",
      "import warnings\n",
      "warnings.filterwarnings('ignore')"])

# ── Cell 2 ──
code(["# loading the dataset\n",
      "\n",
      "df = pd.read_csv('Crop_recommendation.csv')\n",
      "df.head()"])

# ── Cell 3 ──
code(["# checking shape and data types\n",
      "\n",
      "print('Shape:', df.shape)\n",
      "print()\n",
      "print(df.dtypes)\n",
      "print()\n",
      "df.describe()"])

# ── Cell 4 ──
code(["# checking for missing values and duplicates\n",
      "\n",
      "print('Missing values:')\n",
      "print(df.isnull().sum())\n",
      "print()\n",
      "print('Duplicate rows:', df.duplicated().sum())\n",
      "\n",
      "# dropping duplicates if present\n",
      "df = df.drop_duplicates()\n",
      "print('Shape after cleanup:', df.shape)"])

# ── Cell 5 ──
code(["# count of each crop in the dataset\n",
      "\n",
      "plt.figure(figsize=(12, 5))\n",
      "crop_counts = df['label'].value_counts()\n",
      "plt.bar(crop_counts.index, crop_counts.values, color=sns.color_palette('husl', len(crop_counts)), edgecolor='black', linewidth=0.5)\n",
      "plt.title('Crop Distribution')\n",
      "plt.xlabel('Crop')\n",
      "plt.ylabel('Count')\n",
      "plt.xticks(rotation=45, ha='right')\n",
      "plt.tight_layout()\n",
      "plt.show()\n",
      "\n",
      "print('Total crops:', df['label'].nunique())"])

# ── Cell 6 ──
code(["# distribution of each feature\n",
      "\n",
      "features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']\n",
      "\n",
      "fig, axes = plt.subplots(2, 4, figsize=(18, 8))\n",
      "axes = axes.flatten()\n",
      "\n",
      "for i, col in enumerate(features):\n",
      "    sns.histplot(df[col], kde=True, ax=axes[i], color=sns.color_palette('Set2')[i])\n",
      "    axes[i].set_title(col)\n",
      "\n",
      "axes[-1].set_visible(False)\n",
      "plt.tight_layout()\n",
      "plt.show()"])

# ── Cell 7 ──
code(["# correlation heatmap\n",
      "\n",
      "plt.figure(figsize=(8, 6))\n",
      "corr = df[features].corr()\n",
      "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1)\n",
      "plt.title('Correlation Heatmap')\n",
      "plt.tight_layout()\n",
      "plt.show()"])

# ── Cell 8 ──
code(["# boxplots to see feature variation across crops\n",
      "\n",
      "fig, axes = plt.subplots(2, 4, figsize=(20, 10))\n",
      "axes = axes.flatten()\n",
      "\n",
      "for i, col in enumerate(features):\n",
      "    sns.boxplot(data=df, x='label', y=col, ax=axes[i], palette='Set3')\n",
      "    axes[i].set_title(col)\n",
      "    axes[i].set_xlabel('')\n",
      "    axes[i].tick_params(axis='x', rotation=90, labelsize=7)\n",
      "\n",
      "axes[-1].set_visible(False)\n",
      "plt.tight_layout()\n",
      "plt.show()"])

# ── Cell 9 ──
md(["## Model Building\n",
    "\n",
    "Splitting data into train and test sets, then training multiple models to compare."])

# ── Cell 10 ──
code(["# splitting data into features and target\n",
      "\n",
      "X = df[features]\n",
      "y = df['label']\n",
      "\n",
      "# encoding target labels\n",
      "le = LabelEncoder()\n",
      "y_encoded = le.fit_transform(y)\n",
      "\n",
      "# train-test split (80-20)\n",
      "X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)\n",
      "\n",
      "print('Train size:', X_train.shape[0])\n",
      "print('Test size:', X_test.shape[0])"])

# ── Cell 11 ──
code(["# logistic regression\n",
      "\n",
      "lr_model = LogisticRegression(max_iter=2000, random_state=42)\n",
      "lr_model.fit(X_train, y_train)\n",
      "lr_pred = lr_model.predict(X_test)\n",
      "lr_acc = accuracy_score(y_test, lr_pred)\n",
      "print('Logistic Regression Accuracy:', round(lr_acc * 100, 2), '%')"])

# ── Cell 12 ──
code(["# decision tree\n",
      "\n",
      "dt_model = DecisionTreeClassifier(random_state=42)\n",
      "dt_model.fit(X_train, y_train)\n",
      "dt_pred = dt_model.predict(X_test)\n",
      "dt_acc = accuracy_score(y_test, dt_pred)\n",
      "print('Decision Tree Accuracy:', round(dt_acc * 100, 2), '%')"])

# ── Cell 13 ──
code(["# random forest\n",
      "\n",
      "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
      "rf_model.fit(X_train, y_train)\n",
      "rf_pred = rf_model.predict(X_test)\n",
      "rf_acc = accuracy_score(y_test, rf_pred)\n",
      "print('Random Forest Accuracy:', round(rf_acc * 100, 2), '%')"])

# ── Cell 14 ──
code(["# comparing all models\n",
      "\n",
      "models = {'Logistic Regression': lr_acc, 'Decision Tree': dt_acc, 'Random Forest': rf_acc}\n",
      "\n",
      "print('Model Comparison:')\n",
      "for name, acc in models.items():\n",
      "    print(f'  {name}: {acc*100:.2f}%')\n",
      "\n",
      "# bar chart\n",
      "plt.figure(figsize=(8, 4))\n",
      "plt.bar(models.keys(), [v*100 for v in models.values()], color=['#2ecc71','#3498db','#e74c3c'])\n",
      "plt.title('Model Accuracy Comparison')\n",
      "plt.ylabel('Accuracy (%)')\n",
      "plt.ylim(0, 105)\n",
      "for i, (name, acc) in enumerate(models.items()):\n",
      "    plt.text(i, acc*100 + 1, f'{acc*100:.2f}%', ha='center')\n",
      "plt.tight_layout()\n",
      "plt.show()\n",
      "\n",
      "best = max(models, key=models.get)\n",
      "print(f'\\nBest model: {best}')"])

# ── Cell 15 ──
md(["## Model Evaluation\n",
    "\n",
    "Using Random Forest as the best performing model for detailed evaluation."])

# ── Cell 16 ──
code(["# confusion matrix for random forest\n",
      "\n",
      "plt.figure(figsize=(12, 10))\n",
      "cm = confusion_matrix(y_test, rf_pred)\n",
      "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)\n",
      "disp.plot(cmap='Blues', xticks_rotation=45, values_format='d')\n",
      "plt.title('Confusion Matrix - Random Forest')\n",
      "plt.tight_layout()\n",
      "plt.show()"])

# ── Cell 17 ──
code(["# classification report\n",
      "\n",
      "print('Classification Report - Random Forest')\n",
      "print()\n",
      "print(classification_report(y_test, rf_pred, target_names=le.classes_))"])

# ── Cell 18 ──
code(["# feature importance\n",
      "\n",
      "importances = rf_model.feature_importances_\n",
      "feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})\n",
      "feat_df = feat_df.sort_values('Importance', ascending=True)\n",
      "\n",
      "plt.figure(figsize=(8, 5))\n",
      "plt.barh(feat_df['Feature'], feat_df['Importance'], color=sns.color_palette('viridis', len(features)))\n",
      "plt.title('Feature Importance - Random Forest')\n",
      "plt.xlabel('Importance')\n",
      "plt.tight_layout()\n",
      "plt.show()"])

# ── Cell 19 ──
code(["# saving the model\n",
      "\n",
      "import pickle\n",
      "\n",
      "with open('crop_model.pkl', 'wb') as f:\n",
      "    pickle.dump(rf_model, f)\n",
      "\n",
      "with open('label_encoder.pkl', 'wb') as f:\n",
      "    pickle.dump(le, f)\n",
      "\n",
      "print('Model saved successfully')"])

# ── Cell 20 ──
md(["## Prediction Function\n",
    "\n",
    "Function to predict the best crop based on user input values."])

# ── Cell 21 ──
code(["# prediction function\n",
      "\n",
      "def predict_crop(N, P, K, temperature, humidity, ph, rainfall):\n",
      "    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])\n",
      "    prediction = rf_model.predict(input_data)\n",
      "    crop = le.inverse_transform(prediction)[0]\n",
      "    return crop"])

# ── Cell 22 ──
code(["# testing with sample values\n",
      "\n",
      "print('Test 1:', predict_crop(90, 42, 43, 21, 82, 6.5, 200))\n",
      "print('Test 2:', predict_crop(20, 130, 200, 30, 60, 7.0, 100))\n",
      "print('Test 3:', predict_crop(40, 60, 45, 25, 70, 6.8, 150))\n",
      "print('Test 4:', predict_crop(50, 50, 50, 28, 65, 6.0, 120))"])

# ── Build notebook ──
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

with open("Crop_Recommendation_System.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook regenerated successfully")
print(f"Total cells: {len(cells)}")
