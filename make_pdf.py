"""
Generate a complete HTML report with all code, outputs, and graphs.
Then open in browser to save as PDF.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
import base64
import io
import os

warnings.filterwarnings('ignore')

# ─── Helpers ───
plots = []
outputs = []
code_blocks = []

def save_plot(title=""):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plots.append((title, img_b64))
    plt.close('all')
    return len(plots) - 1

def add_section(title, code, output_text="", plot_idx=None):
    code_blocks.append({
        'title': title,
        'code': code,
        'output': output_text,
        'plot_idx': plot_idx
    })

# ═══════════════════════════════════════════════
# RUN ALL THE CODE
# ═══════════════════════════════════════════════

# --- Load data ---
df = pd.read_csv('Crop_recommendation.csv')
head_text = df.head(10).to_string()
add_section("Load Dataset", 
    "df = pd.read_csv('Crop_recommendation.csv')\ndf.head(10)",
    head_text)

# --- Dataset overview ---
desc_text = f"Shape: {df.shape}\n\n{df.dtypes.to_string()}\n\n{df.describe().to_string()}"
add_section("Dataset Overview",
    "print('Shape:', df.shape)\nprint(df.dtypes)\ndf.describe()",
    desc_text)

# --- Missing values ---
missing_text = f"Missing values:\n{df.isnull().sum().to_string()}\n\nTotal Missing: {df.isnull().sum().sum()}\nDuplicate rows: {df.duplicated().sum()}"
df = df.drop_duplicates()
missing_text += f"\nShape after cleanup: {df.shape}"
add_section("Missing Values & Duplicates",
    "print(df.isnull().sum())\nprint('Duplicates:', df.duplicated().sum())\ndf = df.drop_duplicates()",
    missing_text)

# --- Crop distribution ---
plt.figure(figsize=(14, 6))
crop_counts = df['label'].value_counts()
colors = sns.color_palette('husl', len(crop_counts))
bars = plt.bar(crop_counts.index, crop_counts.values, color=colors, edgecolor='black', linewidth=0.5)
plt.title('Distribution of Crops in the Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Crop')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
for bar, count in zip(bars, crop_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, str(count), ha='center', fontsize=8)
plt.tight_layout()
p1 = save_plot("Crop Distribution")
add_section("Crop Distribution",
    "crop_counts = df['label'].value_counts()\nplt.bar(crop_counts.index, crop_counts.values)",
    f"Total unique crops: {df['label'].nunique()}", p1)

# --- Feature distributions ---
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for i, col in enumerate(features):
    sns.histplot(df[col], kde=True, ax=axes[i], color=sns.color_palette('Set2')[i])
    axes[i].set_title(f'{col}', fontsize=12)
axes[-1].set_visible(False)
plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
p2 = save_plot("Feature Distributions")
add_section("Feature Distributions",
    "for col in features:\n    sns.histplot(df[col], kde=True)",
    "", p2)

# --- Correlation heatmap ---
plt.figure(figsize=(10, 8))
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
p3 = save_plot("Correlation Heatmap")
add_section("Correlation Heatmap",
    "corr = df[features].corr()\nsns.heatmap(corr, annot=True, cmap='coolwarm')",
    "Most features have low correlation - good for modeling", p3)

# --- Boxplots ---
fig, axes = plt.subplots(2, 4, figsize=(22, 12))
axes = axes.flatten()
for i, col in enumerate(features):
    sns.boxplot(data=df, x='label', y=col, ax=axes[i], palette='Set3')
    axes[i].set_title(col, fontsize=12)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=90, labelsize=6)
axes[-1].set_visible(False)
plt.suptitle('Feature Distribution Across Crops', fontsize=16, fontweight='bold')
plt.tight_layout()
p4 = save_plot("Boxplots")
add_section("Boxplots - Feature vs Crop",
    "for col in features:\n    sns.boxplot(data=df, x='label', y=col)",
    "", p4)

# --- Train-test split ---
X = df[features]
y = df['label']
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
split_text = f"Train size: {X_train.shape[0]}\nTest size: {X_test.shape[0]}\nFeatures: {X_train.shape[1]}\nClasses: {len(le.classes_)}"
add_section("Prepare Data for Modeling",
    "X = df[features]\ny = df['label']\nle = LabelEncoder()\ny_encoded = le.fit_transform(y)\nX_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)",
    split_text)

# --- Logistic Regression ---
lr_model = LogisticRegression(max_iter=2000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
add_section("Model 1 - Logistic Regression",
    "lr_model = LogisticRegression(max_iter=2000, random_state=42)\nlr_model.fit(X_train, y_train)\nlr_pred = lr_model.predict(X_test)",
    f"Logistic Regression Accuracy: {lr_acc*100:.2f}%")

# --- Decision Tree ---
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
add_section("Model 2 - Decision Tree",
    "dt_model = DecisionTreeClassifier(random_state=42)\ndt_model.fit(X_train, y_train)\ndt_pred = dt_model.predict(X_test)",
    f"Decision Tree Accuracy: {dt_acc*100:.2f}%")

# --- Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
add_section("Model 3 - Random Forest",
    "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\nrf_model.fit(X_train, y_train)\nrf_pred = rf_model.predict(X_test)",
    f"Random Forest Accuracy: {rf_acc*100:.2f}%")

# --- Model comparison ---
plt.figure(figsize=(8, 5))
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accs = [lr_acc*100, dt_acc*100, rf_acc*100]
bars = plt.bar(model_names, accs, color=['#2ecc71','#3498db','#e74c3c'], edgecolor='black')
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 105)
for bar, acc in zip(bars, accs):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{acc:.2f}%', ha='center', fontweight='bold')
plt.tight_layout()
p5 = save_plot("Model Comparison")
comp_text = f"Logistic Regression: {lr_acc*100:.2f}%\nDecision Tree: {dt_acc*100:.2f}%\nRandom Forest: {rf_acc*100:.2f}%\n\nBest Model: Random Forest"
add_section("Model Comparison",
    "models = {'Logistic Regression': lr_acc, 'Decision Tree': dt_acc, 'Random Forest': rf_acc}",
    comp_text, p5)

# --- Confusion matrix ---
plt.figure(figsize=(14, 12))
cm = confusion_matrix(y_test, rf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues', xticks_rotation=45, values_format='d')
plt.title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
p6 = save_plot("Confusion Matrix")
add_section("Confusion Matrix - Random Forest",
    "cm = confusion_matrix(y_test, rf_pred)\nConfusionMatrixDisplay(cm, display_labels=le.classes_).plot()",
    "", p6)

# --- Classification report ---
report = classification_report(y_test, rf_pred, target_names=le.classes_)
add_section("Classification Report - Random Forest",
    "print(classification_report(y_test, rf_pred, target_names=le.classes_))",
    report)

# --- Feature importance ---
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
plt.figure(figsize=(10, 6))
plt.barh(feat_df['Feature'], feat_df['Importance'], color=sns.color_palette('viridis', len(features)))
plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
p7 = save_plot("Feature Importance")
feat_text = "\n".join([f"  {row['Feature']:>12s}: {row['Importance']:.4f}" for _, row in feat_df.sort_values('Importance', ascending=False).iterrows()])
add_section("Feature Importance",
    "importances = rf_model.feature_importances_",
    feat_text, p7)

# --- Prediction function ---
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = rf_model.predict(input_data)
    return le.inverse_transform(prediction)[0]

pred_text = f"Test 1: {predict_crop(90,42,43,21,82,6.5,200)}\nTest 2: {predict_crop(20,130,200,30,60,7.0,100)}\nTest 3: {predict_crop(40,60,45,25,70,6.8,150)}\nTest 4: {predict_crop(50,50,50,28,65,6.0,120)}"
add_section("Prediction Function & Testing",
    "def predict_crop(N, P, K, temperature, humidity, ph, rainfall):\n    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])\n    prediction = rf_model.predict(input_data)\n    return le.inverse_transform(prediction)[0]\n\nprint(predict_crop(90, 42, 43, 21, 82, 6.5, 200))\nprint(predict_crop(20, 130, 200, 30, 60, 7.0, 100))",
    pred_text)


# ═══════════════════════════════════════════════
# BUILD HTML
# ═══════════════════════════════════════════════

html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Crop Recommendation System</title>
<style>
body {
    font-family: 'Segoe UI', Tahoma, sans-serif;
    max-width: 1000px;
    margin: 0 auto;
    padding: 30px;
    background: #fff;
    color: #333;
    line-height: 1.6;
}
h1 { text-align: center; color: #2c3e50; border-bottom: 3px solid #27ae60; padding-bottom: 15px; }
h2 { color: #2c3e50; margin-top: 40px; border-left: 4px solid #27ae60; padding-left: 12px; }
.section { margin-bottom: 30px; page-break-inside: avoid; }
pre.code-block {
    background: #f4f4f4;
    border: 1px solid #ddd;
    border-radius: 6px;
    padding: 15px;
    overflow-x: auto;
    font-size: 13px;
    font-family: 'Consolas', monospace;
}
pre.output-block {
    background: #1e1e1e;
    color: #d4d4d4;
    border-radius: 6px;
    padding: 15px;
    overflow-x: auto;
    font-size: 12px;
    font-family: 'Consolas', monospace;
}
.plot-img { text-align: center; margin: 20px 0; }
.plot-img img { max-width: 100%; border: 1px solid #eee; border-radius: 8px; }
.label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
@media print {
    body { padding: 10px; }
    .section { page-break-inside: avoid; }
    pre.code-block { font-size: 11px; }
}
</style>
</head>
<body>

<h1>Crop Recommendation System</h1>
<p style="text-align:center; color:#555;">
    Predicting the best crop based on soil nutrients and climate conditions<br>
    <strong>Dataset:</strong> Crop_recommendation.csv | <strong>Models:</strong> Logistic Regression, Decision Tree, Random Forest
</p>
<hr>
"""

for i, block in enumerate(code_blocks):
    html += f'\n<div class="section">\n'
    html += f'<h2>{i+1}. {block["title"]}</h2>\n'
    html += f'<div class="label">Code</div>\n'
    html += f'<pre class="code-block">{block["code"]}</pre>\n'
    
    if block["output"]:
        html += f'<div class="label">Output</div>\n'
        html += f'<pre class="output-block">{block["output"]}</pre>\n'
    
    if block["plot_idx"] is not None:
        title, img_b64 = plots[block["plot_idx"]]
        html += f'<div class="plot-img"><img src="data:image/png;base64,{img_b64}" alt="{title}"></div>\n'
    
    html += '</div>\n'

html += """
<hr>
<p style="text-align:center; color:#888; font-size:12px;">
    Crop Recommendation System | Built with Python, Scikit-learn, Matplotlib, Seaborn
</p>
</body>
</html>"""

output_file = "Crop_Recommendation_System_Report.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Report generated: {output_file}")
print(f"Sections: {len(code_blocks)}")
print(f"Plots: {len(plots)}")
print("Opening in browser...")

os.startfile(output_file)
