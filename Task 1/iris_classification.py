# Iris flower classification
# loads the iris dataset, trains a few different models and compares accuracy

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

sns.set_style("whitegrid")

# ---- load data ----
df = pd.read_csv("Iris.csv")
df = df.drop(columns=["Id"])
df["Species"] = df["Species"].str.replace("Iris-", "", regex=False)

feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[feature_cols]
y = df["Species"]

print(df.head())
print()
print(df["Species"].value_counts())
print()
print(X.describe())

# quick look at how the species separate out by feature
pairplot = sns.pairplot(df, hue="Species", palette="Set2")
pairplot.savefig("pairplot.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ---- train/test split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# scale features - mostly matters for knn/svm since they rely on distances
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---- try a few models and see which does best ----
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(kernel="linear", random_state=42),
}

scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    scores[name] = acc
    print(f"{name}: {acc:.4f}")

best_name = max(scores, key=scores.get)
best_model = models[best_name]
print(f"\nbest model: {best_name} ({scores[best_name]:.4f})")

pred = best_model.predict(X_test)
print(classification_report(y_test, pred))

# confusion matrix for the best model
labels = sorted(y.unique())
cm = confusion_matrix(y_test, pred, labels=labels)
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot(cmap="Blues", colorbar=False)
plt.title(f"Confusion matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close("all")

# bar chart comparing all models
plt.figure(figsize=(7, 4.5))
bars = plt.bar(scores.keys(), [s * 100 for s in scores.values()], color=sns.color_palette("Set2", len(scores)))
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
for bar, acc in zip(bars, scores.values()):
    plt.text(bar.get_x() + bar.get_width() / 2, acc * 100 + 1, f"{acc*100:.1f}%", ha="center")
plt.title("Model comparison")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close("all")

# quick sanity check - predict on a new flower
new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=feature_cols)
new_flower_scaled = scaler.transform(new_flower)
print("\nprediction for", new_flower.values.tolist()[0], "->", best_model.predict(new_flower_scaled)[0])
