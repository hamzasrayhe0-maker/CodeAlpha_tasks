# car price prediction
# predicting selling price of used cars from year, mileage, fuel type etc.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")

# ---- load data ----
df = pd.read_csv("car_data.csv")
print(df.head())
print()
print(df.info())
print()
print(df.describe())

# ---- feature engineering ----
# year on its own isn't that useful, age of the car matters more
current_year = 2020  # dataset seems to be from around 2020
df["Car_Age"] = current_year - df["Year"]
df = df.drop(columns=["Year"])

# car name has too many unique values to one-hot encode directly (98 different cars)
# just drop it, the other features carry most of the signal anyway
df = df.drop(columns=["Car_Name"])

# quick look at correlations before encoding
plt.figure(figsize=(6, 5))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between numeric features")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.close()

# selling price vs present price - should be a pretty strong relationship
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x="Present_Price", y="Selling_Price", hue="Fuel_Type")
plt.title("Present price vs selling price")
plt.tight_layout()
plt.savefig("price_vs_price.png", dpi=150)
plt.close()

# price by fuel type / transmission
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price", ax=axes[0])
axes[0].set_title("Selling price by fuel type")
sns.boxplot(data=df, x="Transmission", y="Selling_Price", ax=axes[1])
axes[1].set_title("Selling price by transmission")
plt.tight_layout()
plt.savefig("price_by_category.png", dpi=150)
plt.close()

# ---- encode categorical columns ----
df = pd.get_dummies(df, columns=["Fuel_Type", "Selling_type", "Transmission"], drop_first=True)

X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- try a few regression models ----
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    results[name] = {"mae": mae, "rmse": rmse, "r2": r2}
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

best_name = max(results, key=lambda n: results[n]["r2"])
best_model = models[best_name]
print(f"\nbest model: {best_name} (R2={results[best_name]['r2']:.3f})")

# ---- actual vs predicted for the best model ----
best_pred = best_model.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_pred, alpha=0.6, color="steelblue")
lims = [0, max(y_test.max(), best_pred.max())]
plt.plot(lims, lims, "r--", label="perfect prediction")
plt.xlabel("Actual selling price")
plt.ylabel("Predicted selling price")
plt.title(f"Actual vs predicted - {best_name}")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

# ---- model comparison chart ----
plt.figure(figsize=(7, 4.5))
r2_scores = [results[n]["r2"] for n in results]
bars = plt.bar(results.keys(), r2_scores, color=sns.color_palette("Set2", len(results)))
plt.ylabel("R2 score")
plt.title("Model comparison (R2 on test set)")
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, score + 0.01, f"{score:.3f}", ha="center")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

# ---- feature importance (only makes sense for tree-based models) ----
if hasattr(best_model, "feature_importances_"):
    importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nfeature importance:")
    print(importance)

    plt.figure(figsize=(7, 5))
    importance.plot(kind="barh", color="darkorange")
    plt.gca().invert_yaxis()
    plt.title(f"Feature importance - {best_name}")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

# ---- quick prediction on a made-up car ----
new_car = pd.DataFrame([{
    "Present_Price": 9.85,
    "Driven_kms": 6900,
    "Owner": 0,
    "Car_Age": 3,
    "Fuel_Type_Diesel": 0,
    "Fuel_Type_Petrol": 1,
    "Selling_type_Individual": 0,
    "Transmission_Manual": 1,
}])
# make sure columns line up with training data (in case dummy columns differ)
new_car = new_car.reindex(columns=X.columns, fill_value=0)
pred_price = best_model.predict(new_car)[0]
print(f"\npredicted selling price for sample car: {pred_price:.2f} lakhs")

print("\ndone - charts saved: correlation_heatmap.png, price_vs_price.png, price_by_category.png,")
print("actual_vs_predicted.png, model_comparison.png, feature_importance.png")
