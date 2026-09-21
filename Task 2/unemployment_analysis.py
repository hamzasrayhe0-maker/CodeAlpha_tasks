# Unemployment in India - analysis
# two files: one covers mid 2019 - mid 2020 (rural/urban split), the other
# covers jan-oct 2020 with region + lat/long. using both to look at trends
# and the covid impact around april 2020

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---- load + clean ----
df1 = pd.read_csv("Unemployment_in_India.csv")
df2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# both files have leading spaces in the column names, fix that first
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# df2 has "Region" twice (state name + zone like South/North), rename the second one
df2 = df2.rename(columns={"Region.1": "Zone"})

# drop the empty rows in df1 (28 rows with nothing in them)
df1 = df1.dropna()

# strip whitespace out of the string columns, dates have leading spaces too
for col in ["Region", "Date", "Frequency", "Area"]:
    df1[col] = df1[col].str.strip()
for col in ["Region", "Date", "Frequency", "Zone"]:
    df2[col] = df2[col].str.strip()

df1["Date"] = pd.to_datetime(df1["Date"], format="%d-%m-%Y")
df2["Date"] = pd.to_datetime(df2["Date"], format="%d-%m-%Y")

# shorter column names, easier to work with
rename_map = {
    "Estimated Unemployment Rate (%)": "unemployment_rate",
    "Estimated Employed": "employed",
    "Estimated Labour Participation Rate (%)": "labour_participation",
}
df1 = df1.rename(columns=rename_map)
df2 = df2.rename(columns=rename_map)

print("df1 shape:", df1.shape, "| date range:", df1["Date"].min().date(), "-", df1["Date"].max().date())
print("df2 shape:", df2.shape, "| date range:", df2["Date"].min().date(), "-", df2["Date"].max().date())
print()
print(df1.head())
print()
print(df1["unemployment_rate"].describe())

# ---- national trend over time (df1, since it covers the longer period) ----
monthly_avg = df1.groupby("Date")["unemployment_rate"].mean().reset_index()

plt.figure(figsize=(10, 5))
plt.plot(monthly_avg["Date"], monthly_avg["unemployment_rate"], marker="o", color="steelblue")
plt.axvline(pd.Timestamp("2020-03-25"), color="red", linestyle="--", label="lockdown started (25 Mar 2020)")
plt.title("Average unemployment rate over time - India")
plt.xlabel("Date")
plt.ylabel("Unemployment rate (%)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("national_trend.png", dpi=150)
plt.close()

# ---- rural vs urban ----
area_avg = df1.groupby(["Date", "Area"])["unemployment_rate"].mean().reset_index()

plt.figure(figsize=(10, 5))
for area in area_avg["Area"].unique():
    subset = area_avg[area_avg["Area"] == area]
    plt.plot(subset["Date"], subset["unemployment_rate"], marker="o", label=area)
plt.axvline(pd.Timestamp("2020-03-25"), color="red", linestyle="--", alpha=0.6)
plt.title("Unemployment rate: rural vs urban")
plt.xlabel("Date")
plt.ylabel("Unemployment rate (%)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("rural_vs_urban.png", dpi=150)
plt.close()

# ---- before / during covid comparison ----
# treating march 2020 onwards as "covid period" here
df1["period"] = df1["Date"].apply(lambda d: "covid (Mar 2020+)" if d >= pd.Timestamp("2020-03-01") else "pre-covid")
period_avg = df1.groupby("period")["unemployment_rate"].mean()
print("\navg unemployment before vs during covid:")
print(period_avg)

plt.figure(figsize=(6, 4.5))
period_avg.plot(kind="bar", color=["seagreen", "indianred"])
plt.ylabel("Average unemployment rate (%)")
plt.title("Pre-covid vs covid period")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("covid_comparison.png", dpi=150)
plt.close()

# ---- top states by average unemployment ----
state_avg = df1.groupby("Region")["unemployment_rate"].mean().sort_values(ascending=False)
print("\ntop 10 states by avg unemployment rate:")
print(state_avg.head(10))

plt.figure(figsize=(8, 6))
state_avg.head(10).plot(kind="barh", color="darkorange")
plt.xlabel("Average unemployment rate (%)")
plt.title("Top 10 states - highest avg unemployment")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("top_states.png", dpi=150)
plt.close()

# ---- monthly seasonality (does unemployment rise/fall in certain months?) ----
df1["month"] = df1["Date"].dt.month_name()
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
month_avg = df1.groupby("month")["unemployment_rate"].mean().reindex(month_order).dropna()

plt.figure(figsize=(9, 4.5))
month_avg.plot(kind="bar", color="mediumpurple")
plt.ylabel("Average unemployment rate (%)")
plt.title("Average unemployment rate by month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_seasonality.png", dpi=150)
plt.close()

# ---- zone-level view from the second dataset ----
zone_avg = df2.groupby("Zone")["unemployment_rate"].mean().sort_values(ascending=False)
print("\navg unemployment by zone (2020 data):")
print(zone_avg)

plt.figure(figsize=(7, 4.5))
zone_avg.plot(kind="bar", color="teal")
plt.ylabel("Average unemployment rate (%)")
plt.title("Unemployment rate by zone (2020)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("zone_comparison.png", dpi=150)
plt.close()

print("\ndone - charts saved: national_trend.png, rural_vs_urban.png, covid_comparison.png,")
print("top_states.png, monthly_seasonality.png, zone_comparison.png")
