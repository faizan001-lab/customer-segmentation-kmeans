import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 10

df = pd.read_csv("customer_data.csv")

# ---- Features used for clustering ----
features = ["Age", "Annual_Income_k", "Spending_Score", "Purchase_Frequency"]
X = df[features].values

# Scale features - important because Income (0-140) and Spending Score (0-100)
# are on very different number ranges than Age or Frequency. Without scaling,
# Income would dominate the distance calculation just because its numbers are bigger.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- 1. Elbow Method ----
wcss = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

print("=== ELBOW METHOD (WCSS per K) ===")
for k, w in zip(K_range, wcss):
    print(f"K={k:2d}  WCSS={w:8.1f}")

plt.figure(figsize=(7, 5))
plt.plot(list(K_range), wcss, marker='o', linewidth=2, color="#1F4E78")
plt.title("Elbow Method for Optimal K", fontsize=14, fontweight='bold')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.xticks(list(K_range))
plt.grid(alpha=0.3)
plt.annotate("Elbow ~ K=4", xy=(4, wcss[3]), xytext=(6, wcss[3]+150),
             arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=130)
plt.close()
print("\nSaved elbow_method.png")

# ---- 2. Final clustering with K=4 ----
K_FINAL = 4
km_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
df["Cluster"] = km_final.fit_predict(X_scaled)

print(f"\n=== FINAL CLUSTERING (K={K_FINAL}) ===")
print(df["Cluster"].value_counts().sort_index())

# ---- 3. Cluster profiling (average characteristics per cluster) ----
profile = df.groupby("Cluster")[features].mean().round(1)
profile["Count"] = df["Cluster"].value_counts().sort_index()
print("\n=== CLUSTER PROFILE (averages) ===")
print(profile.to_string())

profile.to_csv("cluster_profile.csv")

# Name the segments based on their characteristics (done after inspecting the profile)
def name_segment(row):
    if row["Annual_Income_k"] > 75 and row["Spending_Score"] > 60:
        return "High-Value VIP Customers"
    elif row["Annual_Income_k"] > 75 and row["Spending_Score"] <= 60:
        return "High-Income Cautious Spenders"
    elif row["Annual_Income_k"] <= 45:
        return "Young Budget Shoppers"
    else:
        return "Average Balanced Customers"

profile["Segment_Name"] = profile.apply(name_segment, axis=1)
print("\n=== NAMED SEGMENTS ===")
print(profile[["Count", "Age", "Annual_Income_k", "Spending_Score", "Purchase_Frequency", "Segment_Name"]].to_string())

# Map segment names back onto main dataframe
cluster_to_name = profile["Segment_Name"].to_dict()
df["Segment_Name"] = df["Cluster"].map(cluster_to_name)
df.to_csv("customer_data_segmented.csv", index=False)

# ---- 4. Visualizations ----
colors = ["#1F4E78", "#2E7D32", "#B45309", "#6D28D9", "#DB2777"]

# Scatter: Income vs Spending Score (the classic 2D view)
plt.figure(figsize=(7.5, 6))
for c in sorted(df["Cluster"].unique()):
    sub = df[df["Cluster"] == c]
    plt.scatter(sub["Annual_Income_k"], sub["Spending_Score"],
                s=45, color=colors[c], alpha=0.75, label=f"{cluster_to_name[c]} (n={len(sub)})")
centers_orig = scaler.inverse_transform(km_final.cluster_centers_)
plt.scatter(centers_orig[:, 1], centers_orig[:, 2], s=280, color="black", marker="X", label="Centroids", zorder=5)
plt.title("Customer Segments: Income vs Spending Score", fontsize=14, fontweight='bold')
plt.xlabel("Annual Income (k Rs / month equivalent)")
plt.ylabel("Spending Score (1-100)")
plt.legend(fontsize=8, loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("clusters_income_spending.png", dpi=130)
plt.close()
print("\nSaved clusters_income_spending.png")

# Scatter: Age vs Purchase Frequency
plt.figure(figsize=(7.5, 6))
for c in sorted(df["Cluster"].unique()):
    sub = df[df["Cluster"] == c]
    plt.scatter(sub["Age"], sub["Purchase_Frequency"],
                s=45, color=colors[c], alpha=0.75, label=f"{cluster_to_name[c]}")
plt.title("Customer Segments: Age vs Purchase Frequency", fontsize=14, fontweight='bold')
plt.xlabel("Age")
plt.ylabel("Purchase Frequency (times/month)")
plt.legend(fontsize=8, loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("clusters_age_frequency.png", dpi=130)
plt.close()
print("Saved clusters_age_frequency.png")

# Bar chart: cluster profile comparison
fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
for ax, feat in zip(axes, features):
    vals = profile[feat]
    bars = ax.bar([cluster_to_name[i][:12]+"..." if len(cluster_to_name[i])>12 else cluster_to_name[i] for i in vals.index],
                  vals.values, color=[colors[i] for i in vals.index])
    ax.set_title(feat.replace("_"," "), fontsize=10, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.grid(alpha=0.3, axis='y')
plt.suptitle("Segment Profile Comparison (averages)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("segment_profile_bars.png", dpi=130)
plt.close()
print("Saved segment_profile_bars.png")

print("\nDone. Files ready: customer_data_segmented.csv, cluster_profile.csv, 3 chart PNGs")
