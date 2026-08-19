# Customer Segmentation using K-Means Clustering

**Thiranex Internship — Data Analytics Track**
**Intern:** Faizan (THX-AUG1526-027)

## 1. Objective

Segment customers based on their behavior and demographics using K-Means clustering, so each group can be targeted with tailored marketing strategies.

## 2. Dataset

`customer_data.csv` — 200 customers with the following fields:

| Column | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Age | Customer's age |
| Annual_Income_k | Annual income (in thousands) |
| Spending_Score | Score (1-100) based on spending behavior |
| Purchase_Frequency | Average purchases per month |
| Gender | Male / Female |

## 3. Method

### Step 1 — Feature Scaling
Age, Income, Spending Score, and Purchase Frequency are on very different numeric scales (e.g. Income ranges 15-140, Age ranges 18-59). Without scaling, K-Means would let the largest-range feature dominate the distance calculation. `StandardScaler` was used to bring all features to a comparable scale before clustering.

### Step 2 — Elbow Method
K-Means was run for K = 1 to 10, and the Within-Cluster Sum of Squares (WCSS) was recorded for each:

| K | WCSS |
|---|---|
| 1 | 800.0 |
| 2 | 479.4 |
| 3 | 204.8 |
| **4** | **139.8** |
| 5 | 125.1 |
| 6 | 111.1 |

See `elbow_method.png`. The steepest drop happens up to K=3, and the curve visibly flattens after **K=4** — so K=4 was chosen as the optimal number of segments.

### Step 3 — Final Clustering (K=4)
`KMeans(n_clusters=4, random_state=42)` was fit on the scaled data, and each customer was assigned to one of 4 clusters.

## 4. Resulting Segments

| Segment | Count | Avg Age | Avg Income (k) | Avg Spending Score | Avg Purchase Frequency |
|---|---|---|---|---|---|
| **High-Value VIP Customers** | 50 | 36.9 | 97.1 | 85.4 | 9.3 |
| **High-Income Cautious Spenders** | 52 | 49.9 | 91.5 | 22.0 | 2.2 |
| **Average Balanced Customers** | 46 | 36.6 | 56.0 | 48.9 | 5.4 |
| **Young Budget Shoppers** | 52 | 23.1 | 28.2 | 35.0 | 6.3 |

### Segment Interpretation

- **High-Value VIP Customers** — High income, high spending, frequent buyers. The most profitable segment; ideal for loyalty programs and premium product offers.
- **High-Income Cautious Spenders** — High income but low spending and rare purchases. Likely price-sensitive or unengaged despite spending power; a good target for personalized re-engagement campaigns.
- **Average Balanced Customers** — Moderate across all metrics. A stable, general-purpose segment for standard marketing.
- **Young Budget Shoppers** — Lower income, younger, but relatively frequent (smaller, regular) purchases. Good candidates for discount bundles, student offers, and loyalty-building programs early in their customer lifecycle.

## 5. Visualizations

- `elbow_method.png` — WCSS vs K, showing why K=4 was chosen
- `clusters_income_spending.png` — Segments plotted by Income vs Spending Score
- `clusters_age_frequency.png` — Segments plotted by Age vs Purchase Frequency
- `segment_profile_bars.png` — Side-by-side comparison of average characteristics per segment

## 6. Files in this Repository

| File | Purpose |
|---|---|
| `customer_data.csv` | Raw customer dataset |
| `generate_data.py` | Script that generated the sample dataset |
| `kmeans_analysis.py` | Full clustering pipeline: scaling, elbow method, K-Means, profiling, charts |
| `customer_data_segmented.csv` | Final dataset with assigned Cluster and Segment_Name per customer |
| `cluster_profile.csv` | Average characteristics per segment |
| `*.png` | Chart outputs |

## 7. Tools Used

Python — pandas, scikit-learn (KMeans, StandardScaler), matplotlib

## 8. Learning Outcome

This project demonstrates end-to-end customer analytics: preparing and scaling behavioral/demographic data, using the Elbow Method to objectively choose the number of clusters rather than guessing, applying K-Means to group customers algorithmically, and translating the resulting clusters into named, business-relevant segments with actionable interpretations.
