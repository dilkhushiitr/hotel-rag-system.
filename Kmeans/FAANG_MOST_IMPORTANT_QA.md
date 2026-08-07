# FAANG-Style Most Important Interview Q&A

Project: **User Segmentation Using K-Means and DBSCAN**

This guide is written as if a FAANG-style interviewer is evaluating you for a Data Scientist role. The questions are grouped by project step. Each answer is detailed enough for interview preparation but still practical to speak aloud.

---

## Step 1: Business Understanding and Problem Framing

### 1. What business problem does this project solve?
**Answer:**  
The project solves the problem of broad and generic ad targeting. Earlier, all users were treated similarly, so high-value users, casual users, inactive users, and ad-responsive users were receiving similar campaigns. This reduces CTR, conversion, retention, and revenue per user. By segmenting users based on behavior, the ad system can personalize targeting and improve campaign efficiency.

### 2. Why is segmentation useful for an ad monetization platform like GreedyGame?
**Answer:**  
Ad monetization depends on showing the right ad to the right user at the right time. Segmentation helps identify groups such as high-value users, low-engagement users, and ad clickers. For example, ad clickers can receive performance campaigns, high-value users can receive premium offers, and churn-risk users can receive reactivation campaigns.

### 3. Why did you choose unsupervised learning instead of supervised learning?
**Answer:**  
In this project, we do not have predefined labels like “high-value”, “casual”, or “churn-risk” for every user. Since the goal is to discover natural behavior groups from data, unsupervised learning is appropriate. Once segments are deployed and campaign outcomes are collected, we can later build supervised models for CTR prediction, churn prediction, or conversion propensity.

### 4. What is the final business output of this project?
**Answer:**  
The final output is a segment label for each user, such as:
- High-value users
- Casual users
- Low engagement / churn risk
- Ad clickers

These labels can be passed to the ad targeting system to personalize campaigns.

### 5. How would you measure whether this project is successful?
**Answer:**  
I would measure both model quality and business impact. Model quality can be checked using silhouette score, cluster separation, and segment stability. Business impact should be measured using A/B testing with metrics like CTR uplift, revenue per user, retention, conversion rate, and ad efficiency.

### 6. What would be your North Star metric?
**Answer:**  
For this project, I would choose revenue per user or campaign ROI as the North Star metric because the business goal is monetization. CTR is important, but CTR alone can be misleading if clicks do not convert into meaningful revenue.

### 7. How would you explain this project to a non-technical business stakeholder?
**Answer:**  
I would say: “We grouped users based on how they use the app, how often they engage, how they interact with ads, and how much value they generate. These groups help us target users more personally instead of showing the same ads to everyone.”

### 8. What are the risks of incorrect segmentation?
**Answer:**  
Incorrect segmentation can send the wrong campaigns to users, reduce CTR, hurt user experience, waste ad inventory, and possibly reduce revenue. For example, if churn-risk users are incorrectly treated as high-value active users, the platform may miss reactivation opportunities.

### 9. What makes a segment useful from a business perspective?
**Answer:**  
A useful segment should be interpretable, stable, actionable, and measurable. If a segment cannot guide a campaign decision or does not show different business behavior, it is not very useful even if it looks mathematically separated.

### 10. How would you turn this project into a business experiment?
**Answer:**  
I would run an A/B test. The control group gets generic targeting, while the treatment group gets segment-based targeting. Then I would compare CTR, conversion rate, revenue per user, retention, and user experience metrics between the two groups.

---

## Step 2: Data Understanding and Schema

### 1. Why do you need seven different data tables?
**Answer:**  
User behavior is distributed across multiple systems. `users.csv` gives profile and signup information, `sessions.csv` gives engagement, `ad_events.csv` gives ad response, `wallet.csv` gives monetization behavior, `transactions.csv` gives financial behavior, and location/device tables help with profiling. Combining them creates a complete user-level view.

### 2. What is the primary entity in this project?
**Answer:**  
The primary entity is the user. Every final feature row represents one `user_id`. Event-level data like sessions and transactions is aggregated to user level before clustering.

### 3. Why is `user_id` important?
**Answer:**  
`user_id` is the join key across all tables. Without a consistent user identifier, we cannot combine sessions, ad interactions, wallet activity, and transactions into one feature vector.

### 4. What information does `sessions.csv` provide?
**Answer:**  
It provides engagement signals such as number of sessions, session duration, active days, last activity date, and content depth. These features help identify active users, casual users, and churn-risk users.

### 5. What information does `ad_events.csv` provide?
**Answer:**  
It provides impressions and clicks. From these, we calculate CTR, which is a key signal for ad responsiveness.

### 6. Why is wallet data important?
**Answer:**  
Wallet data captures user monetization value. Earnings, redemptions, wallet balance, and redemption rate help identify high-value users and users who are actively participating in reward systems.

### 7. Why are transaction features useful?
**Answer:**  
Transactions provide financial behavior signals. Total transaction amount, average transaction value, and success rate can separate high-value users from casual or low-intent users.

### 8. Would you use country, city, device type, and OS directly in clustering?
**Answer:**  
I would be careful. These features are useful for profiling and campaign strategy, but direct use in clustering can create location/device-driven groups instead of behavior-driven groups. It may also introduce fairness or bias concerns.

### 9. What data issues would you expect in this schema?
**Answer:**  
Possible issues include missing timestamps, duplicate sessions, negative duration, clicks greater than impressions, missing earnings, inconsistent transaction statuses, timezone mismatch, and users existing in one table but missing in others.

### 10. What is the difference between raw data and model-ready data here?
**Answer:**  
Raw data is event-level or table-level data from different sources. Model-ready data is a cleaned, aggregated, numeric user-level feature table where every row represents one user and every column is a clustering feature.

---

## Step 3: Data Cleaning and Validation

### 1. Why is data cleaning critical for clustering?
**Answer:**  
Clustering depends heavily on distances. Invalid values like negative duration, extreme outliers, or missing timestamps can distort distances and cause wrong cluster assignments.

### 2. Why do you remove sessions with missing timestamps?
**Answer:**  
Timestamps are needed to calculate activity window, recency, and active days. A session without timestamp cannot be used reliably for time-based features.

### 3. Why do you enforce `duration > 0`?
**Answer:**  
Session duration should represent actual user activity. Zero or negative duration usually indicates logging errors or invalid records.

### 4. Why fill missing wallet values with zero?
**Answer:**  
If earnings or redeemed values are missing, it often means no observed wallet activity. Filling with zero is a reasonable business assumption, especially for sparse monetization data.

### 5. How do you handle transaction status values?
**Answer:**  
I normalize them by converting to lowercase and trimming spaces. This ensures values like `Success`, `success`, and ` success ` are treated consistently.

### 6. What would you do if clicks are greater than impressions?
**Answer:**  
That is invalid because clicks cannot exceed impressions. I would investigate the data source, and depending on frequency, either cap clicks at impressions, remove those rows, or flag them for data quality review.

### 7. Why do you validate required columns before processing?
**Answer:**  
Validation catches schema problems early. If `sessions.csv` is missing `timestamp`, the pipeline should fail with a clear error instead of silently producing incorrect features.

### 8. How would you handle duplicate session records?
**Answer:**  
I would deduplicate by `session_id`. If there are duplicate session IDs with different values, I would investigate source logic and define a deterministic rule such as keeping the latest ingestion record.

### 9. What production data quality checks would you add?
**Answer:**  
I would add checks for row counts, missing rates, duplicate keys, timestamp freshness, non-negative values, clicks less than or equal to impressions, and distribution shifts in key features.

### 10. Where is cleaning done in this project?
**Answer:**  
Cleaning is handled in `data_loader.py`, mainly inside `clean_tables()`. This keeps cleaning separate from feature engineering, which makes the pipeline easier to debug and maintain.

---

## Step 4: Feature Engineering

### 1. Why do you aggregate event data to user level?
**Answer:**  
Clustering needs one vector per entity. Since the entity is the user, all session, ad, wallet, and transaction events must be summarized into user-level features.

### 2. Explain the engagement features.
**Answer:**  
Engagement features include `session_count`, `avg_session_duration`, `total_session_duration`, `active_days`, `recency_days`, and `pages_per_session`. These capture how often, how recently, and how deeply a user interacts with the platform.

### 3. Why is `recency_days` an important feature?
**Answer:**  
Recency measures how long it has been since the user was last active. Higher recency usually indicates inactivity or churn risk.

### 4. How is CTR calculated and why is it important?
**Answer:**  
CTR is calculated as:

```python
ctr = total_clicks / total_impressions
```

It measures ad responsiveness. Users with high CTR are more likely to engage with ads.

### 5. Why do you use safe division?
**Answer:**  
Some users may have zero impressions or zero earnings. Safe division prevents division-by-zero errors and avoids infinite values.

### 6. Explain monetization features.
**Answer:**  
Monetization features include `total_earnings`, `total_redeemed`, `wallet_balance`, and `redemption_rate`. These describe how much value the user generates and how actively they redeem rewards.

### 7. Explain transaction features.
**Answer:**  
Transaction features include `total_transactions`, `total_txn_amount`, `avg_txn_amount`, and `txn_success_rate`. These capture financial engagement and transaction quality.

### 8. Why create derived scores like engagement score and churn risk score?
**Answer:**  
Derived scores combine multiple raw features into interpretable business signals. For example, churn risk combines high recency, low session count, and low earnings into one score that business teams can understand.

### 9. Why use normalized features inside derived scores?
**Answer:**  
The raw features have different scales. Sessions, earnings, and recency cannot be directly combined unless normalized. Min-max normalization makes weighted scoring meaningful.

### 10. Where is feature engineering done in this project?
**Answer:**  
Feature engineering is mainly done in `features.py`, especially inside `build_user_features()`.

---

## Step 5: Preprocessing and Scaling

### 1. Why do we need preprocessing before K-Means and DBSCAN?
**Answer:**  
Both algorithms depend on distances. If features are not cleaned, capped, transformed, and scaled, high-magnitude features like revenue or duration can dominate the clustering.

### 2. Why use median imputation?
**Answer:**  
Median is robust to outliers. Since user behavior data is usually skewed, median is safer than mean for missing value imputation.

### 3. Why cap outliers using IQR?
**Answer:**  
Outliers can pull centroids away from the majority of users. IQR capping reduces the impact of extreme values while keeping the user in the dataset.

### 4. Why use `log1p` transformation?
**Answer:**  
Features like earnings, sessions, and transaction amounts are often right-skewed. `log1p` compresses large values and handles zero safely.

### 5. Why use StandardScaler?
**Answer:**  
StandardScaler makes each feature have mean 0 and standard deviation 1. This prevents large-scale features from dominating distance calculations.

### 6. What is training-serving skew?
**Answer:**  
Training-serving skew happens when preprocessing during training differs from preprocessing during inference. For example, if the scaler is refit on new data during prediction, predictions become inconsistent.

### 7. How does this project avoid training-serving skew?
**Answer:**  
The fitted preprocessor is saved with the model artifact. During inference, the same imputation values, outlier bounds, log transform, scaler, and feature order are reused.

### 8. Why is feature order important?
**Answer:**  
ML models expect columns in the same order used during training. If feature order changes, values may be interpreted incorrectly, causing wrong predictions.

### 9. Where is preprocessing implemented?
**Answer:**  
Preprocessing is implemented in `UserSegmentationPreprocessor` inside `features.py`.

### 10. What preprocessing objects are saved?
**Answer:**  
The saved artifact includes the preprocessor, feature columns, K-Means model, DBSCAN model, and business label mapping.

---

## Step 6: K-Means Modeling

### 1. Why did you use K-Means?
**Answer:**  
K-Means is simple, scalable, and easy to explain. It creates clean macro segments that business teams can understand and use for targeting.

### 2. How does K-Means work?
**Answer:**  
K-Means starts with K centroids, assigns each user to the nearest centroid, recalculates centroids based on assigned users, and repeats until the centroids stabilize.

### 3. Why is K-Means suitable for macro segmentation?
**Answer:**  
It creates broad, global groups. For example, high-value users, casual users, and low-engagement users are macro-level groups useful for campaign strategy.

### 4. How did you choose the number of clusters?
**Answer:**  
The project trains K-Means over a range of K values and calculates inertia and silhouette score. The best K is selected mainly using silhouette score, then checked for business interpretability.

### 5. What is inertia?
**Answer:**  
Inertia is the sum of squared distances between each point and its assigned centroid. Lower inertia means tighter clusters, but it always decreases as K increases, so it cannot be used alone.

### 6. What is silhouette score?
**Answer:**  
Silhouette score measures how close a user is to its own cluster compared with other clusters. A higher score means better cluster separation.

### 7. What are the limitations of K-Means?
**Answer:**  
K-Means assumes spherical clusters, requires choosing K, is sensitive to outliers, and forces every user into a cluster even if the user is unusual.

### 8. Why are K-Means cluster IDs not directly meaningful?
**Answer:**  
Cluster IDs are arbitrary. Cluster 0 does not always mean high-value. We must profile clusters using business metrics and then assign readable labels.

### 9. Where is K-Means trained in the code?
**Answer:**  
K-Means is trained in `modeling.py`, especially inside `choose_kmeans_model()` and `train_models()`.

### 10. How would you improve K-Means performance?
**Answer:**  
I would improve feature quality, remove redundant features, tune K, check scaling, try PCA, compare MiniBatchKMeans for large data, and validate segments with business outcomes.

---

## Step 7: DBSCAN Modeling

### 1. Why did you use DBSCAN along with K-Means?
**Answer:**  
K-Means creates clean macro segments, but it does not handle outliers well. DBSCAN detects unusual users and dense niche groups, such as extreme spenders or abnormal ad clickers.

### 2. How does DBSCAN work?
**Answer:**  
DBSCAN groups points based on density. If a point has enough neighbors within radius `eps`, it becomes a core point. Connected dense regions become clusters, and isolated points become noise.

### 3. What are `eps` and `min_samples`?
**Answer:**  
`eps` is the neighborhood radius. `min_samples` is the minimum number of nearby points required to form a dense region.

### 4. What does label `-1` mean in DBSCAN?
**Answer:**  
Label `-1` means the user is considered noise or an outlier. In business terms, this user does not fit normal dense behavior patterns.

### 5. Why is DBSCAN good for anomaly detection?
**Answer:**  
DBSCAN identifies points that do not belong to any dense cluster. These points may represent unusual behavior, extreme users, or potential anomalies.

### 6. What are the limitations of DBSCAN?
**Answer:**  
DBSCAN is sensitive to `eps`, struggles with varying-density clusters, can perform poorly in high dimensions, and does not naturally provide prediction for new points.

### 7. How does this project choose DBSCAN parameters?
**Answer:**  
It tries multiple `eps` values from config and evaluates number of clusters, noise rate, and silhouette score when valid.

### 8. Why not use DBSCAN alone?
**Answer:**  
DBSCAN may produce noisy or hard-to-explain clusters. Business teams usually need clean global segments, which K-Means provides better.

### 9. How does this project handle DBSCAN prediction for new data?
**Answer:**  
Since sklearn DBSCAN has no `.predict()`, the project approximates prediction by assigning a new point to the nearest trained core sample if it is within `eps`; otherwise, it marks the user as an outlier.

### 10. Where is DBSCAN implemented?
**Answer:**  
DBSCAN training is in `choose_dbscan_model()` in `modeling.py`. Approximate DBSCAN prediction is in `approximate_dbscan_predict()` in `pipeline.py`.

---

## Step 8: Segment Profiling and Business Labeling

### 1. Why is segment profiling needed?
**Answer:**  
Clustering only gives numeric labels. Segment profiling converts those numeric clusters into business meaning using metrics like CTR, revenue, retention, sessions, and churn risk.

### 2. What metrics are used for profiling?
**Answer:**  
The project calculates users, average CTR, average revenue, average retention, average sessions, active days, recency, churn risk, monetization score, and engagement score.

### 3. How do you identify high-value users?
**Answer:**  
The cluster with the highest average revenue and monetization score is labeled as high-value users.

### 4. How do you identify ad clickers?
**Answer:**  
The cluster with the highest average CTR is labeled as ad clickers.

### 5. How do you identify low-engagement or churn-risk users?
**Answer:**  
The cluster with low engagement, low session count, and high churn risk is labeled low engagement / churn risk.

### 6. Why should labels be generated after profiling instead of hardcoded?
**Answer:**  
K-Means cluster numbers are arbitrary and can change after retraining. Business labels should be assigned based on actual cluster behavior.

### 7. What makes a segment actionable?
**Answer:**  
An actionable segment has a clear campaign strategy. For example, high-value users get premium offers, ad clickers get ad-heavy campaigns, and churn-risk users get reactivation campaigns.

### 8. Where is segment profiling done?
**Answer:**  
It is done in `segment_labels.py`, mainly through `profile_segments()` and `assign_business_labels()`.

### 9. How would you validate segment labels with business teams?
**Answer:**  
I would show segment profiles, sample users, key metrics, and campaign recommendations. Then I would run experiments to confirm that segments produce business lift.

### 10. What if two clusters both look like high-value users?
**Answer:**  
I would inspect secondary differences such as CTR, retention, redemption rate, or transaction behavior. They may represent subtypes, such as high-value ad clickers versus high-value loyal users.

---

## Step 9: Pipeline, CLI, and Project Structure

### 1. Why split the project into multiple files?
**Answer:**  
Splitting improves maintainability. Data loading, cleaning, feature engineering, modeling, prediction, API, and testing are separate, so each part can be modified and tested independently.

### 2. What does `pipeline.py` do?
**Answer:**  
`pipeline.py` connects all steps. `train_pipeline()` runs the full training workflow, and `batch_predict_pipeline()` runs prediction on new CSV data.

### 3. What does `cli.py` do?
**Answer:**  
It allows users to run training and prediction from the terminal using commands instead of writing Python code.

### 4. Why is CLI useful in production?
**Answer:**  
A CLI can be scheduled by Airflow, cron, or CI/CD pipelines. It makes the ML workflow operational and repeatable.

### 5. What is stored in the model artifact?
**Answer:**  
It stores the preprocessor, K-Means model, DBSCAN model, business labels, and feature column list.

### 6. Why do we save reports separately from the model artifact?
**Answer:**  
Reports are human-readable outputs for analysis and monitoring. The artifact is for machine inference. Keeping them separate is cleaner.

### 7. What does `generate_sample_data.py` do?
**Answer:**  
It generates synthetic data for all seven input tables. This helps test the project even without real production data.

### 8. What is the role of `config.yaml`?
**Answer:**  
It controls paths, feature columns, K-Means parameters, DBSCAN parameters, and preprocessing settings without changing code.

### 9. How would you schedule this pipeline?
**Answer:**  
I would run feature generation daily, batch scoring daily or hourly depending on business need, and retraining weekly or monthly based on drift.

### 10. How would you monitor pipeline failures?
**Answer:**  
I would monitor logs, data freshness, row counts, missing rates, artifact creation, report generation, and prediction output size.

---

## Step 10: Deployment, Testing, Monitoring, and New Dataset Prediction

### 1. How is this model deployed?
**Answer:**  
The project includes a FastAPI app and Dockerfile. The API loads the saved model artifact on startup and exposes `/health` and `/predict` endpoints.

### 2. What does the `/health` endpoint do?
**Answer:**  
It returns a simple status response so deployment systems can check whether the API is running.

### 3. What does the `/predict` endpoint do?
**Answer:**  
It accepts engineered user features, applies saved preprocessing, predicts K-Means segment, applies DBSCAN outlier logic, and returns segment labels.

### 4. Why does the API accept engineered features instead of raw CSV tables?
**Answer:**  
In production, raw events are usually processed by a feature pipeline or feature store. The API should be lightweight and fast, so it accepts already-engineered features.

### 5. How does the model work on a new dataset?
**Answer:**  
The new dataset goes through the same pipeline: load CSVs, validate schema, clean data, build user-level features, load saved artifact, apply saved preprocessing, predict clusters, assign business labels, and output predictions.

### 6. How do you test locally before deployment?
**Answer:**  
I would run unit tests, generate sample data, train the model, check reports, run batch prediction, start FastAPI locally, test `/health`, and test `/predict`.

### 7. What happens when a new model version is created?
**Answer:**  
The new model is trained, reports are generated, and it is compared with the old version using model metrics, business metrics, segment stability, and drift checks. If it passes, it is deployed to staging, tested, and then promoted to production.

### 8. What monitoring would you add after deployment?
**Answer:**  
I would monitor API latency, error rate, prediction volume, segment distribution, feature drift, model version, CTR by segment, revenue by segment, and retention by segment.

### 9. How would you handle rollback?
**Answer:**  
I would keep the previous model artifact. If the new model causes poor metrics or API issues, I would redeploy the previous artifact quickly.

### 10. What tests are included in the project?
**Answer:**  
The project includes a feature-engineering test in `tests/test_features.py`. It generates sample data, runs cleaning and feature building, and verifies that all required clustering columns are created.

---

# 20 Additional Overall Questions Not Asked Above

## 1. Write pandas code to calculate CTR safely.
**Answer:**
```python
import numpy as np

df["ctr"] = df["clicks"] / df["impressions"].replace(0, np.nan)
df["ctr"] = df["ctr"].fillna(0)
```

This avoids division by zero when impressions are zero.

## 2. Write pandas code to calculate user-level session features.
**Answer:**
```python
sessions["timestamp"] = pd.to_datetime(sessions["timestamp"])
sessions["activity_date"] = sessions["timestamp"].dt.date

session_features = sessions.groupby("user_id").agg(
    session_count=("session_id", "nunique"),
    avg_session_duration=("duration", "mean"),
    total_session_duration=("duration", "sum"),
    active_days=("activity_date", "nunique"),
    last_session_at=("timestamp", "max")
).reset_index()
```

This converts event-level session data into user-level engagement features.

## 3. How would you calculate recency?
**Answer:**
```python
reference_date = sessions["timestamp"].max()
session_features["recency_days"] = (
    reference_date - session_features["last_session_at"]
).dt.days
```

Recency tells how many days have passed since the user was last active.

## 4. How would you calculate transaction success rate?
**Answer:**
```python
transactions["is_success"] = (
    transactions["txn_status"].str.lower().str.strip() == "success"
).astype(int)

txn_features = transactions.groupby("user_id").agg(
    txn_success_rate=("is_success", "mean")
).reset_index()
```

The mean of a binary success column gives success rate.

## 5. What is the time complexity of K-Means?
**Answer:**  
K-Means is approximately `O(n * k * d * i)`, where `n` is number of users, `k` is clusters, `d` is number of features, and `i` is number of iterations.

## 6. Why can high dimensionality hurt clustering?
**Answer:**  
In high dimensions, distances become less meaningful because points tend to appear similarly far apart. This is called the curse of dimensionality. It can make distance-based clustering less reliable.

## 7. How would you reduce dimensionality?
**Answer:**  
I would remove duplicate or highly correlated features, use feature importance from downstream models, apply PCA for numerical compression, or use UMAP/t-SNE only for visualization.

## 8. Would you use PCA before K-Means?
**Answer:**  
I might use PCA if there are many correlated features or noisy dimensions. But PCA reduces interpretability, so I would compare cluster quality and business explainability before using it.

## 9. What is feature drift?
**Answer:**  
Feature drift means production feature distributions differ from training distributions. For example, average session count may drop after a product change. Drift can make cluster assignments less reliable.

## 10. How would you detect drift?
**Answer:**  
I would compare training and production distributions using PSI, KS test, missing-rate changes, mean/std changes, quantile shifts, and segment distribution changes.

## 11. What is PSI?
**Answer:**  
PSI stands for Population Stability Index. It measures how much a distribution has shifted between two datasets. Higher PSI indicates stronger drift.

## 12. How would you scale this project to millions of users?
**Answer:**  
I would compute features in Spark or a data warehouse, use MiniBatchKMeans, batch score users, store results in a feature store or user profile table, and keep the API for low-latency individual scoring.

## 13. What is MiniBatchKMeans?
**Answer:**  
MiniBatchKMeans is a faster version of K-Means that updates centroids using small random batches instead of the full dataset at every iteration.

## 14. How would you explain one user’s segment assignment?
**Answer:**  
I would compare the user’s features with the average profile of the assigned segment. For example, if a user has high sessions, high earnings, low recency, and high wallet activity, I would explain why they belong to high-value users.

## 15. What if one cluster contains 90% of users?
**Answer:**  
That may mean features are not separating users well, K is too low, scaling is poor, or behavior is genuinely concentrated. I would inspect feature distributions, try better features, tune K, and check silhouette score.

## 16. How is clustering different from classification?
**Answer:**  
Clustering discovers unknown groups without labels. Classification predicts predefined labels using supervised training data.

## 17. How can this project become supervised later?
**Answer:**  
After segments are used in campaigns, we can collect labels such as clicked, converted, retained, or churned. Then we can train supervised models for CTR prediction, churn prediction, or revenue prediction.

## 18. What ethical issues can arise in segmentation?
**Answer:**  
Segmentation may unintentionally target or exclude users based on sensitive proxies such as geography, device type, or income-related behavior. We should monitor fairness and avoid discriminatory targeting.

## 19. What would you do if CTR increases but revenue decreases?
**Answer:**  
That means clicks are not converting into valuable outcomes. I would optimize for revenue per user or campaign ROI instead of CTR alone and inspect segment-level conversion quality.

## 20. Give a strong 90-second interview pitch for this project.
**Answer:**  
“I built a production-ready user segmentation system for an ad monetization platform. The business problem was that all users were receiving generic ad targeting, which reduced CTR and revenue efficiency. I combined seven data sources including users, sessions, ad events, wallet, transactions, location, and device data. I cleaned invalid records, handled missing values, and engineered user-level features such as session count, active days, recency, CTR, earnings, redemption rate, transaction value, and churn risk score.  

For modeling, I used K-Means to create clean macro segments for business targeting and DBSCAN to detect outliers and niche user groups. I selected K using silhouette and inertia, tuned DBSCAN using eps search, and profiled every cluster using CTR, revenue, retention, engagement, and churn risk. Finally, I saved the preprocessing and model artifacts, created batch prediction, exposed the model using FastAPI, added Docker support, and included tests. The output can be used by the ad system to personalize campaigns for high-value users, casual users, churn-risk users, and ad clickers.”

