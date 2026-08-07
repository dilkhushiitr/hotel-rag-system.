# FAANG-Style Data Scientist Interview Q&A

Project: **User Segmentation Using K-Means and DBSCAN**

Use this as a mock interview guide. Each question is written the way a strong interviewer may ask it, followed by a model answer you can adapt in your own words.

---

## 1. Business Context and Problem Framing

### 1. Why was user segmentation needed in this project?
**Answer:** Ad targeting was broad and generic, so every user received similar campaigns. This caused inefficient ad spend, lower CTR, lower conversion, and weaker revenue per user. Segmentation helps group users by behavior so targeting can be personalized.

### 2. What business problem are you solving with clustering?
**Answer:** I am solving an unsupervised user understanding problem. The objective is to identify behaviorally similar users so ad campaigns can be tailored to high-value users, casual users, churn-risk users, and ad-responsive users.

### 3. Why is this an unsupervised learning problem?
**Answer:** There is no predefined ground-truth label such as “high-value” or “churn-risk” for every user. We use behavioral patterns in the data to discover natural groups.

### 4. What would be the business success metrics?
**Answer:** Key metrics are CTR uplift, revenue per user, campaign conversion rate, ad efficiency, retention, and reduction in wasted impressions.

### 5. How would you explain this project to a non-technical stakeholder?
**Answer:** We analyzed user behavior, spending, ad interaction, and activity patterns to group users into meaningful segments. These segments help the ad system show more relevant campaigns to each user group, improving engagement and revenue.

### 6. What is the risk of using broad targeting?
**Answer:** Broad targeting wastes impressions on users who are unlikely to click or convert, while under-serving users who are highly valuable or highly responsive to ads.

### 7. Why not directly optimize for revenue instead of clustering?
**Answer:** Direct revenue optimization needs reliable labels and experimentation. Clustering is useful first because it discovers hidden behavior groups, which can then be used for targeting, experimentation, and supervised modeling later.

### 8. How would you validate that segmentation improved business performance?
**Answer:** I would run an A/B test. The control group gets generic targeting, and the treatment group gets segment-based targeting. I would compare CTR, revenue per user, retention, and conversion rate.

### 9. What would you do if a segment is statistically interesting but not actionable?
**Answer:** I would avoid using it directly for campaign targeting. A good segment must be interpretable, stable, and actionable. If it cannot guide a business decision, it is not useful enough.

### 10. What is the difference between a model metric and a business metric here?
**Answer:** A model metric, such as silhouette score, measures cluster separation. A business metric, such as CTR uplift or revenue per user, measures actual impact. Both matter, but business impact is the final success criterion.

---

## 2. Data Schema and Data Understanding

### 1. Why did you use multiple tables instead of only one user table?
**Answer:** User behavior is spread across different systems: sessions, ads, wallet, transactions, location, and device. Combining them gives a richer user-level view.

### 2. What is the primary key for joining these tables?
**Answer:** `user_id` is the common key across all tables and is used to aggregate event-level data into one user-level feature table.

### 3. Why is `signup_date` useful?
**Answer:** It helps calculate user age. New users and older users often behave differently, so user age can help distinguish onboarding behavior from mature behavior.

### 4. Why are sessions important?
**Answer:** Sessions capture engagement. High session count, longer duration, and more active days indicate stronger user involvement.

### 5. Why are ad impressions and clicks important?
**Answer:** They allow us to calculate CTR, which indicates how responsive a user is to ads.

### 6. Why are wallet features useful?
**Answer:** Earnings, redemptions, and wallet balance indicate user value and monetization behavior.

### 7. Why include transactions?
**Answer:** Transactions capture financial behavior, spend intensity, and success rate, all of which are strong segmentation signals.

### 8. Would you use location and device in clustering?
**Answer:** I would use them carefully. They are useful for profiling, but for clustering I prefer behavioral and monetary features first. Location/device may introduce bias or create segments that are less behavior-driven.

### 9. What data quality issues would you expect?
**Answer:** Missing timestamps, duplicate user IDs, negative durations, missing wallet values, impossible click counts, inconsistent transaction statuses, and timezone issues.

### 10. What is the target granularity of this project?
**Answer:** The final model works at user level. Raw event-level data is aggregated so each row represents one user.

---

## 3. Data Cleaning

### 1. Why is cleaning especially important for clustering?
**Answer:** Clustering is distance-based, so noisy or impossible values can distort distances and pull centroids in the wrong direction.

### 2. Why remove sessions with missing timestamps?
**Answer:** Timestamps are needed to calculate active days and recency. Without a timestamp, the session cannot be placed in the activity window.

### 3. Why enforce `duration > 0`?
**Answer:** Zero or negative duration is not meaningful user activity and may indicate logging errors.

### 4. Why fill missing wallet earnings with zero?
**Answer:** Missing wallet values often mean no observed earning or redemption activity. Zero is a reasonable business default.

### 5. How would you handle duplicate rows?
**Answer:** I would deduplicate based on stable keys such as `session_id` for sessions and carefully aggregate repeated user-level rows for wallet or ad events.

### 6. How would you handle clicks greater than impressions?
**Answer:** That is invalid. I would either cap clicks at impressions, investigate the source, or remove those records depending on data ownership and frequency.

### 7. Why normalize transaction status strings?
**Answer:** Values like `Success`, `success`, and ` SUCCESS ` should mean the same thing. Normalization prevents incorrect success-rate calculations.

### 8. How do you handle missing optional tables like device or location?
**Answer:** The pipeline should not fail. It can create empty placeholder tables and still train on core behavioral features.

### 9. Why convert timestamps to datetime?
**Answer:** Datetime format is needed for time-window filtering, recency calculation, and active-day aggregation.

### 10. What cleaning checks would you automate in production?
**Answer:** Required column checks, data type checks, non-negative metric checks, duplicate checks, freshness checks, and row-count anomaly checks.

---

## 4. Feature Engineering

### 1. Why aggregate to user-level features?
**Answer:** Clustering requires each entity to have one feature vector. Since the entity is a user, all events must be summarized at user level.

### 2. Why use a 30-day activity window?
**Answer:** It captures recent behavior while reducing the effect of very old activity. Recent behavior is more relevant for ad targeting and churn risk.

### 3. Explain `session_count`.
**Answer:** It is the total number of sessions in the activity window and measures frequency of engagement.

### 4. Explain `recency_days`.
**Answer:** It measures days since the user’s last session. High recency usually means the user may be inactive or at churn risk.

### 5. How do you calculate CTR?
**Answer:** `ctr = total_clicks / total_impressions`. If impressions are zero, CTR is set to zero to avoid division errors.

### 6. Why create wallet balance?
**Answer:** Wallet balance equals earnings minus redeemed amount. It indicates remaining value and user monetization state.

### 7. Why create redemption rate?
**Answer:** It shows how much of earned value the user withdraws. This can separate active reward users from passive earners.

### 8. Why create transaction success rate?
**Answer:** It reflects transaction reliability and financial engagement quality.

### 9. Explain engagement score.
**Answer:** Engagement score combines session count, active days, and inverse recency. Users with frequent, recent, and consistent activity get higher scores.

### 10. Explain churn risk score.
**Answer:** Churn risk is high when recency is high, session count is low, and earnings are low. It is a weighted proxy score for inactivity risk.

---

## 5. Preprocessing and Scaling

### 1. Why is scaling required for K-Means?
**Answer:** K-Means uses distance. Without scaling, large-value features like earnings or duration can dominate smaller features like CTR.

### 2. Why use median imputation?
**Answer:** Median is robust to outliers and works well for skewed numerical data.

### 3. Why cap outliers using IQR?
**Answer:** Extreme values can distort distances and centroids. IQR capping reduces their influence while preserving the record.

### 4. Why use `log1p` transformation?
**Answer:** Many features like revenue, sessions, and duration are right-skewed. `log1p` compresses large values and handles zero safely.

### 5. Why not normalize before outlier treatment?
**Answer:** Outliers can affect scaling statistics. It is usually better to handle extreme values first and then scale.

### 6. What is StandardScaler?
**Answer:** It transforms each feature to mean 0 and standard deviation 1.

### 7. Should you fit scaler on train and test separately?
**Answer:** No. The scaler must be fit only on training data and reused for inference to avoid data leakage and inconsistent transformations.

### 8. What is data leakage in this project?
**Answer:** Leakage happens if future data or inference data is used to compute training-time preprocessing statistics or segment definitions.

### 9. Would one-hot encoded device/country need scaling?
**Answer:** Binary one-hot features usually do not require StandardScaler, but if mixed with continuous scaled features, careful preprocessing is needed.

### 10. What preprocessing artifacts must be saved?
**Answer:** Imputation values, outlier bounds, scaler parameters, feature column order, and trained model objects.

---

## 6. K-Means Macro Segmentation

### 1. Why use K-Means?
**Answer:** K-Means gives clean, interpretable macro segments and is easy to explain to business teams.

### 2. What assumption does K-Means make?
**Answer:** It assumes clusters are roughly spherical, similarly sized, and separable by distance to centroids.

### 3. How does K-Means work?
**Answer:** It initializes K centroids, assigns points to the nearest centroid, recalculates centroids, and repeats until convergence.

### 4. How do you choose K?
**Answer:** I use elbow method for inertia and silhouette score for separation. Then I also check business interpretability.

### 5. What is inertia?
**Answer:** Inertia is the sum of squared distances between points and their assigned cluster centroid. Lower inertia means tighter clusters.

### 6. What is silhouette score?
**Answer:** It measures how similar a point is to its own cluster compared with other clusters. It ranges from -1 to 1, where higher is better.

### 7. Why are K-Means cluster IDs arbitrary?
**Answer:** Cluster number 0 or 1 has no inherent business meaning. Labels must be assigned by profiling cluster behavior.

### 8. What are limitations of K-Means?
**Answer:** It struggles with outliers, non-spherical clusters, different-density clusters, and requires choosing K beforehand.

### 9. How would you label clusters?
**Answer:** I would aggregate metrics by cluster and label based on dominant behavior, such as high revenue, high CTR, low engagement, or casual usage.

### 10. How would you monitor K-Means in production?
**Answer:** Track segment size distribution, centroid drift, feature drift, CTR/revenue by segment, and stability of labels over time.

---

## 7. DBSCAN Micro Segmentation and Anomaly Detection

### 1. Why use DBSCAN after K-Means?
**Answer:** K-Means gives global macro segments, while DBSCAN detects dense niche groups and outliers that K-Means may force into broad clusters.

### 2. What does DBSCAN stand for?
**Answer:** Density-Based Spatial Clustering of Applications with Noise.

### 3. What are DBSCAN’s main parameters?
**Answer:** `eps`, the neighborhood radius, and `min_samples`, the minimum points needed to form a dense region.

### 4. What is a core point?
**Answer:** A point with at least `min_samples` neighbors within radius `eps`.

### 5. What is a noise point?
**Answer:** A point that does not belong to any dense region. In sklearn, it gets label `-1`.

### 6. Why is DBSCAN useful for anomalies?
**Answer:** Users far from dense normal behavior groups are labeled as noise, making DBSCAN useful for identifying unusual users.

### 7. What are DBSCAN limitations?
**Answer:** It is sensitive to `eps`, struggles with varying-density clusters, and does not naturally predict labels for new unseen points.

### 8. How do you choose `eps`?
**Answer:** Use a k-distance plot, business validation, and metrics like cluster count, noise rate, and silhouette on non-noise points.

### 9. Why not use DBSCAN alone?
**Answer:** DBSCAN may produce messy or unstable clusters and may not give clean business-friendly global segments.

### 10. How would you serve DBSCAN in an API?
**Answer:** Since DBSCAN lacks native predict, I would either use it mostly for batch anomaly detection or approximate serving by assigning new users to nearest trained core samples within `eps`.

---

## 8. Segment Profiling, Evaluation, and Deployment

### 1. Why profile clusters after training?
**Answer:** Clustering gives numeric labels, not business meaning. Profiling translates clusters into actionable segments.

### 2. Which metrics would you calculate by segment?
**Answer:** Average CTR, revenue per user, retention, session count, active days, recency, churn risk, and segment size.

### 3. How do you know if a cluster is useful?
**Answer:** It should be distinct, stable, interpretable, large enough or strategically important, and tied to a campaign action.

### 4. What is segment stability?
**Answer:** It means users with similar behavior continue to map to similar segments over time, and segment definitions do not fluctuate randomly.

### 5. How would you deploy this project?
**Answer:** Train batch models offline, save preprocessing and model artifacts, expose batch prediction and API scoring, and monitor drift and business metrics.

### 6. Why save the preprocessor with the model?
**Answer:** Inference must use the exact same feature order, imputation, capping, log transform, and scaling as training.

### 7. What should the API accept?
**Answer:** In production, it should usually accept engineered user features from a feature store, not raw event tables.

### 8. What would you log in production?
**Answer:** Request IDs, user IDs or hashed IDs, feature freshness, assigned segment, model version, latency, and errors.

### 9. How often would you retrain?
**Answer:** It depends on user behavior drift. I would start with weekly or monthly retraining and adjust based on drift and segment performance.

### 10. What can go wrong after deployment?
**Answer:** Feature drift, broken upstream data, segment-size collapse, stale models, API latency, poor business uplift, or segments becoming less interpretable.

---

# 20 Additional Mixed Interview Questions Not Asked Above

## 1. Write Python code to calculate CTR safely.
**Answer:**
```python
df["ctr"] = df["clicks"] / df["impressions"].replace(0, np.nan)
df["ctr"] = df["ctr"].fillna(0)
```

## 2. Write pandas code to calculate session count per user.
**Answer:**
```python
session_features = sessions.groupby("user_id").agg(
    session_count=("session_id", "nunique")
).reset_index()
```

## 3. How would you calculate recency days?
**Answer:**
```python
last_session = sessions.groupby("user_id")["timestamp"].max()
recency_days = (reference_date - last_session).dt.days
```

## 4. What is the time complexity of K-Means?
**Answer:** Approximately `O(n * k * d * i)`, where `n` is users, `k` is clusters, `d` is features, and `i` is iterations.

## 5. What is the time complexity concern with DBSCAN?
**Answer:** DBSCAN can be expensive for large datasets because it needs neighborhood searches. With indexing it can be efficient, but high-dimensional data can still make it slow.

## 6. Why can high-dimensional data hurt clustering?
**Answer:** In high dimensions, distances become less meaningful because points often appear similarly far apart. This is called the curse of dimensionality.

## 7. How would you reduce dimensionality?
**Answer:** Use feature selection, correlation pruning, PCA, UMAP for visualization, or remove low-signal/high-noise features.

## 8. Would you use PCA before K-Means?
**Answer:** Possibly. PCA can reduce noise and multicollinearity, but it may reduce interpretability. I would compare performance and business interpretability with and without PCA.

## 9. How would you detect feature drift?
**Answer:** Compare training and production feature distributions using PSI, KS test, mean/std changes, missing-rate changes, and quantile shifts.

## 10. What is PSI?
**Answer:** Population Stability Index measures how much a feature distribution has shifted between two populations.

## 11. How would you handle millions of users?
**Answer:** Use distributed feature generation, MiniBatchKMeans, approximate nearest neighbors for serving, and scheduled batch scoring.

## 12. What is MiniBatchKMeans?
**Answer:** It is a faster K-Means variant that updates centroids using small random batches instead of the full dataset each iteration.

## 13. How would you make the project reproducible?
**Answer:** Fix random seeds, version data and models, save config, save preprocessing artifacts, log package versions, and use Docker.

## 14. How would you version segments?
**Answer:** Store model version, config version, training date, feature schema version, and segment label mapping.

## 15. How would you explain a user’s segment assignment?
**Answer:** Compare the user’s features with the segment profile. For example, high sessions, high earnings, and low recency explain high-value assignment.

## 16. What would you do if one cluster has 90% of users?
**Answer:** Check scaling, feature quality, K choice, outlier influence, and whether the data truly has weak separation. I may add better features or use another algorithm.

## 17. What is the difference between clustering and classification?
**Answer:** Clustering discovers groups without labels. Classification predicts known labels learned from historical examples.

## 18. How could this project evolve into supervised learning?
**Answer:** After running campaigns, we can collect labels such as clicked/not clicked or converted/not converted and train propensity models for targeting.

## 19. What ethical concerns exist in segmentation?
**Answer:** Segments may encode sensitive geographic, device, or socioeconomic patterns. We should avoid discriminatory targeting and monitor fairness.

## 20. Give a strong interview summary of this project.
**Answer:** I built an unsupervised user segmentation system for ad targeting. I joined seven behavioral and monetization data blocks, cleaned invalid records, engineered user-level features such as engagement, CTR, wallet behavior, transaction success, and churn risk, then applied scaling and outlier treatment. K-Means produced interpretable macro segments, while DBSCAN detected anomalies and niche groups. Finally, I profiled each segment by CTR, revenue, and retention, and packaged the system with batch scoring, FastAPI deployment, Docker, and monitoring-ready outputs.

