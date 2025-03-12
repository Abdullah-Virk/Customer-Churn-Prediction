# Customer-Churn-Prediction
A machine learning project to predict customer churn for a video streaming platform.   Uses Random Forest and XGBoost to analyze user behavior and subscription data.   Key features: AccountAge, MonthlyCharges, SupportTickets. ROC AUC = 0.75.  
Customer Churn Prediction for Streaming Services 🎥📊
Table of Contents

Project Overview
Key Insights
Model Performance
Feature Importance
How It Works
Installation & Usage
File Structure
Competitive Results
License
🚀 Project Overview

Objective: Predict which video streaming subscribers will churn using behavioral and account data.
Impact: Enable targeted retention strategies to reduce customer attrition.
Data: 243,787 training samples with 20 features (e.g., AccountAge, MonthlyCharges, ViewingHoursPerWeek).

🔍 Key Insights

Churn Rate: 18% of subscribers churned.
Strongest Predictors:
AccountAge (-0.2 correlation with churn)
SupportTicketsPerMonth (+0.08 correlation)
Data Quality:
No missing values or duplicates.
High TotalCharges outliers validated as genuine (old accounts).
📊 Model Performance

Model	ROC AUC	Precision	Recall
Random Forest	0.7507	0.3447	0.6219
XGBoost	0.7188	0.32	0.60
Logistic Regression	0.7577	0.00	0.00
Threshold Tuning:

At threshold=0.3, recall reaches 89.85% (prioritizing customer retention).
📌 Feature Importance

Top predictors from Random Forest:

Feature	Importance	Role in Churn Prediction
Charge_Engagement_Ratio	19.87%	Combines spending & activity
AccountAge	17.96%	Older accounts churn less
TotalCharges	10.36%	Lifetime spend indicates loyalty
SupportTicketsPerMonth	3.86%	Frequent tickets signal frustration
⚙️ How It Works

Data Preprocessing:
Engineered features like CompositeEngagementScore (normalized viewing/download metrics).
Handled class imbalance using class_weight='balanced' in Random Forest.
Model Training:
Optimized via GridSearchCV for hyperparameters (max_depth=12, n_estimators=200).
Threshold Adjustment: Custom thresholds to balance precision/recall trade-offs.
💻 Installation & Usage

Dependencies

bash
Copy
pip install pandas scikit-learn numpy matplotlib seaborn xgboost
Run Prediction

python
Copy
# Load data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Generate predictions (example)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(max_depth=12, n_estimators=200, class_weight='balanced')
model.fit(X_train, y_train)
predicted_probability = model.predict_proba(X_test)[:, 1]

# Save submission
prediction_df.to_csv("prediction_submission.csv", index=False)
📂 File Structure

Copy
.
├── train.csv                 # Training data (243,787 rows)
├── test.csv                  # Test data (104,480 rows)
├── ChurnPrediction.ipynb     # Main analysis notebook
├── prediction_submission.csv # Sample output
└── data_descriptions.csv     # Feature metadata
🏆 Competitive Results

ROC AUC: Achieved 0.7462 on private leaderboard.
Ranking: Top 25% in Coursera's Data Science Challenge.
Business Impact: Model identifies 85% of at-risk customers, enabling proactive retention campaigns.
📜 License

MIT License - Free for academic/commercial use with attribution.

Let's connect💼 
[[LinkedIn](https://www.linkedin.com/in/abdullah-virk-b7bb851ab/)] | 📧 [[Email](abdullahkaemail000@gmail.com)]
