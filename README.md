# Credit_Card-Fraud_Detectionn
# **Introduction:**

Credit card fraud detection is a critical challenge in the financial sector due to the increasing number of fraudulent transactions. This project aims to build a machine learning pipeline to identify potentially fraudulent transactions while minimizing false positives.

# **Dataset Inormation:**

The dataset used for this project is masked credit card transaction data, containing the following:

Features: 30 features (Time, Amount, V1 to V28).
Target: Binary labels:
0: Legitimate transaction.
1: Fraudulent transaction.
Class Imbalance: The dataset is highly imbalanced, with a significantly smaller number of fraudulent transactions.

# **Project Workflow:**

Data Exploration and Preprocessing

Analyzed class distribution and feature correlations.
Scaled features (Time and Amount).
Addressed class imbalance using SMOTE.
Model Development

Implemented various machine learning algorithms:

Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes (GNB)
Hyperparameter Tuning

Optimized models using GridSearchCV.

# **Model Evaluation:**

Evaluated models using accuracy, precision, recall, F1-score, and confusion matrices.
Visualization



# **Technologies Used:**

Python: Programming language.
Libraries:
Pandas, NumPy: Data manipulation and analysis.
Matplotlib, Seaborn: Data visualization.
Scikit-learn: Machine learning and evaluation metrics.

# **Usage:**

Open the credit_card_fraud_detection.ipynb file.
Follow the step-by-step workflow to preprocess data, train models, and evaluate results.

# **Conclusion:**

In this project, we developed and evaluated multiple machine learning models to detect fraudulent credit card transactions using a real-world dataset. The Logistic Regression achieved the highest performance with an accuracy of 95.03%. and an F1-score of 0.95, making it the most reliable model for this task. Our analysis highlighted that features like transaction amount and time play a significant role in identifying fraud. This work contributes to the ongoing efforts to fight against financial fraud, providing a scalable solution for detecting anomalies in transaction data.


