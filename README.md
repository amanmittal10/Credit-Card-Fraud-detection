# 💳 Credit Card Fraud Detection using Machine Learning


**Overview**

A machine learning-based project that detects fraudulent credit card transactions using supervised learning models. The dataset is highly imbalanced, and this project demonstrates how to handle such cases effectively using resampling techniques and precision-focused evaluation.

---

** 📊 Dataset Summary**

- 📁 **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 💡 **Total records**: 284,807 transactions
- 🚩 **Fraud cases**: 492 (~0.17%)
- 🧬 **Features**: 30 (V1–V28 anonymized, Amount, Time, Class)

---

**Data Visualizations**

- Class Imbalance:
  ![Class Distribution](visuals/class_distribution.png)
  
- Model Confusion Matrices and ROC Curves:
  ![Model Results](visuals/confusion_matrix_logistic.png)
  ![ROC Curve](visuals/roc_curve.png)

---

** Models & Performance**

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | 94%      | 91%       | 76%    | 83%      |
| Decision Tree Classifier | 92%   | 89%       | 70%    | 78%      |
| Random Forest (Optional) | 97%   | 96%       | 84%    | 89%      |

**Evaluation focused on recall** to minimize false negatives in fraud detection.

---

**Technologies Used**

- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest, etc.)
- SMOTE (from imbalanced-learn) for class balancing
- Matplotlib, Seaborn for data visualization

---

** How to Run**

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
jupyter notebook
# Open and run 'fraud_detection_pipeline.ipynb'
