# 💳 Fraud-Detection-System Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2.2-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.44.1-green)
![License](https://img.shields.io/badge/License-MIT-blue)

Fraudulent transactions are a major challenge in today’s digital banking and financial systems. This project develops a **robust fraud detection system** using advanced Machine Learning techniques on a highly **imbalanced dataset**.  

The system not only predicts fraudulent transactions with high accuracy but also provides **explainability** using **SHAP (SHapley Additive exPlanations)**, helping stakeholders understand *why* a transaction is flagged as fraud.


## 🚀 Project Overview

- **Goal:** Detect fraudulent transactions from financial datasets.  
- **Dataset:** Kaggle dataset (Financial Transactions – Imbalanced).  
- **Approach:**  
  - Preprocessed and cleaned data (feature engineering, encoding, removing irrelevant features).  
  - Addressed **class imbalance** using **SMOTE (Synthetic Minority Oversampling Technique)**.  
  - Trained multiple models: Logistic Regression, Random Forest, SVM, and XGBoost.  
  - Selected **XGBoost** as the best-performing model.  
  - Applied **SHAP** for model interpretability.

## 🛠️ Tech Stack

- **Programming:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, SHAP  
- **Visualization:** Seaborn, Matplotlib, Plotly  
- **Tools:** Jupyter Notebook, Streamlit (for deployment)


## 📊 Model Performance

| Metric      | Score |
|-------------|-------|
| **F1 Score**    | 0.76  |
| **Precision**   | 0.73  |
| **Recall**      | 0.78  |
| **ROC-AUC**     | 0.82  |

✅ The model balances **fraud detection** with **minimizing false positives**.


## 🔑 Key Features

- Handles **highly imbalanced datasets** using SMOTE.  
- Evaluates models beyond accuracy (F1, Precision, Recall, ROC-AUC).  
- Provides **explainability** via SHAP to highlight features influencing fraud predictions.  
- Built as a **scalable, reproducible pipeline** for preprocessing, training, and evaluation.  
- Includes **Streamlit web app** for interactive fraud prediction.


## 🖥️ How to Run

### 1️⃣ Clone Repository
```
git clone https://github.com/naveen-142/Fraud-Detection-System.git

```
### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
### 3️⃣ Run Jupyter Notebook (for training & analysis) or Google Colab or Kaggle
```
jupyter notebook or Google Colab or Kaggle
```
### 4️⃣ Run Streamlit App (for predictions)
```
streamlit run app.py
```
## 📌 Future Improvements
- Experiment with deep learning models (LSTM/Autoencoders).

- Deploy model as a REST API with Flask/FastAPI.

- Integrate with real-time fraud detection systems.

- Extend app with dashboarding (Power BI/Tableau) for monitoring.

## 🤝 Contributions
Contributions, issues, and feature requests are welcome!
Feel free to fork this repo and submit pull requests.

📧 Contact
- 👤 Naveen Kumar Viruvuru
- 📍 Hyderabad, India
- 📩 naveenkv681@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/naveen-kumar-viruvuru/)

*✨ Detect fraud early. Save money. Protect trust.*


