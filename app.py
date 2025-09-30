import streamlit as st
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

# ------------------ Load trained model & encoders ------------------
# Uncomment whichever model you want to test
model = joblib.load('Pickles/randomforest.pkl')   # Random Forest (GridSearchCV or normal)
# model = joblib.load('Pickles/adaboost.pkl')    # Adaboost (GridSearchCV or normal)
encoder = joblib.load('Pickles/le_encoder.pkl')   # Dictionary of LabelEncoders

# ------------------ Page Layout ------------------
st.set_page_config(page_title="Bank Fraud Detection", layout="wide")

# ------------------ Title & Description ------------------
st.title("💳 Bank Transaction Fraud Detection")

st.markdown("""
<p style='font-size:18px; text-align:center;'>
Predict whether a bank transaction is <b>fraudulent</b> or <b>legitimate</b>.
</p>
""", unsafe_allow_html=True)

# ------------------ Centered Image ------------------
st.markdown("""
<div style="text-align: center;">
    <img src="https://www.devfi.com/wp-content/uploads/2022/05/Usecase-Recommender-Image-2.svg" width="600">
</div>
""", unsafe_allow_html=True)

st.divider()

st.subheader("Why This App Matters in Today’s Market")
st.markdown("""
<p style='font-size:16px;'>
In today’s digital banking world, fraudulent transactions are increasing rapidly. Hackers and scammers are constantly finding new ways to exploit online banking, mobile payments, and other digital channels. 
A single fraudulent transaction can cause significant financial losses and erode customer trust.<br><br>

This application helps banks, financial institutions, and individuals proactively identify suspicious transactions in real time. 
By analyzing factors such as transaction amount, device used, account age, previous fraud history, and timing, the model predicts the likelihood of fraud.<br><br>

Using this app, businesses can:<br>
<b>• Reduce financial losses</b> by flagging risky transactions before processing.<br>
<b>• Protect customers</b> by preventing unauthorized activities.<br>
<b>• Save time</b> compared to manual monitoring of each transaction.<br><br>

For users, it provides a simple way to assess the safety of a transaction and make informed decisions quickly.
</p>
""", unsafe_allow_html=True)

st.divider()
st.subheader("Enter Transaction Details:")

# ------------------ Single Transaction Input ------------------
col1, col2 = st.columns(2)

with col1:
    Transaction_Amount = st.number_input("💰 Transaction Amount:", min_value=0.0, value=100.0)
    Transaction_Type = st.selectbox("📝 Transaction Type:", ["Deposit", "Online Payment", "Transfer", "Other"])
    Device_Used = st.selectbox("📱 Device Used:", ["Mobile", "Web", "ATM", "POS Terminal"])
    Account_Age = st.number_input("📅 Account Age (years):", min_value=0, max_value=100, value=1)
    Previous_Fraud = st.selectbox("⚠️ Previous Fraud?", ["Yes", "No"])

with col2:
    Credit_Score = st.slider("📊 Credit Score:", min_value=0.0, max_value=850.0, value=600.0)
    Day = st.slider("📆 Day of Month:", min_value=1, max_value=31, value=15)
    Day_of_Week = st.selectbox("📅 Day of Week:", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    Hour = st.slider("⏰ Hour (0-23):", min_value=0, max_value=23, value=12)
    Minute = st.slider("🕒 Minute (0-59):", min_value=0, max_value=59, value=30)

user_data = pd.DataFrame({
    'Transaction_Amount':[Transaction_Amount],
    'Transaction_Type':[Transaction_Type],
    'Device_Used':[Device_Used],
    'Account_Age':[Account_Age],
    'Credit_Score':[Credit_Score],
    'Previous_Fraud':[Previous_Fraud],
    'Day':[Day],
    'Day_of_Week':[Day_of_Week],
    'Hour':[Hour],
    'Minute':[Minute],
})

# ------------------ Encode categorical columns ------------------
for col, le in encoder.items():
    if col in user_data.columns:
        user_data[col] = user_data[col].map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

# ------------------ Prediction ------------------
if st.button("🔍 Predict Transaction"):
    st.subheader("User Input Data:")
    st.dataframe(user_data)

    # If model is GridSearchCV, unwrap it
    if isinstance(model, GridSearchCV):
        best_model = model.best_estimator_
    else:
        best_model = model

    prediction = best_model.predict(user_data)[0]
    prediction_proba = best_model.predict_proba(user_data)[:,1][0]

    if prediction == 1:
        st.markdown(f"""
        <div style='text-align:center;'>
            <h2 style='color:red; font-size:28px; font-weight:bold;'>⚠️ ALERT: Transaction is FRAUD!</h2>
            <p style='color:red; font-size:20px;'>Probability of Fraud: {round(prediction_proba*100,2)}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='text-align:center;'>
            <h2 style='color:green; font-size:28px; font-weight:bold;'>✅ Transaction is LEGITIMATE</h2>
            <p style='color:green; font-size:20px;'>Probability of Fraud: {round(prediction_proba*100,2)}%</p>
        </div>
        """, unsafe_allow_html=True)

    st.balloons()


## ------------------ SHAP Explanation ------------------
if st.button("📊 Explain Prediction with SHAP"):
    with st.spinner("Generating SHAP explanation..."):
        try:
            # Use the best_estimator_ if it's a GridSearchCV
            fitted_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model

            # TreeExplainer
            explainer = shap.TreeExplainer(fitted_model)

            # Convert single-row dataframe to numpy array
            X = user_data.to_numpy()

            # Compute shap values
            shap_values = explainer.shap_values(X)

            # Handle multi-class: take class 1 (fraud)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Convert 1-row shap_values to 1D array
            shap_values = shap_values.flatten()

            # Feature names & values
            feature_names = user_data.columns
            feature_values = X.flatten()

            # Build Plotly bar chart
            colors = ["green" if val >= 0 else "red" for val in shap_values]
            fig = go.Figure(go.Bar(
                x=shap_values,
                y=feature_names,
                orientation='h',
                text=[f"Value: {v}" for v in feature_values],
                marker_color=colors
            ))

            fig.update_layout(
                title="SHAP Feature Contributions (Why this transaction is predicted as Fraud/Legit)",
                xaxis_title="Impact on Prediction",
                yaxis_title="Features",
                template="plotly_white",
                height=500,
                margin=dict(l=120, r=20, t=50, b=50)
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ SHAP could not generate explanation: {e}")
