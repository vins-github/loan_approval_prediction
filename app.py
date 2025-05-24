import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model_rf.pkl')

FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Property_Area']

def get_recommendation(data, pred):
    max_loan_ratio = 20
    income = data['ApplicantIncome']
    loan = data['LoanAmount']
    term = data['Loan_Amount_Term']

    if term > 0:
        cicilan_per_bulan = loan / term
        if cicilan_per_bulan > income * 0.3:
            return (f"Loan Rejected: Estimated monthly installment ({cicilan_per_bulan:.2f}) "
                    f"is too high compared to your income ({income}). "
                    "Try increasing the loan term or reducing loan amount.")
    else:
        return "Loan Rejected: Invalid loan term."

    if loan > income * max_loan_ratio:
        return (f"Loan Rejected: Loan amount ({loan}) exceeds {max_loan_ratio}x your income ({income}).")

    if pred == 1:
        return "Congratulations! Your loan is likely to be approved."
    else:
        reasons = []
        if data['Credit_History'] == 0:
            reasons.append("Poor credit history")
        if income < 5000:
            reasons.append("Low applicant income")
        if loan > 200:
            reasons.append("Requested loan amount is too high")
        if not reasons:
            reasons.append("Some input features indicate risk")
        return "Loan Rejected due to: " + ", ".join(reasons) + " Try to improve these aspects."

st.title("Loan Approval Prediction App")

with st.form("loan_form"):
            st.markdown("""
            ### About This App

            This app predicts loan approval based on your input data using a machine learning model.  
            Fill in the form to see if your loan is likely to be **approved** or **rejected**, along with helpful recommendations.
            """)
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        credit_history_label = st.selectbox("Credit History", ["Good", "Bad"])
        credit_history = 1.0 if credit_history_label == "Good" else 0.0

    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0.0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0)
        loan_term = st.number_input("Loan Amount Term (in months)", min_value=1.0)
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Submit")

    if submitted:
        data = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }

        # Validasi awal
        cicilan_per_bulan = loan_amount / loan_term
        if cicilan_per_bulan > applicant_income * 0.3:
            st.error(
                f"Loan Rejected: Monthly installment ({cicilan_per_bulan:.2f}) "
                f"is more than 30% of your income ({applicant_income})."
            )
        elif loan_amount > applicant_income * 20:
            st.error(
                f"Loan Rejected: Loan amount ({loan_amount}) exceeds 20x your income ({applicant_income})."
            )
        else:
            input_df = pd.DataFrame([data], columns=FEATURES)
            pred = model.predict(input_df)[0]
            prediction = 'Approved' if pred == 1 else 'Rejected'
            recommendation = get_recommendation(data, pred)

            st.subheader(f"Prediction Result: {prediction}")
            st.info(recommendation)
