import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('logistic.pkl','rb'))
def predict(gender,age,hypertension,heart_disease,avg_glucose_level,bmi,smoking_status):
    input=np.array([[gender,age,hypertension,heart_disease,avg_glucose_level,bmi,smoking_status]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0],2)
    return float(pred)
def main():
    st.title("Prediction")
    html_temp="""
    <div style="background-color:#e2062c ;padding:10px">
    <h2 style="color:white;text-align:center;">Stroke Prediction </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    gender=st.text_input("gender")
    age=st.text_input("age")
    hypertension=st.text_input("hypertension")
    heart_disease=st.text_input("heart_disease")
    avg_glucose_level=st.text_input("avg_glucose_level")
    bmi=st.text_input("bmi")
    smoking_status=st.text_input("smoking_status")
    safe_html="""
    <div style="background-color:##32cd32;padding:10px">
    <h2 style="color:white;text-align:center;">True</h2>
    </div>
    """
    danger_html="""
    <div style="background-color:#DC3545;padding:10px">
    <h2 style="color:black;text-align:center;">false</h2>
    </div>
    """
    if st.button("Predict"):
        output=predict(gender,age,hypertension,heart_disease,avg_glucose_level,bmi,smoking_status)
        if output>0.5:
            st.markdown(safe_html,unsafe_allow_html=True)
        else:
            st.markdown(danger_html,unsafe_allow_html=True)
if __name__=='__main__':
    main()