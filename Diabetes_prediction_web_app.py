# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 19:49:46 2023

@author: HP
"""

import pickle
import numpy as np
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav','rb'))

def diabetes_prediction(input_data):
    
    input_data_as_np_array = np.asarray(input_data)
    
    input_data_reshaped = input_data_as_np_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    #giving the title
    st.title('Diabetes Prediction Web App')
    
    #getting the user input
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Level')
    Insulin = st.text_input('Insulin')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree function')
    Age = st.text_input('Age')
    
    diagnosis = ''
    
    if st.button('Diabetes Test Result : '):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
    
if __name__ == "__main__":
    main()
