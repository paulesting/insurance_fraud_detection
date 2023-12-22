import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('models/RandomForestClassifier.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Define a function to perform feature engineering on user input
def feature_engineering(user_input):
    # Define mapping dictionaries for categorical features
    month_mapping = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'Jun': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
    day_of_week_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    make_mapping = {'Lexus': 0, 'Ferrari': 1, 'Mecedes': 2, 'Porche': 3, 'Jaguar': 4, 'BMW': 5, 'Nisson': 6, 'Saturn': 7, 'Mercury': 8, 'Dodge': 9, 'Saab': 10, 'VW': 11, 'Ford': 12, 'Accura': 13, 'Chevrolet': 14, 'Mazda': 15, 'Honda': 16, 'Toyota': 17, 'Pontiac': 18}
    accident_area_mapping = {'Rural': 0, 'Urban': 1}
    day_of_week_claimed_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    month_claimed_mapping = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'Jun': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11}
    sex_mapping = {'Female': 0, 'Male': 1}
    marital_status_mapping = {'Widow': 0, 'Divorced': 1, 'Single': 2, 'Married': 3}
    fault_mapping = {'Third Party': 0, 'Policy Holder': 1}
    policy_type_mapping = {'Sport - Liability': 0, 'Sport - All Perils': 1, 'Utility - Liability': 2, 'Utility - Collision': 3, 'Utility - All Perils': 4, 'Sport - Collision': 5, 'Sedan - All Perils': 6, 'Sedan - Liability': 7, 'Sedan - Collision': 8}
    vehicle_category_mapping = {'Utility': 0, 'Sport': 1, 'Sedan': 2}
    vehicle_price_mapping = {'less than 20000': 0, '20000 to 29000': 1, '30000 to 39000': 2, '40000 to 59000': 3, '60000 to 69000': 4, 'more than 69000': 5}
    days_policy_accident_mapping = {'none': 0, '1 to 7': 1, '8 to 15': 2, '15 to 30': 3, 'more than 30': 4}
    days_policy_claim_mapping = {'8 to 15': 0, '15 to 30': 1, 'more than 30': 2}
    past_number_of_claims_mapping = {'none': 0, '1': 1, '2 to 4': 2, 'more than 4': 3}
    age_of_vehicle_mapping = {'new': 0, '2 years': 1, '3 years': 2, '4 years': 3, '5 years': 4, '6 years': 5, '7 years': 6, 'more than 7': 7}
    age_of_policy_holder_mapping = {'16 to 17':0,'18 to 20': 1, '21 to 25': 2, '26 to 30': 3, '31 to 35': 4, '36 to 40': 5, '41 to 50': 6, '51 to 65': 7, 'over 65': 8}
    police_report_filed_mapping = {'Yes': 0, 'No': 1}
    witness_present_mapping = {'Yes': 0, 'No': 1}
    agent_type_mapping = {'Internal': 0, 'External': 1}
    num_suppliments_mapping = {'none': 0, '1 to 2': 1, '3 to 5': 2, 'more than 5': 3}
    address_change_claim_mapping = {'no change': 0, 'under 6 months': 1, '1 year': 2, '2 to 3 years': 3, '4 to 8 years': 4}
    num_cars_mapping = {'1 vehicle': 0, '2 vehicles': 1, '3 to 4': 2, '5 to 8': 3, 'more than 8': 4}
    base_policy_mapping = {'All Perils': 0, 'Liability': 1, 'Collision': 2}
    week_of_month_claimed_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
 
    user_input['Month'] = month_mapping[user_input['Month']]
    user_input['DayOfWeek'] = day_of_week_mapping[user_input['DayOfWeek']]
    user_input['Make'] = make_mapping[user_input['Make']]
    user_input['AccidentArea'] = accident_area_mapping[user_input['AccidentArea']]
    user_input['DayOfWeekClaimed'] = day_of_week_claimed_mapping[user_input['DayOfWeekClaimed']]
    user_input['MonthClaimed'] = month_claimed_mapping[user_input['MonthClaimed']]
    user_input['Sex'] = sex_mapping[user_input['Sex']]
    user_input['MaritalStatus'] = marital_status_mapping[user_input['MaritalStatus']]
    user_input['Fault'] = fault_mapping[user_input['Fault']]
    user_input['PolicyType'] = policy_type_mapping[user_input['PolicyType']]
    user_input['VehicleCategory'] = vehicle_category_mapping[user_input['VehicleCategory']]
    user_input['VehiclePrice'] = vehicle_price_mapping[user_input['VehiclePrice']]
    user_input['Days_Policy_Accident'] = days_policy_accident_mapping[user_input['Days_Policy_Accident']]
    user_input['Days_Policy_Claim'] = days_policy_claim_mapping[user_input['Days_Policy_Claim']]
    user_input['PastNumberOfClaims'] = past_number_of_claims_mapping[user_input['PastNumberOfClaims']]
    user_input['AgeOfVehicle'] = age_of_vehicle_mapping[user_input['AgeOfVehicle']]
    user_input['AgeOfPolicyHolder'] = age_of_policy_holder_mapping[user_input['AgeOfPolicyHolder']]
    user_input['PoliceReportFiled'] = police_report_filed_mapping[user_input['PoliceReportFiled']]
    user_input['WitnessPresent'] = witness_present_mapping[user_input['WitnessPresent']]
    user_input['AgentType'] = agent_type_mapping[user_input['AgentType']]
    user_input['NumberOfSuppliments'] = num_suppliments_mapping[user_input['NumberOfSuppliments']]
    user_input['AddressChange_Claim'] = address_change_claim_mapping[user_input['AddressChange_Claim']]
    user_input['NumberOfCars'] = num_cars_mapping[user_input['NumberOfCars']]
    user_input['BasePolicy'] = base_policy_mapping[user_input['BasePolicy']]
    user_input['WeekOfMonthClaimed'] = week_of_month_claimed_mapping[user_input['WeekOfMonthClaimed']]
    return user_input

# Streamlit App
st.title("Insurance Fraud Detection App")
def get_user_input():
    user_input = {
        'Month': st.sidebar.selectbox('Month', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')),
        'WeekOfMonth': st.sidebar.slider('WeekOfMonth', 1, 5, 1),
        'DayOfWeek': st.sidebar.selectbox('DayOfWeek', ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')),
        'Make': st.sidebar.selectbox('Make', ('Lexus', 'Ferrari', 'Mecedes', 'Porche', 'Jaguar', 'BMW', 'Nisson', 'Saturn', 'Mercury', 'Dodge', 'Saab', 'VW', 'Ford', 'Accura', 'Chevrolet', 'Mazda', 'Honda', 'Toyota', 'Pontiac')),
        'AccidentArea': st.sidebar.selectbox('AccidentArea', ('Rural', 'Urban')),
        'DayOfWeekClaimed': st.sidebar.selectbox('DayOfWeekClaimed', ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')),
        'MonthClaimed': st.sidebar.selectbox('MonthClaimed', ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')),
        'WeekOfMonthClaimed': st.sidebar.slider('WeekOfMonthClaimed', 1, 5, 1),
        'RepNumber': st.sidebar.slider('RepNumber', 1, 30, 1),
        'Sex': st.sidebar.selectbox('Sex', ('Female', 'Male')),
        'MaritalStatus': st.sidebar.selectbox('MaritalStatus', ('Widow', 'Divorced', 'Single', 'Married')),
        'Age': st.sidebar.slider('Age', 16, 80, 30),
        'Fault': st.sidebar.selectbox('Fault', ('Third Party', 'Policy Holder')),
        'PolicyType': st.sidebar.selectbox('PolicyType', ('Sport - Liability', 'Sport - All Perils', 'Utility - Liability', 'Utility - Collision', 'Utility - All Perils', 'Sport - Collision', 'Sedan - All Perils', 'Sedan - Liability', 'Sedan - Collision')),
        'VehicleCategory': st.sidebar.selectbox('VehicleCategory', ('Utility', 'Sport', 'Sedan')),
        'VehiclePrice': st.sidebar.selectbox('VehiclePrice', ('less than 20000', '20000 to 29000', '30000 to 39000', '40000 to 59000', '60000 to 69000', 'more than 69000')),
        'Deductible': st.sidebar.slider('Deductible', 0, 1000, 500),
        'DriverRating': st.sidebar.slider('DriverRating', 0, 10, 5),
        'Days_Policy_Accident': st.sidebar.selectbox('Days_Policy_Accident', ('none', '1 to 7', '8 to 15', '15 to 30', 'more than 30')),
        'Days_Policy_Claim': st.sidebar.selectbox('Days_Policy_Claim', ('8 to 15', '15 to 30', 'more than 30')),
        'PastNumberOfClaims': st.sidebar.selectbox('PastNumberOfClaims', ('none', '1', '2 to 4', 'more than 4')),
        'AgeOfVehicle': st.sidebar.selectbox('AgeOfVehicle', ('new', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', 'more than 7')),
        'AgeOfPolicyHolder': st.sidebar.selectbox('AgeOfPolicyHolder', ('16 to 17', '18 to 20', '21 to 25', '26 to 30', '31 to 35', '36 to 40', '41 to 50', '51 to 65', 'over 65')),
        'PoliceReportFiled': st.sidebar.selectbox('PoliceReportFiled', ('Yes', 'No')),
        'WitnessPresent': st.sidebar.selectbox('WitnessPresent', ('Yes', 'No')),
        'AgentType': st.sidebar.selectbox('AgentType', ('Internal', 'External')),
        'NumberOfSuppliments': st.sidebar.selectbox('NumberOfSuppliments', ('none', '1 to 2', '3 to 5', 'more than 5')),
        'AddressChange_Claim': st.sidebar.selectbox('AddressChange_Claim', ('no change', 'under 6 months', '1 year', '2 to 3 years', '4 to 8 years')),
        'NumberOfCars':st.sidebar.selectbox('NumberOfCars', ('1 vehicle', '2 vehicles', '3 to 4', '5 to 8', 'more than 8')),
        'Year': st.sidebar.slider('Year', 1990, 2022, 2010),
        'BasePolicy': st.sidebar.selectbox('BasePolicy', ('All Perils', 'Liability', 'Collision')),
    }
    return user_input
def display_prediction(prediction):
    st.subheader('Prediction:')
    st.write(prediction)

# Sidebar for user input
st.sidebar.header("User Input Features")
user_input = get_user_input()

# Submit button
if st.sidebar.button('Submit'):
    # Perform feature engineering on user input
    user_input = feature_engineering(user_input)

    # Display user input
    st.subheader('User Input:')
    st.write(user_input)

    # Predict using the loaded model
    input_df = pd.DataFrame([user_input])
    prediction = loaded_model.predict(input_df)

    # Display the prediction
    display_prediction(prediction)