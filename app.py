import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor

pipe = pickle.load(open('pipe.pkl','rb'))

import base64



# Function to set the background image
def set_background(image_url):
    # Set the background image with custom CSS using a URL
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the URL to your image file
image_url = "https://i.postimg.cc/4xPqXTL6/Score.jpg"  # Replace this with the actual URL of your image
set_background(image_url)
st.title("T20 Score Predictor")
teams = ['Australia',
 'India',
 'Bangladesh',
 'New Zealand',
 'South Africa',
 'England',
 'West Indies',
 'Afghanistan',
 'Pakistan',
 'Sri Lanka',
 'Scotland',
 'Ireland',
 'Zimbabwe']

cities = ['Melbourne', 'Adelaide', 'Harare', 'Napier', 'Mount Maunganui',
       'Auckland', 'Southampton', 'Cardiff', 'Chester-le-Street',
       'Nagpur', 'Bangalore', 'Greater Noida', 'Lauderhill', 'Dubai',
       'Abu Dhabi', 'Sydney', 'Hobart', 'Wellington', 'Hamilton',
       'Bloemfontein', 'Potchefstroom', 'Barbados', 'Trinidad', 'Colombo',
       'St Kitts', 'Jamaica', 'Nelson', 'Ranchi', 'Birmingham',
       'Manchester', 'Bristol', 'Delhi', 'Rajkot', 'Lahore',
       'Johannesburg', 'Centurion', 'Cape Town', 'Cuttack', 'Indore',
       'Mumbai', 'Edinburgh', 'Dhaka', 'Sylhet', 'Sharjah', 'Karachi',
       'Dublin', 'Deventer', 'East London', 'Brisbane', 'Dehradun',
       'Bready', 'Kolkata', 'Lucknow', 'Chennai', 'Basseterre',
       'Dehra Dun', 'Visakhapatnam', 'Bengaluru', 'Canberra', 'Perth',
       'Durban', 'Port Elizabeth', 'Chandigarh', 'Christchurch', 'Kandy',
       'Chattogram', 'Pune', 'Rawalpindi', 'London', 'Nottingham',
       'King City', 'Guyana', 'St Lucia', 'Antigua', 'Pallekele',
       'Mirpur', 'Hambantota', 'Bulawayo', 'St Vincent', 'Chittagong',
       'Dominica', 'Khulna']
images = pd.read_csv("logo.csv", encoding="ISO-8859-1")
images.columns = ['Team', 'Logo Image']

# Sample list of teams; ideally, you would extract this from the CSV to ensure consistency
teams = images['Team'].tolist()

# Set up columns for team selection and logo display
col1, col2, col3 = st.columns([1, 0.5, 1])

with col1:
    # Batting team selection
    batting_team = st.selectbox('Batting team', sorted(teams), index=None, placeholder="Choose a Team")
    # Get the logo image URL for the selected batting team
    if batting_team:
        batting_logo_url = images.loc[images['Team'] == batting_team, 'Logo Image'].values[0]
        st.markdown(
            f"<div style='text-align: center;'><img src='{batting_logo_url}' width='200'><br>{batting_team}</div>",
            unsafe_allow_html=True
        )

with col2:
    # Display "VS" text in the middle
    st.markdown("<h1 style='text-align: center; font-size: 60px;'>VS</h1>", unsafe_allow_html=True)

with col3:
    # Filter options to exclude the batting team for the bowling team dropdown
    bowling_team_options = [team for team in teams if team != batting_team]
    bowling_team = st.selectbox('Bowling team', sorted(bowling_team_options), index=None, key="bowling_team", placeholder="Choose a Team")
    
    # Get the logo image URL for the selected bowling team and display it
    if bowling_team:
        bowling_logo_url = images.loc[images['Team'] == bowling_team, 'Logo Image'].values[0]
        st.markdown(
            f"<div style='text-align: center;'><img src='{bowling_logo_url}' width='200'><br>{bowling_team}</div>",
            unsafe_allow_html=True
        )

city = st.selectbox('Venue',sorted(cities))
col3,col4,col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score', min_value=0, step=1, format="%d")
with col4:
    overs = st.number_input('Overs done(works for over>8)', min_value=0, step=1, format="%d")
with col5:
    wickets = st.number_input('Wickets out', min_value=0, step=1, format="%d")
if st.button('Predict Score'):
    balls_left = 120 - (overs*6)
    wickets_left = 10 -wickets
    crr = current_score/overs

    input_df = pd.DataFrame(
     {'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':city, 'current_score': [current_score],'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr]})
    result = pipe.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))
