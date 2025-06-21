#MoodBoard Python Code

#Use Instructions as Navigation for Troubleshooting and Verification.

#Python Libraries in Use
import streamlit as st
import pandas as pd
import json
from datetime import datetime, date
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from streamlit_autorefresh import st_autorefresh

# Load Google Sheets Data

## Instructions: 
# - scope is standard in Google Sheets and should not be changed. 
# - Only alter creds with the name of the credentials file downloaded from Goodle Drive API.
# - Use sheets_url to connect to Google Sheet file with URL.
# - If multiple sheets are used, change sheet1 in sheet to the name of the sheet with Moodboard_LogSheet data. 

try: # Verify connection to Google Sheets
    #st.write("Connecting to Google Sheets...")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"] 
    creds_dict = json.loads(st.secrets["GOOGLE_CREDS_JSON"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    
    #st.write("Connecting to URL...")
    sheet_url = "https://docs.google.com/spreadsheets/d/1PCGMvbs-4QKf-enm3PC2e3HOah1lYXTicvsmI6Zuu2g"
    sheet = client.open_by_url(sheet_url).sheet1
    
except Exception as e: # Eject message of connection error
    st.error(f"Error connecting to Database. Try Again Later")
    st.stop()  
 

# User Interface Application 

## Instructions: 
# - Change the title using st.title() as desired. 
# - Provide the user with different options to log their daily mochi_mood. Note: changing or renaming the options could result in app error and data log. 
# - Add a text input for users to add additional comments on their mood for the day.
# - Use the st.button() to submit the mood and note to the Google Sheets document.

st.title("Moodboard Log!") 
st.markdown("---")
st.subheader("Your mood is very important to us. ")
st.write("Use this Moodboard to log your daily mood, add notes and visualize how your mood has changed over time since joining Mochi!")
mochi_mood = st.selectbox("Select your Mood for today:", 
    [
    "😊 Feeling Good",
    "😴 Tired",
    "😠 Frustrated",
    "😕 Uncertain",
    "😢 Sad",
    "🎉 Motivated",
    "🥴 Not Feeling Well",
    "😌 Calm"
    ])

mochi_note = st.text_input("Add any additional comments on your mood for today:")

if st.button("Submit Mood"):
    try:
        # Create a timestamp and append the entry to Google Sheets
        mood_timestamp = datetime.now().isoformat()
        sheet.append_row([mood_timestamp, mochi_mood, mochi_note])
        st.success(" Mood logged successfully! Come back tomorrow to log your next mood!")
    except Exception as e:
        st.error(f" Failed to log mood, try again.")


# Data Visualization on Streamlit

## Instructions: 
# - Appl will refresh every 60 seconds.
# - Use the get_all_records() method to retrieve all data from the Google Sheets document.
# - Check that the dataframe is not empty (contains the headers in order to continue).
# - mochimood_* contains the dataframe as established for the filters, dates and mood selection for the backend and front end filtering.

try:
    streamlit_autorefresh = 60000 
    
    mood_data = sheet.get_all_records()
    mood_df = pd.DataFrame(mood_data)

    if not mood_df.empty:
        
        mood_df['timestamp_id'] = pd.to_datetime(mood_df['timestamp_id'], errors='coerce')
        mochimood_date= mood_df[mood_df['timestamp_id'].dt.date == date.today()]
        
        st.markdown("---")
        st.subheader("See how your Mood has changed.")
        st.write("Select a date range and mood to filter your daily Mood log.")
        
        mood_startdate, mood_enddate = st.date_input("Select Date Range:", [date.today(), date.today()])
        mask = (mood_df['timestamp_id'].dt.date >= mood_startdate) & (mood_df['timestamp_id'].dt.date <= mood_enddate)
        mood_filtered = mood_df[mask]

        mood_moodfilter = mood_df['mood'].unique().tolist()
        mood_selectmood = st.multiselect("Select Moods:", options=mood_moodfilter, default=mood_moodfilter)

        
        if mood_selectmood:
            mood_filtered = mood_filtered[mood_filtered['mood'].isin(mood_selectmood)]

        mood_counts = mood_filtered['mood'].value_counts().reset_index()
        mood_counts.columns = ['Mood', 'Count']
        
        if not mood_counts.empty:
            mood_barplot = px.bar(mood_counts, x='Mood', y='Count', color='Mood',
                                  title=f"Mood Count from {mood_startdate} to {mood_enddate}")
            st.plotly_chart(mood_barplot)
            
            
            if 'note' in mood_filtered.columns:
                st.subheader(" Keep Track of your previous Mochi-Moods")
                st.write("See your logged moods and notes from the selected date and mood filters!")
                for index, row in mood_filtered.iterrows():
                    timestamp_str = row['timestamp_id'].strftime('%Y-%m-%d %H:%M')
                    mood = row['mood']
                    note = row['note']
                    st.write(f"- **{timestamp_str}** ({mood}): {note}")
                    
        else:
            st.info("No moods logged for the selected filters.")

    else:
        st.info("No data found. Log your first mood today!")

except Exception as e:
    st.error(f"Backend Loading Error. Try again.")

