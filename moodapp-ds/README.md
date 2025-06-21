# MoodBoard Walkthrough

A simple Streamlit app practice exercise to visualize mood logs stored in Google Sheets.  
Filter by date range and moods, and see a bar chart plus notes for the selected period.

### Mood Logging
- Select from 8 emoji-based moods (e.g., 😊, 😴, 🎉, 😢)
- Add optional comments for context
- Submit logs to Google Sheets with an automatic timestamp

### Visualization Dashboard
- View mood trends over time using interactive bar charts
- Filter data by:
  - **Date range**
  - **Mood categories**
- View a full list of mood entries + notes for the selected filters

### Auto-Refreshing Front-End
- The dashboard refreshes every 60 seconds to keep visualizations up-to-date
- Designed to work smoothly with Google Sheets as a backend

### Secure Credential Handling
- Uses `st.secrets` to safely manage Google API credentials
- Prevents hardcoding sensitive data

### Libraries and Tools Used

- **Streamlit** – Web app framework for building data dashboards in Python
- **pandas** – Handles mood log data, filtering, and DataFrame manipulation
- **gspread** – Connects and writes to Google Sheets using Python
- **oauth2client** – Manages Google API authentication with service account credentials
- **plotly.express** – Generates interactive bar charts to visualize mood counts
- **streamlit_autorefresh** – Automatically refreshes the dashboard every 60 seconds
- **datetime** – Generates timestamps for each mood log submission
- **json** – Parses and loads credentials securely from secrets

### Access Instructions

- **Access the app directly here:**  
[https://okg7set28or5g4ebr33tuw.streamlit.app](https://okg7set28or5g4ebr33tuw.streamlit.app)

---

- **Link to the Google Sheet:**  
  [Google Sheet](https://docs.google.com/spreadsheets/d/1PCGMvbs-4QKf-enm3PC2e3HOah1lYXTicvsmI6Zuu2g)

---
