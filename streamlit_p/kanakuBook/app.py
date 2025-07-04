import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ---------------- GOOGLE SHEET SETUP ----------------
# Define scope
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# Load credentials (update the filename with your JSON key)
credentials = ServiceAccountCredentials.from_json_keyfile_name("incomeexpensetracker-464907-3fa726c704f9.json", scope)
client = gspread.authorize(credentials)
# Open your Google Sheet
sheet = client.open("IncomeExpenseTracker").sheet1

# ---------------- APP ----------------
st.set_page_config(page_title="Expense & Income Tracker", layout="centered")

# Tabs
tab1, tab2 = st.tabs(["📥 Record Entry", "📊 View Summary"])

with tab1:
    st.header("💼 Record Income or Expense")

    # Form
    with st.form("entry_form", clear_on_submit=True):
        date = st.date_input("Select Date", datetime.date.today())
        location = st.radio("Location", ["Farm", "Home"])
        entry_type = st.radio("Type", ["Expense", "Income"])
        particulars = st.text_input("Particulars", max_chars=100)
        amount = st.number_input("Amount (₹)", min_value=1, step=1)
        category = st.selectbox("Category", ["Need", "Want", "Others"])
        comments = st.text_area("Comments / Notes (optional)")

        submitted = st.form_submit_button("Submit Entry")

        if submitted:
            if not particulars or not amount or not category:
                st.warning("Please fill all mandatory fields.")
            else:
                sheet.append_row([str(date), location, entry_type, particulars, int(amount), category, comments])
                st.success("Entry saved to Google Sheet!")

with tab2:
    st.header("📈 Income / Expense Graphs")

    df = pd.DataFrame(sheet.get_all_records())
    if df.empty:
        st.info("No data to display yet.")
    else:
        df["Date"] = pd.to_datetime(df["Date"])
        from_date = st.date_input("From Date", value=df["Date"].min())
        to_date = st.date_input("To Date", value=df["Date"].max())

        # Filters
        selected_type = st.multiselect("Filter by Type", options=["Income", "Expense"], default=["Income", "Expense"])
        selected_location = st.multiselect("Filter by Location", options=["Farm", "Home"], default=["Farm", "Home"])

        filtered = df[
            (df["Date"] >= pd.to_datetime(from_date)) &
            (df["Date"] <= pd.to_datetime(to_date)) &
            (df["Type"].isin(selected_type)) &
            (df["Location"].isin(selected_location))
        ]

        if not filtered.empty:
            fig = px.bar(
                filtered,
                x="Date",
                y="Amount",
                color="Type",
                barmode="group",
                title="Income vs Expense Over Time",
                hover_data=["Particulars", "Category", "Comments"]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No matching records found.")
