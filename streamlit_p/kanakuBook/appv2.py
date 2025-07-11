import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------- GOOGLE SHEET SETUP ----------------
# Define scope
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# Load credentials (update the filename with your JSON key)
credentials = ServiceAccountCredentials.from_json_keyfile_name("incomeexpensetracker-464907-3fa726c704f9.json", scope)
client = gspread.authorize(credentials)
# Open your Google Sheet
sheet = client.open("IncomeExpenseTracker").sheet1

# ---------------- AI MODEL LOADING ----------------
@st.cache_resource
def load_model():
    """
    Load the Magistral-Small-2506 model and tokenizer
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Magistral-Small-2506")
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Magistral-Small-2506",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

# ---------------- AI PROCESSING FUNCTION ----------------
def process_text_with_ai(text):
    """
    Process natural language text using Mistral-7B-Instruct to extract form details
    """
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error("Model not loaded. Please check your setup.")
        return None
    
    # Create a detailed prompt for the AI
    prompt = f"""<s>[INST] Extract the following information from this expense/income description and return it in JSON format:

Text: "{text}"

Please extract:
- date: Date mentioned (format: YYYY-MM-DD, if not mentioned use today's date {datetime.date.today()})
- location: Either "Farm" or "Home" (if not clear, use "Home")
- entry_type: Either "Expense" or "Income"
- particulars: Brief description of the item/transaction
- amount: Numeric amount (extract numbers only)
- category: Either "Need", "Want", or "Others"
- comments: Any additional notes or context

Return only a valid JSON object with these exact keys. Example:
{{
    "date": "2024-01-15",
    "location": "Home",
    "entry_type": "Expense",
    "particulars": "Grocery shopping",
    "amount": 500,
    "category": "Need",
    "comments": "Monthly groceries"
}}
[/INST]"""
    
    try:
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Move to the same device as the model
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the generated part (after the prompt)
        generated_text = response[len(prompt):].strip()
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed_json = json.loads(json_str)
            return parsed_json
        else:
            # Fallback: try to parse the entire generated text as JSON
            try:
                return json.loads(generated_text)
            except:
                st.error("Could not extract valid JSON from AI response")
                st.write("AI Response:", generated_text)
                return None
            
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse AI response: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during AI processing: {e}")
        return None

# ---------------- APP ----------------
st.set_page_config(page_title="Expense & Income Tracker", layout="centered")

# Tabs
tab1, tab2 = st.tabs(["📥 Record Entry", "📊 View Summary"])

with tab1:
    st.header("💼 Record Income or Expense")
    
    # AI Text Input Section
    st.subheader("🤖 AI-Powered Entry")
    st.write("Describe your expense or income in natural language, and AI will fill the form for you!")
    
    # Text input and process button
    col1, col2 = st.columns([4, 1])
    with col1:
        ai_text = st.text_area(
            "Describe your transaction",
            placeholder="e.g., 'Bought groceries for ₹500 at home today' or 'Received ₹2000 from farm sale yesterday'",
            height=100
        )
    with col2:
        st.write("")  # Empty space for alignment
        st.write("")  # Empty space for alignment
        process_button = st.button("🔍 Process", type="primary")
    
    # Initialize session state for form values
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
            'date': datetime.date.today(),
            'location': 'Home',
            'entry_type': 'Expense',
            'particulars': '',
            'amount': 1,
            'category': 'Need',
            'comments': ''
        }
    
    # Process AI text when button is clicked
    if process_button and ai_text.strip():
        with st.spinner("Processing with AI..."):
            ai_result = process_text_with_ai(ai_text)
            
            if ai_result:
                # Update session state with AI results
                try:
                    st.session_state.form_data['date'] = datetime.datetime.strptime(ai_result.get('date', str(datetime.date.today())), '%Y-%m-%d').date()
                except:
                    st.session_state.form_data['date'] = datetime.date.today()
                
                st.session_state.form_data['location'] = ai_result.get('location', 'Home')
                st.session_state.form_data['entry_type'] = ai_result.get('entry_type', 'Expense')
                st.session_state.form_data['particulars'] = ai_result.get('particulars', '')
                st.session_state.form_data['amount'] = max(1, int(ai_result.get('amount', 1)))
                st.session_state.form_data['category'] = ai_result.get('category', 'Need')
                st.session_state.form_data['comments'] = ai_result.get('comments', '')
                
                st.success("✅ Form filled automatically! Please review and modify if needed.")
            else:
                st.error("Failed to process the text. Please try again or fill the form manually.")
    
    elif process_button and not ai_text.strip():
        st.warning("Please enter some text to process.")
    
    st.divider()
    
    # Manual Form (now with AI-populated values)
    st.subheader("📝 Review and Submit")
    
    with st.form("entry_form", clear_on_submit=True):
        date = st.date_input("Select Date", value=st.session_state.form_data['date'])
        location = st.radio("Location", ["Farm", "Home"], index=0 if st.session_state.form_data['location'] == 'Farm' else 1)
        entry_type = st.radio("Type", ["Expense", "Income"], index=0 if st.session_state.form_data['entry_type'] == 'Expense' else 1)
        particulars = st.text_input("Particulars", value=st.session_state.form_data['particulars'], max_chars=100)
        amount = st.number_input("Amount (₹)", min_value=1, step=1, value=st.session_state.form_data['amount'])
        category = st.selectbox("Category", ["Need", "Want", "Others"], index=["Need", "Want", "Others"].index(st.session_state.form_data['category']))
        comments = st.text_area("Comments / Notes (optional)", value=st.session_state.form_data['comments'])

        submitted = st.form_submit_button("Submit Entry")

        if submitted:
            if not particulars or not amount or not category:
                st.warning("Please fill all mandatory fields.")
            else:
                sheet.append_row([str(date), location, entry_type, particulars, int(amount), category, comments])
                st.success("Entry saved to Google Sheet!")
                
                # Reset form data after successful submission
                st.session_state.form_data = {
                    'date': datetime.date.today(),
                    'location': 'Home',
                    'entry_type': 'Expense',
                    'particulars': '',
                    'amount': 1,
                    'category': 'Need',
                    'comments': ''
                }

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