import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import tempfile
import os
import json
import re
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoModelForCausalLM
import torch
import librosa
import numpy as np
import streamlit.components.v1 as components
import base64
import io

# ---------------- GOOGLE SHEET SETUP ----------------
# Define scope
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
# Load credentials (update the filename with your JSON key)
credentials = ServiceAccountCredentials.from_json_keyfile_name("incomeexpensetracker-464907-3fa726c704f9.json", scope)
client = gspread.authorize(credentials)
# Open your Google Sheet
sheet = client.open("IncomeExpenseTracker").sheet1

# ---------------- WHISPER SETUP (Using Hugging Face Transformers) ----------------
@st.cache_resource
def load_whisper_model():
    """Load Whisper model from Hugging Face instead of OpenAI"""
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            "openai/whisper-base",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        processor = AutoProcessor.from_pretrained("openai/whisper-base")
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16,
        )
        return pipe
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

# ---------------- IMPROVED MISTRAL SETUP ----------------
@st.cache_resource
def load_mistral_model():
    """Load Mistral-7B model using AutoTokenizer and AutoModelForCausalLM"""
    try:
        # Load tokenizer and model directly
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        # Create pipeline with the loaded model and tokenizer
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        return pipe
    except Exception as e:
        st.error(f"Error loading Mistral model: {str(e)}")
        return None

# Alternative: Use pipeline directly (simpler approach)
@st.cache_resource
def load_mistral_pipeline():
    """Load Mistral using pipeline directly - simpler approach"""
    try:
        pipe = pipeline(
            "text-generation", 
            model="mistralai/Mistral-7B-Instruct-v0.3",
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.9
        )
        return pipe
    except Exception as e:
        st.error(f"Error loading Mistral pipeline: {str(e)}")
        return None

# ---------------- VOICE PROCESSING FUNCTIONS ----------------
def process_audio_from_base64(base64_data):
    """Process audio data from base64 string"""
    try:
        if not base64_data:
            return None
            
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(base64_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # Load and resample audio
        audio_array, sr = librosa.load(tmp_file_path, sr=16000)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        # Transcribe audio
        whisper_pipe = load_whisper_model()
        if whisper_pipe is None:
            return None
            
        result = whisper_pipe(audio_array)
        return result["text"]
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def get_audio_recorder_component():
    """Create a custom audio recorder component using JavaScript"""
    audio_recorder_html = """
    <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin: 10px 0;">
        <h4>🎤 Voice Recorder</h4>
        <button id="startBtn" onclick="startRecording()" style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; margin: 5px; cursor: pointer;">
            🎤 Start Recording
        </button>
        <button id="stopBtn" onclick="stopRecording()" disabled style="background-color: #f44336; color: white; padding: 10px 20px; border: none; border-radius: 5px; margin: 5px; cursor: pointer;">
            🔴 Stop Recording
        </button>
        <button id="clearBtn" onclick="clearRecording()" style="background-color: #ff9800; color: white; padding: 10px 20px; border: none; border-radius: 5px; margin: 5px; cursor: pointer;">
            🗑️ Clear
        </button>
        <div id="status" style="margin: 10px 0; font-weight: bold;">Ready to record...</div>
        <div id="timer" style="margin: 10px 0; font-size: 18px;">00:00</div>
        <audio id="audioPlayback" controls style="display: none; margin: 10px 0; width: 100%;"></audio>
    </div>

    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let recordingTimer;
        let startTime;

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    document.getElementById('audioPlayback').src = audioUrl;
                    document.getElementById('audioPlayback').style.display = 'block';
                    
                    // Convert to base64 and send to Streamlit
                    const reader = new FileReader();
                    reader.onload = function() {
                        const base64data = reader.result.split(',')[1];
                        window.parent.postMessage({
                            type: 'audioData',
                            data: base64data
                        }, '*');
                    };
                    reader.readAsDataURL(audioBlob);
                    
                    document.getElementById('status').textContent = 'Recording completed! Use the upload method below to transcribe.';
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                    clearInterval(recordingTimer);
                };

                mediaRecorder.start();
                isRecording = true;
                startTime = Date.now();
                
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('status').textContent = 'Recording...';
                
                recordingTimer = setInterval(updateTimer, 1000);
                
            } catch (err) {
                console.error('Error accessing microphone:', err);
                document.getElementById('status').textContent = 'Error: Could not access microphone';
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
            }
        }

        function clearRecording() {
            audioChunks = [];
            document.getElementById('audioPlayback').style.display = 'none';
            document.getElementById('status').textContent = 'Ready to record...';
            document.getElementById('timer').textContent = '00:00';
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            clearInterval(recordingTimer);
            
            // Notify Streamlit to clear data
            window.parent.postMessage({
                type: 'clearAudio'
            }, '*');
        }

        function updateTimer() {
            if (isRecording) {
                const elapsed = Math.floor((Date.now() - startTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                document.getElementById('timer').textContent = 
                    `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }
    </script>
    """
    return audio_recorder_html

def extract_expense_details(text):
    """Extract expense details from text using Mistral-7B with improved messaging format"""
    try:
        # Try to load the pipeline-based approach first (simpler)
        mistral = load_mistral_pipeline()
        if mistral is None:
            # Fallback to the manual model loading approach
            mistral = load_mistral_model()
        
        if mistral is None:
            return None
        
        # Use the chat format with messages (as per your requirement)
        messages = [
            {
                "role": "user", 
                "content": f"""You are a helpful assistant that extracts financial information from text. 

Extract expense/income details from the following text and return ONLY a valid JSON object with these exact fields:
- "type": "Income" or "Expense"
- "particulars": brief description of what it's for
- "amount": numeric amount only (no currency symbols)
- "category": "Need", "Want", or "Others"
- "location": "Farm" or "Home"
- "comments": any additional notes

If any information is missing, make reasonable assumptions:
- If type is unclear, assume "Expense"
- If location is unclear, assume "Home"  
- If category is unclear, assume "Others"

Text: "{text}"

Return only the JSON object, no other text."""
            }
        ]
        
        # Generate response
        try:
            # Try the messages format first
            response = mistral(messages, max_new_tokens=200, temperature=0.1)
            if isinstance(response, list) and len(response) > 0:
                response_text = response[0].get('generated_text', '')
                # Extract the assistant's response if it's in conversation format
                if isinstance(response_text, list) and len(response_text) > 1:
                    response_text = response_text[-1].get('content', '')
                elif isinstance(response_text, str):
                    # If the response includes the full conversation, extract the last part
                    parts = response_text.split('Return only the JSON object, no other text.')
                    if len(parts) > 1:
                        response_text = parts[-1].strip()
            else:
                response_text = str(response)
        except:
            # Fallback to simple string prompt if messages format fails
            simple_prompt = f"""Extract expense/income details from the following text and return ONLY a valid JSON object with these exact fields:
- "type": "Income" or "Expense"
- "particulars": brief description of what it's for
- "amount": numeric amount only (no currency symbols)
- "category": "Need", "Want", or "Others"
- "location": "Farm" or "Home"
- "comments": any additional notes

Text: "{text}"

Return only the JSON object, no other text."""
            
            response = mistral(simple_prompt, max_new_tokens=200, temperature=0.1)
            response_text = response[0]['generated_text'] if isinstance(response, list) else str(response)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                parsed_data = json.loads(json_str)
                return parsed_data
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                json_str = json_str.replace("'", '"')  # Replace single quotes
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
                parsed_data = json.loads(json_str)
                return parsed_data
        else:
            # If no JSON found, try to extract information manually
            st.warning("Could not extract JSON from AI response. Trying manual extraction...")
            return extract_manually(text)
            
    except Exception as e:
        st.error(f"Error processing with Mistral: {str(e)}")
        return extract_manually(text)

def extract_manually(text):
    """Manual extraction as fallback"""
    try:
        # Simple rule-based extraction
        text_lower = text.lower()
        
        # Determine type
        if any(word in text_lower for word in ['earned', 'income', 'received', 'got', 'salary']):
            entry_type = "Income"
        else:
            entry_type = "Expense"
        
        # Extract amount
        amount_match = re.search(r'(\d+)', text)
        amount = int(amount_match.group(1)) if amount_match else 0
        
        # Determine location
        location = "Farm" if any(word in text_lower for word in ['farm', 'field']) else "Home"
        
        # Determine category
        if any(word in text_lower for word in ['need', 'necessary', 'essential', 'bill', 'groceries']):
            category = "Need"
        elif any(word in text_lower for word in ['want', 'coffee', 'entertainment', 'movie']):
            category = "Want"
        else:
            category = "Others"
        
        # Extract particulars
        particulars = text[:50] if len(text) > 50 else text
        
        return {
            "type": entry_type,
            "particulars": particulars,
            "amount": amount,
            "category": category,
            "location": location,
            "comments": "Auto-extracted"
        }
    except:
        return None

def validate_extracted_data(data):
    """Validate and clean extracted data"""
    if not data:
        return None
        
    # Set defaults and validate
    validated = {
        "type": data.get("type", "Expense") if data.get("type") in ["Income", "Expense"] else "Expense",
        "particulars": data.get("particulars", "")[:100],  # Max 100 chars
        "amount": 0,
        "category": data.get("category", "Others") if data.get("category") in ["Need", "Want", "Others"] else "Others",
        "location": data.get("location", "Home") if data.get("location") in ["Farm", "Home"] else "Home",
        "comments": data.get("comments", "")
    }
    
    # Clean amount
    try:
        amount_str = str(data.get("amount", 0))
        amount_num = re.findall(r'\d+', amount_str)
        if amount_num:
            validated["amount"] = int(amount_num[0])
        else:
            validated["amount"] = 0
    except:
        validated["amount"] = 0
    
    return validated

# ---------------- APP ----------------
st.set_page_config(page_title="Expense & Income Tracker", layout="centered")

# Add custom CSS for voice input styling
st.markdown("""
<style>
.voice-section {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    border-left: 5px solid #1f77b4;
}
.voice-status {
    font-size: 16px;
    font-weight: bold;
    margin-bottom: 10px;
}
.transcription-box {
    background-color: #e8f4f8;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #d4edda;
    margin: 10px 0;
    min-height: 100px;
}
.submit-audio-btn {
    background-color: #007bff;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 16px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📥 Record Entry", "📊 View Summary"])

with tab1:
    st.header("💼 Record Income or Expense")
    
    # Voice Input Section
    st.markdown('<div class="voice-section">', unsafe_allow_html=True)
    st.markdown("### 🎤 Voice Input")
    
    # Initialize session state
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None
    if 'processing_audio' not in st.session_state:
        st.session_state.processing_audio = False
    if 'audio_processed' not in st.session_state:
        st.session_state.audio_processed = False
    
    # Custom audio recorder component
    audio_recorder_html = get_audio_recorder_component()
    
    # Handle JavaScript messages
    audio_component = components.html(audio_recorder_html, height=250)
    
    # Alternative: Upload Audio File
    st.markdown("**Alternative: Upload Audio File**")
    uploaded_file = st.file_uploader("Upload an audio file if the recorder doesn't work", 
                                   type=['wav', 'mp3', 'mp4', 'm4a', 'webm'])
    
    # Process uploaded file for transcription only
    if uploaded_file is not None:
        if st.button("🔊 Transcribe Audio", key="transcribe_btn"):
            with st.spinner("🔄 Transcribing audio..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # Transcribe audio
                whisper_pipe = load_whisper_model()
                if whisper_pipe:
                    try:
                        audio_array, sr = librosa.load(tmp_file_path, sr=16000)
                        result = whisper_pipe(audio_array)
                        transcribed_text = result["text"]
                        
                        if transcribed_text:
                            st.session_state.transcribed_text = transcribed_text
                            st.session_state.audio_processed = True
                            st.success("✅ Audio transcribed successfully!")
                        else:
                            st.error("❌ Could not transcribe audio. Please try again.")
                            
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
                    finally:
                        # Clean up temp file
                        os.unlink(tmp_file_path)
    
    # Always show the transcription text area and controls
    st.markdown("### 📝 Transcribed Text")
    st.markdown("**Upload an audio file above and click 'Transcribe Audio', or manually type your expense/income details:**")
    
    # Editable text area showing transcribed text
    edited_text = st.text_area(
        "Transcribed Text / Manual Input",
        value=st.session_state.transcribed_text,
        height=100,
        placeholder="E.g., I spent 500 rupees on groceries at home, it's a need",
        help="Edit the transcribed text if needed, or manually type your expense/income details, then click 'Submit Audio for Processing'"
    )
    
    # Update session state if text is edited
    if edited_text != st.session_state.transcribed_text:
        st.session_state.transcribed_text = edited_text
    
    # Submit Audio for Processing button - always show if there's text
    if st.session_state.transcribed_text.strip():
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🧠 Submit Text for Processing", key="submit_audio_btn", help="Process the text with AI to extract expense/income details"):
                with st.spinner("🧠 Processing text with AI..."):
                    # Extract details using Mistral
                    extracted_data = extract_expense_details(st.session_state.transcribed_text)
                    validated_data = validate_extracted_data(extracted_data)
                    st.session_state.extracted_data = validated_data
                
                if validated_data:
                    st.success("✅ Details extracted successfully! Check the form below.")
                else:
                    st.warning("⚠️ Could not extract all details. Please fill the form manually.")
        
        with col2:
            if st.button("🗑️ Clear All Data", key="clear_audio_btn", help="Clear current text and start fresh"):
                st.session_state.audio_data = None
                st.session_state.transcribed_text = ""
                st.session_state.extracted_data = None
                st.session_state.audio_processed = False
                st.rerun()
    
    # Display extracted data preview
    if st.session_state.extracted_data:
        st.markdown("### 🔍 Extracted Details Preview")
        st.json(st.session_state.extracted_data)
        st.info("💡 The form below has been automatically filled with the extracted data. You can review and modify before submitting.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Manual Entry Form (with pre-filled data from voice if available)
    st.markdown("### ✍️ Manual Entry / Review")
    st.markdown("**Review and modify the details below, then click 'Submit Entry' to save to Google Sheets:**")
    
    with st.form("entry_form", clear_on_submit=True):
        date = st.date_input("Select Date", datetime.date.today())
        
        # Pre-fill with voice data if available
        if st.session_state.get('extracted_data') and st.session_state.extracted_data:
            validated = st.session_state.extracted_data
            default_location = validated.get("location", "Home")
            default_type = validated.get("type", "Expense")
            default_particulars = validated.get("particulars", "")
            default_amount = validated.get("amount", 1)
            default_category = validated.get("category", "Others")
            default_comments = validated.get("comments", "")
        else:
            default_location = "Home"
            default_type = "Expense"
            default_particulars = ""
            default_amount = 1
            default_category = "Others"
            default_comments = ""
        
        location = st.radio("Location", ["Farm", "Home"], 
                          index=0 if default_location == "Farm" else 1)
        entry_type = st.radio("Type", ["Expense", "Income"], 
                            index=0 if default_type == "Expense" else 1)
        particulars = st.text_input("Particulars", value=default_particulars, max_chars=100)
        amount = st.number_input("Amount (₹)", min_value=1, step=1, value=max(1, default_amount))
        category = st.selectbox("Category", ["Need", "Want", "Others"], 
                              index=["Need", "Want", "Others"].index(default_category))
        comments = st.text_area("Comments / Notes (optional)", value=default_comments)

        submitted = st.form_submit_button("💾 Submit Entry", help="Save this entry to Google Sheets")

        if submitted:
            if not particulars or not amount or not category:
                st.warning("Please fill all mandatory fields.")
            else:
                try:
                    sheet.append_row([str(date), location, entry_type, particulars, int(amount), category, comments])
                    st.success("✅ Entry saved to Google Sheet successfully!")
                    
                    # Clear session state after successful submission
                    st.session_state.transcribed_text = ""
                    st.session_state.extracted_data = None
                    st.session_state.audio_data = None
                    st.session_state.audio_processed = False
                    
                    # Show success message with details
                    st.info(f"Saved: {entry_type} of ₹{amount} for {particulars} ({category}) at {location}")
                    
                except Exception as e:
                    st.error(f"❌ Error saving to Google Sheet: {str(e)}")

with tab2:
    st.header("📈 Income / Expense Graphs")

    try:
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
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_income = filtered[filtered["Type"] == "Income"]["Amount"].sum()
                    st.metric("Total Income", f"₹{total_income:,}")
                with col2:
                    total_expense = filtered[filtered["Type"] == "Expense"]["Amount"].sum()
                    st.metric("Total Expense", f"₹{total_expense:,}")
                with col3:
                    net_amount = total_income - total_expense
                    st.metric("Net Amount", f"₹{net_amount:,}")
            else:
                st.warning("No matching records found.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

# ---------------- SIDEBAR INFO ----------------
st.sidebar.header("🎤 Voice Input Guide")
st.sidebar.markdown("""
**How to use Voice Input:**

**New Workflow:**
1. **Record Audio**: Click "🎤 Start Recording" and speak your expense/income details
2. **Stop Recording**: Click "🔴 Stop Recording" when done
3. **Upload Method**: Save the recording and upload it using the file uploader below the recorder
4. **Transcribe**: Click "🔊 Transcribe Audio" to convert speech to text
5. **Review Text**: Check the transcribed text in the text box (you can edit it if needed)
6. **Process with AI**: Click "🧠 Submit Text for Processing" to extract details
7. **Review Form**: Check the auto-filled form below
8. **Submit**: Click "💾 Submit Entry" to save to Google Sheets

**Alternative Method:**
- Directly upload an audio file and click "🔊 Transcribe Audio"
- Or manually type your details in the text area
- Follow the same workflow from step 6

**Example phrases:**
- "I spent 500 rupees on groceries at home, it's a need"
- "Earned 2000 from farm selling vegetables"
- "Bought coffee for 150 rupees, it's a want"
- "Paid electricity bill 800 rupees at home"

**Tips for better recognition:**
- Speak clearly and at moderate pace
- Mention amount, purpose, and location
- Specify if it's income or expense
- Record in a quiet environment
- Keep recordings under 30 seconds

**New Features:**
- ✅ Improved Mistral integration with better error handling
- ✅ Enhanced AI processing with fallback methods
- ✅ Support for chat-based message format
- ✅ Manual extraction as backup
- ✅ Better JSON parsing and validation

**Troubleshooting:**
- If recorder doesn't work, use file upload method
- Ensure microphone permissions are granted
- Try refreshing the page if issues persist
- Supported file formats: WAV, MP3, MP4, M4A, WEBM
- AI processing includes fallback methods for reliability
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**💡 Pro Tip:** The updated Mistral integration now supports both message-based and simple prompt formats with better error handling!")