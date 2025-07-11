import streamlit as st
import time
import threading
import json
from datetime import datetime, timedelta
from streamlit_js_eval import streamlit_js_eval

# Initialize session state variables
if 'timer_active' not in st.session_state:
    st.session_state.timer_active = False
if 'timer_end_time' not in st.session_state:
    st.session_state.timer_end_time = None
if 'player1_submitted' not in st.session_state:
    st.session_state.player1_submitted = False
if 'player2_submitted' not in st.session_state:
    st.session_state.player2_submitted = False
if 'player1prompt' not in st.session_state:
    st.session_state.player1prompt = ""
if 'player2prompt' not in st.session_state:
    st.session_state.player2prompt = ""
if 'contest_task' not in st.session_state:
    st.session_state.contest_task = ""
if 'evaluation_result' not in st.session_state:
    st.session_state.evaluation_result = ""

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        color: #A23B72;
        font-size: 1.5rem;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .timer-active {
        background-color: #90EE90;
        color: #006400;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .timer-expired {
        background-color: #FF6B6B;
        color: #8B0000;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .player-column {
        border: 2px solid #2E86AB;
        border-radius: 10px;
        padding: 20px;
        margin: 10px;
    }
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .stTextInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .copy-button {
        background-color: #4CAF50;
        color: white;
        padding: 8px 16px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .copy-button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

def start_timer():
    """Start the 60-second countdown timer"""
    st.session_state.timer_active = True
    st.session_state.timer_end_time = datetime.now() + timedelta(seconds=60)
    st.session_state.player1_submitted = False
    st.session_state.player2_submitted = False
    st.session_state.evaluation_result = ""

def get_remaining_time():
    """Calculate remaining time in seconds"""
    if st.session_state.timer_end_time:
        remaining = (st.session_state.timer_end_time - datetime.now()).total_seconds()
        return max(0, remaining)
    return 0

def format_time(seconds):
    """Format seconds into MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def disable_copy_paste():# Global disable paste for all elements
    st.components.v1.html("""
        <script>
            document.addEventListener('paste', function (e) {
                e.preventDefault();
                alert('Pasting is disabled across this app. Please type your input.');
            });
        </script>
    """, height=0)

def copy_to_clipboard(text):
    """Return HTML + JS to copy `text` to clipboard with a button"""
    # Escape for JavaScript string inside HTML
    safe_text = json.dumps(text)  # Properly escapes quotes, newlines, backslashes
    js_code = f"""
    <script>
        function copyToClipboard() {{
            const text = {text};
            navigator.clipboard.writeText(text).then(function() {{
                alert('Evaluation results copied to clipboard!');
            }}, function(err) {{
                console.error('Could not copy text: ', err);
                const textArea = document.createElement('textarea');
                textArea.value = text;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
                alert('Evaluation results copied to clipboard!');
            }});
        }}
    </script>
    <button onclick="copyToClipboard()">📋 Copy to Clipboard</button>
    """
    return js_code

def evaluate_prompts(prompt1, prompt2):
    """Evaluate and concatenate both prompts"""
    if not prompt1.strip() or not prompt2.strip():
        return "Error: Both players must submit their prompts before evaluation."
    
    combined_prompt = f"""
    === PROMPT BATTLE EVALUATION ===
    
    Contest Task: {st.session_state.contest_task}
    
    {p1_first} {p1_last}'s Prompt:
    {prompt1}
    
    {p2_first} {p2_last}'s Prompt:
    {prompt2}
    
    Combined Evaluation Prompt:
    Compare and analyze the effectiveness of these two prompts in addressing the given contest task.
    Consider creativity, clarity, relevance, and potential impact.
    Rank them between 1-5 for each category and show that on table format, with total score.

    """
    return combined_prompt

# Main App Layout
st.markdown('<h1 class="main-title">🦅 Prompt Battle PlayGround</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-title">Play like an Eagle</h2>', unsafe_allow_html=True)

# Contest Task Input
st.markdown("### 📝 Contest Task")
contest_task = st.text_area(
    "Enter the contest task or challenge prompt:",
    value=st.session_state.contest_task,
    height=100,
    key="task_input",
    placeholder="Example: Create a prompt that generates a creative story about time travel..."
)

if contest_task != st.session_state.contest_task:
    st.session_state.contest_task = contest_task

# Timer Section
st.markdown("### ⏰ Timer")
col_timer1, col_timer2, col_timer3 = st.columns([1, 2, 1])

with col_timer2:
    if st.button("🚀 START", type="primary", use_container_width=True):
        if st.session_state.contest_task.strip():
            start_timer()
            st.rerun()
        else:
            st.error("Please enter a contest task before starting the timer!")

# Display Timer
if st.session_state.timer_active:
    remaining_time = get_remaining_time()
    if remaining_time > 0:
        st.markdown(f'<div class="timer-active">⏱️ Time Remaining: {format_time(remaining_time)}</div>', unsafe_allow_html=True)
        time.sleep(1)
        st.rerun()
    else:
        st.markdown('<div class="timer-expired">🔴 TIME\'S UP!</div>', unsafe_allow_html=True)
        st.session_state.timer_active = False

# Player Information Section
st.markdown("### 👥 Players")
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="player-column">', unsafe_allow_html=True)
    st.markdown("#### 🎮 Player 1")
    p1_first = st.text_input("First Name", key="p1_first", placeholder="Enter first name")
    p1_last = st.text_input("Last Name", key="p1_last", placeholder="Enter last name")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="player-column">', unsafe_allow_html=True)
    st.markdown("#### 🎮 Player 2")
    p2_first = st.text_input("First Name", key="p2_first", placeholder="Enter first name")
    p2_last = st.text_input("Last Name", key="p2_last", placeholder="Enter last name")
    st.markdown('</div>', unsafe_allow_html=True)

# Prompt Entry Section
st.markdown("### ✍️ Prompt Entry")
st.markdown("**Note:** Copy and paste functions are disabled for fair play!")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"#### Player 1: {p1_first} {p1_last}" if p1_first and p1_last else "#### Player 1")
    player1_prompt = st.text_area(
        "Enter your prompt:",
        height=200,
        key="player1_textarea",
        placeholder="Type your creative prompt here...",
        disabled=st.session_state.player1_submitted
    )
    
    if st.button("Submit Player 1 Prompt", key="submit_p1", disabled=st.session_state.player1_submitted):
        if player1_prompt.strip():
            st.session_state.player1prompt = player1_prompt
            st.session_state.player1_submitted = True
            st.success(f"✅ {p1_first}'s prompt submitted successfully!")
            st.rerun()
        else:
            st.error("Please enter a prompt before submitting!")

with col2:
    st.markdown(f"#### Player 2: {p2_first} {p2_last}" if p2_first and p2_last else "#### Player 2")
    player2_prompt = st.text_area(
        "Enter your prompt:",
        height=200,
        key="player2_textarea",
        placeholder="Type your creative prompt here...",
        disabled=st.session_state.player2_submitted
    )
    
    if st.button("Submit Player 2 Prompt", key="submit_p2", disabled=st.session_state.player2_submitted):
        if player2_prompt.strip():
            st.session_state.player2prompt = player2_prompt
            st.session_state.player2_submitted = True
            st.success(f"✅ {p2_first}'s prompt submitted successfully!")
            st.rerun()
        else:
            st.error("Please enter a prompt before submitting!")

# Evaluation Section
st.markdown("### 🏆 Evaluation")
if st.session_state.player1_submitted and st.session_state.player2_submitted:
    if st.button("🔍 EVALUATE PROMPTS", type="primary", use_container_width=True):
        st.session_state.evaluation_result = evaluate_prompts(
            st.session_state.player1prompt, 
            st.session_state.player2prompt
        )
        st.rerun()

# Display Evaluation Results
if st.session_state.evaluation_result:
    st.markdown("### Combined Prompt for Evaluation:")
    
    evaluation_result_text = st.session_state.evaluation_result
    # Add copy to clipboard button
    st.code(evaluation_result_text, language='html')
    st.code(evaluation_result_text, language='plain')


# Reset Button
if st.button("🔄 Reset Contest", type="secondary"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666; font-style: italic;">🦅 Social Eagle Prompt Battle Contest - May the best prompt win!</p>',
    unsafe_allow_html=True
)