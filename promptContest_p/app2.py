import streamlit as st
import streamlit.components.v1 as components

st.title("✍️ Paste-Protected Input")

# HTML + JS to block clipboard paste events
components.html("""
<script>
    document.addEventListener('DOMContentLoaded', function () {
        // Block paste in all inputs and textareas
        document.querySelectorAll("textarea, input").forEach(function(el) {
            el.addEventListener("paste", function(e) {
                e.preventDefault();
                alert("Pasting is disabled in this field.");
            });
        });
    });
</script>
""")

# Native Streamlit textarea (still gets protected by JS above)
user_input = st.text_area("Enter text (paste disabled)")

# Submit button
if st.button("Submit"):
    st.write("You typed:")
    st.code(user_input)
