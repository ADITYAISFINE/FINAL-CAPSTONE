import streamlit as st
import uuid

from agent import ask  # your main function

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="💼",
    layout="centered"
)

st.title("💼 HR Policy Assistant")

# -------------------------------
# Initialize session state
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "employee_name" not in st.session_state:
    st.session_state.employee_name = ""

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("ℹ️ About")

    st.write("""
    This AI assistant helps employees with:
    - Leave policies
    - Working hours
    - Benefits
    - Resignation process
    - Attendance & payroll
    """)

    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.employee_name = ""
        st.rerun()

# -------------------------------
# Display chat history
# -------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# User input
# -------------------------------
user_input = st.chat_input("Ask your HR question...")

if user_input:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # call agent
    response = ask(user_input, thread_id=st.session_state.thread_id)

    answer = response.get("answer", "Sorry, I could not generate a response.")

    # store assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # display assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)