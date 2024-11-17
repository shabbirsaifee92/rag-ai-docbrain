import streamlit as st
import requests
import os

st.set_page_config(
    page_title="Chat",
    page_icon="🤖",
    layout="wide"
)

# Simple title and description
st.title("🤖 SOX Helper")
st.markdown("Ask me anything about companies SOX")

# Two columns layout
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Sidebar")

with col1:
    # Chat input
    question = st.text_area("Your Question", height=100)
    if st.button("Ask Chat", type="primary"):
        if question:
            with st.spinner("Thinking...."):
                try:
                    # Only include resource details if they're provided
                    payload = {
                        "text": question
                    }

                    response = requests.post(
                        f"{os.environ.get('BACKEND_URL')}/ask",
                        json=payload
                    )

                    if response.status_code == 200:
                        st.markdown("### Answer:")
                        st.write(response.json()["answer"])
                    else:
                        st.error(f"Error: {response}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {str(e)}")
        else:
            st.warning("Please enter a question!")

    st.markdown("---")
    # Show recent conversation
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if question and st.session_state.chat_history:
        st.subheader("Recent Conversations")
        for q, a in st.session_state.chat_history[-3:]:  # Show last 3 conversations
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")
