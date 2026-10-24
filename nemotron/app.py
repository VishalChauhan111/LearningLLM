
import streamlit as st
from nemotron import ask_nemotron, MODEL

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a helpful AI assistant. "
        "Explain answers clearly and provide examples."
    )
}

st.set_page_config(
    page_title="Nemotron AI Chat",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Nemotron AI Chat")
st.caption(f"Powered by NVIDIA Nemotron")

if "messages" not in st.session_state:
    st.session_state.messages = [SYSTEM_MESSAGE.copy()]

with st.sidebar:
    st.subheader("Model")
    st.caption(MODEL)

    if st.button("New chat", use_container_width=True):
        st.session_state.messages = [SYSTEM_MESSAGE.copy()]
        st.rerun()

for message in st.session_state.messages:
    if message["role"] in ("user", "assistant"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

prompt = st.chat_input("Message Nemotron...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Nemotron is thinking..."):
                answer = ask_nemotron(st.session_state.messages)

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as error:
            st.session_state.messages.pop()
            st.error(f"Request failed: {error}")
