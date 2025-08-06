import streamlit as st
from PIL import Image, ImageDraw
from io import BytesIO
import requests
import httpx
import asyncio
# Simulated backend URL (you can use your own)
CHAT_URL = "http://localhost:8000/chat"
UPDATE_DATA_URL = "http://localhost:8000/data"
# Setup session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_log" not in st.session_state:
    st.session_state.user_log = []

if "images" not in st.session_state:
    st.session_state.images = []

# Function to contact backend
async def generate_response(user_input):
    with st.spinner("Waiting for assistant..."):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(CHAT_URL, json={"message": user_input})
                print(123)
                # try:
                try:
                    data = res.json()
                    # print("Backend response:", data)
                    return {
                        "content": data.get("content", "No content"),
                        "images_link": data.get("images_link", [])
                    }
                except Exception as json_error:
                    # print("JSON parsing failed:", res.text)
                    return {
                        "content": f"Failed to parse JSON: {res.text}",
                        "images_link": []
                    }
        except Exception as e:
            return {'func': 'generate_response',"content": f"Error contacting backend: {e}",'images_link':[]}

async def update_data():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(UPDATE_DATA_URL, json={})
            # print(123)
            data_log = [a['description'] for a in res.json()['content']]
            # data = res.json()
            return data_log
    except Exception as e:
        return {'func': 'update_data',"content": f"Error contacting backend: {e}"}


# Layout with tabs
tab1, tab2 = st.tabs(["💬 Chat", "📜 Data Storage"])

# --- Tab 1: Chat Interface ---
with tab1:
    st.title("Data analysis agent")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["content"] != "":
                st.markdown(msg["content"])
            if msg['images_link'] != []:
                st.image(msg["images_link"])

    user_input = st.chat_input("Type your message...")
    if user_input:
        # Add to log

        # Display user message
        st.session_state.messages.append({
            "role": "user", "content": user_input, "images_link":[],
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get assistant response
        
        response = asyncio.run(generate_response(user_input))
        print(response)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response["content"],
            "images_link": response["images_link"],
        })

        st.session_state.user_log = asyncio.run(update_data())
        with st.chat_message("assistant"):
            if response["content"]:
                st.markdown(response["content"])
            if response.get("images_link"):
                st.image(response["images_link"])
                st.session_state.images.extend(response["images_link"])

# --- Tab 2: Message Log ---
with tab2:
    st.header("Data storage")
    if st.session_state.user_log:
        for i, msg in zip(range(len(st.session_state.user_log),0,-1),st.session_state.user_log):
            st.markdown(f"**{i}.** {msg}")
    else:
        st.info("No messages yet.")

