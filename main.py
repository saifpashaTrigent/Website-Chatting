import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from website_chatting.functions import (
    get_vectorstore_from_url,
    get_context_retriever_chain,
    get_conversational_rag_chain,
)
from PIL import Image

favicon = Image.open("favicon.png")

st.set_page_config(
    page_title="GenAI Demo | Trigent AXLR8 Labs",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke(
        {"chat_history": st.session_state.chat_history, "input": user_input}
    )

    return response["answer"]


st.header("A place where you can chat with any of the websites over the Internet 🌐.")

api_key = st.secrets["OPENAI_API_KEY"]

if api_key is None:
    raise ValueError(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
    )

if api_key:
    success_message_html = """
    <span style='color:green; font-weight:bold;'>✅ Powering the Chatbot using Open AI's 
    <a href='https://platform.openai.com/docs/models/gpt-3-5' target='_blank'>gpt-4o model</a>!</span>
    """

    # Display the success message with the link
    st.markdown(success_message_html, unsafe_allow_html=True)
    openai_api_key = api_key
else:
    openai_api_key = st.text_input("Enter your OPENAI_API_KEY: ", type="password")
    if not openai_api_key:
        st.warning("Please, enter your OPENAI_API_KEY", icon="⚠️")
        stop = True
    else:
        st.success("Ask Ai career councellor about guidance!", icon="👉")


# Sidebar Logo
logo_html = """
<style>
    [data-testid="stSidebarNav"] {
        background-image: url(https://trigent.com/wp-content/uploads/Trigent_Axlr8_Labs.png);
        background-repeat: no-repeat;
        background-position: 20px 20px;
        background-size: 80%;
    }
</style>
"""
st.sidebar.markdown(logo_html, unsafe_allow_html=True)


website_url = st.text_input("Please enter your Website URL and hit enter","https://trigent.com/")
analyze_button = st.button("Analyze")
if analyze_button:
    # Reset chat history and vector store for the new website URL
    st.session_state.chat_history = [AIMessage(content="Hello, I am a Website chatting bot. How can I help you?")]
    st.session_state.vector_store = get_vectorstore_from_url(website_url)

if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a Website chatting bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    user_query = st.chat_input("Type your message here...")

    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)



# Footer
footer_html = """
<div style="text-align: right; margin-right: 10%;">
    <p>
        Copyright © 2024, Trigent Software, Inc. All rights reserved. | 
        <a href="https://www.facebook.com/TrigentSoftware/" target="_blank">Facebook</a> |
        <a href="https://www.linkedin.com/company/trigent-software/" target="_blank">LinkedIn</a> |
        <a href="https://www.twitter.com/trigentsoftware/" target="_blank">Twitter</a> |
        <a href="https://www.youtube.com/channel/UCNhAbLhnkeVvV6MBFUZ8hOw" target="_blank">YouTube</a>
    </p>
</div>
"""

# Custom CSS to make the footer sticky
footer_css = """
<style>
.footer {
    position: fixed;
    z-index: 1000;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
}
</style>
"""


footer = f"{footer_css}<div class='footer'>{footer_html}</div>"

# Rendering the footer
st.markdown(footer, unsafe_allow_html=True)
