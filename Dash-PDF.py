import streamlit as st
from openai import OpenAI
import dotenv
import os
from PyPDF2 import PdfReader
from io import StringIO

dotenv.load_dotenv()

# Function to query and stream the response from the LLM
def stream_llm_response(client, model_params):
    response_message = ""

    for chunk in client.chat.completions.create(
        model=model_params["model"] if "model" in model_params else "gpt-4-turbo",
        messages=st.session_state.messages,
        temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
        max_tokens=4096,
        stream=True,
    ):
        response_message += chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        yield chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""

    st.session_state.messages.append({
        "role": "assistant", 
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]
    })

# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = StringIO()

    for page in reader.pages:
        text.write(page.extract_text())
    
    return text.getvalue()

def main():

    # --- Page Config ---
    st.set_page_config(
        page_title="The HR CONVERSATIONAL DASHBOARD",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Header ---
    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>GenAI Based: The HR CONVERSATIONAL DASHBOARD </i> 💬</h1>""")
    st.html("""<h2 style="text-align: center; color: #6ca395;"><i>Idea from Ha-Er_Weh</i></h2>""")
    st.markdown("""
    
ChatGPT
An HR Conversational Dashboard is an AI-powered tool that allows users to interact with human resources data through a conversational interface, similar to a chatbot. Unlike traditional dashboards that require manual data searches, this system enables HR managers or executives to query real-time information, such as employee performance, attendance trends, or turnover rates, and receive instant responses in the form of text or data visualizations. This streamlines the process of accessing key HR insights, saving time and improving data accessibility.

The scope of the conversational dashboard includes performance management, recruitment analytics, attendance monitoring, and employee well-being tracking. It also offers predictive analysis and personalized recommendations based on workforce data, enabling faster and more informed decision-making. By utilizing this tool, organizations can optimize their HR operations, making data-driven decisions more efficiently while enhancing the user experience for HR professionals.
    """)

    # --- Side Bar ---
    with st.sidebar:
        default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""
        with st.popover("🔐 OpenAI API Key"):
            openai_api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)", value=default_openai_api_key, type="password")

    # --- Main Content ---
    if openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key:
        st.warning("⬅️ Please introduce your OpenAI API Key to continue...")

    else:
        client = OpenAI(api_key=openai_api_key)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display previous messages if any
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])

        # Side bar model options and inputs
        with st.sidebar:
            model = st.selectbox("Select a model:", [
                "gpt-4-turbo", 
                "gpt-3.5-turbo-16k", 
                "gpt-4", 
                "gpt-4-32k",
            ], index=0)
            
            model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
            model_params = {"model": model, "temperature": model_temp}

            st.button("🗑️ Reset conversation", on_click=lambda: st.session_state.pop("messages", None))

            st.write("### **📄 Add a PDF:**")

            # PDF Upload
            uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")
            if uploaded_pdf:
                pdf_text = extract_pdf_text(uploaded_pdf)
                st.session_state.messages.append(
                    {"role": "user", "content": [{"type": "text", "text": pdf_text}]}
                )

        # Chat input
        if prompt := st.chat_input("Ask about the PDF..."):
            st.session_state.messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                st.write_stream(stream_llm_response(client, model_params))

if __name__ == "__main__":
    main()
