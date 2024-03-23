from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from streamlit.components.v1 import html
from streamlit_chat import message
import os

def main():
    st.title("Video Quiz Generator Project")
    st.write("This is the Video Quiz Generator project.")
    st.write("Back to menu:")
    st.button("Back", on_click=lambda: os.system("streamlit run main.py"))

    if 'prompts' not in st.session_state:
        st.session_state.prompts = []
    if 'responses' not in st.session_state:
        st.session_state.responses = []

    col1, col2 = st.columns([1,2])

    def send_click():
        if st.session_state.user != '':
            prompt = st.session_state.user
            if prompt:
                docs = knowledge_base.similarity_search(prompt)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=prompt)
            st.session_state.prompts.append(prompt)
            st.session_state.responses.append(response)

    load_dotenv()
    col1.header("PDF Text")
    col1.header("Ask your PDF 💬")

    pdf_path = r"C:\Users\ujwal\Pictures\Screenshots\uj.pdf"  

    pdf_reader = PdfReader(pdf_path)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    t1=f"""<font color='black'>{text}</fon>"""
    with col2:
        html(t1, height=400, scrolling=True)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    st.text_input("Ask a question about your PDF:", key="user")
    st.button("Send", on_click=send_click)

    if st.session_state.prompts:
        for i in range(len(st.session_state.responses)-1, -1, -1):
            message(st.session_state.responses[i], key=str(i), seed='Milo')
            message(st.session_state.prompts[i], is_user=True, key=str(i) + '_user', seed=83)

if __name__ == "__main__":
    main()
