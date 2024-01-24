import streamlit as st
from PyPDF2 import PdfReader



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def handle_userinput():
    pass
def main():
    st.set_page_config(page_title="Query Answering Application")
    st.header("Welcome to Query Answering Application")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf
                raw_text = get_pdf_text(pdf_docs)
                # proces pdf
                # Rag Mixtreal AI
                # chatgpt openai query
                # conversation chain

if __name__ == '__main__':
    main()
