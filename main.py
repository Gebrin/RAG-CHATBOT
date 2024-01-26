import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from htmlTemplate import css, gpt_template, mixtral_template
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain


# This function used to get the pdf document and get the text out of it
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# this function is to create the chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# to vecotrize the chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    repo_id = "mistralai/Mistral-7B-v0.1"
    llm = HuggingFaceHub(huggingfacehub_api_token='XXXXXXXXXXX',
                         repo_id=repo_id, model_kwargs={"temperature": 0.2, "max_new_tokens": 50})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            messages = []
            openai.api_key = 'XXXXXXXXXXXXX'
            # Keep repeating the following
            while True:
                # This store the user questio to message so that openai api can retrive the answer
                message = user_question

                # Exit program if user inputs "quit"
                if message.lower() == "quit":
                    break

                # Add each new message to the list
                messages.append({"role": "user", "content": message})

                # Request gpt-3.5-turbo for chat completion
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )

                # Print the response and add it to the messages list
                chat_message = response['choices'][0]['message']['content']
                st.write(mixtral_template.replace(
                    "{{MSG}}", chat_message), unsafe_allow_html=True)
                messages.append({"role": "assistant", "content": chat_message})
        else:
            st.write(gpt_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Query Answering Application")
    st.header("Welcome to Query Answering Application")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
