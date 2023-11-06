import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

with st.sidebar:
    st.title('🗨️ PDF Based Chatbot')
    st.markdown('''
    ## About App:

    The app's primary resource is utilised to create:

    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://docs.langchain.com/docs/)
    - [OpenAI](https://openai.com/)

    ## About me:

    - [Linkedin](https://www.linkedin.com/in/kunal-pamu-710674230/)
    
    ''')
    st.write("Made by Kunal Shripati Pamu")
    
def main():
    load_dotenv()
    st.header("Chat with your PDF File")

    # uploade the pdf
    pdf = st.file_uploader("Upload Your PDF:", type="pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # split text into chunks
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        # check cache for pdf name and if present use the previous embeddings else create new ones 
        store_name = pdf.name[:-4]
        # st.write(f'{store_name}')
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)
        
        # read user query
        query = st.text_input("Ask question about your PDF file:")

        # save the conversation memory
        memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
 
        if query:
            chat_history=[] 
            llm = OpenAI(temperature=0)
            qa_chain = ConversationalRetrievalChain.from_llm(llm,vector_store.as_retriever(),memory=memory)
            response=qa_chain({"question":query,"chat_history":chat_history})
            st.write(response["answer"])
            chat_history.append((query,response))

if __name__ == '__main__':
    main()