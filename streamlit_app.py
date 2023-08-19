import os
import streamlit as st
#from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from config import apikey

#load_dotenv()

def main():
    st.header("Chat with PDF")

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # to avoid attribute error
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = " "
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        #st.write(text) display the contents of the pdf

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 20,
            length_function = len
        )
        chunks = text_splitter.split_text(text)

        # to store the embeddings
        store_name = pdf.name[:-4] # leaves .pdf extension
        st.write(f'{store_name}')

        if os.path.exists(f'{store_name}.pkl'):
            with open(f'{store_name}.pkl' , 'rb') as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f'{store_name}.pkl', 'wb') as f:
                pickle.dump(VectorStore, f)
        
        query = st.text_input("Ask Questions related to the PDF file: ")

        if query:                                              # k is no.of Documents to return
            docs = VectorStore.similarity_search(query = query, k=3)
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            chain = load_qa_chain(llm = llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = query)
                print(cb)
            st.write(response)



if __name__ == "__main__":
    main()