import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.set_page_config(page_title="PDF Q&A with GPT", layout="wide")
st.title("📄 Chat with Your PDFs using GPT + FAISS")

# Check for OpenAI API key
if "openai_api_key" not in st.secrets.get("general", {}):
    st.error("Please add your OpenAI API key to .streamlit/secrets.toml")
    st.stop()

openai_api_key = st.secrets["general"]["openai_api_key"]

# Initialize vectorstore in session_state if not present
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and openai_api_key:
    try:
        # Read PDF
        with st.spinner("Reading PDF..."):
            reader = PdfReader(uploaded_file)
            raw_text = ""
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    raw_text += content

        # Check if any text was extracted
        if not raw_text.strip():
            st.error("No text could be extracted from this PDF. Please try a different file.")
            st.stop()

        st.success(f"Extracted {len(raw_text)} characters from PDF")

        # Split text
        with st.spinner("Processing text..."):
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_text(raw_text)

        if not texts:
            st.error("Could not split the text into chunks. The PDF might be too short or contain unreadable text.")
            st.stop()

        st.success(f"Split text into {len(texts)} chunks")

        # Create embeddings
        with st.spinner("Creating embeddings..."):
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

            if st.session_state.vectorstore is None:
                # First upload — create new vectorstore
                st.session_state.vectorstore = FAISS.from_texts(texts, embeddings)
            else:
                # Additional upload — add to existing vectorstore
                st.session_state.vectorstore.add_texts(texts)

        st.success("PDF added to knowledge base successfully!")

        # Build QA chain
        llm = OpenAI(openai_api_key=openai_api_key)
        retriever = st.session_state.vectorstore.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # User query
        query = st.text_input("Ask something about your uploaded PDFs:")
        if query:
            with st.spinner("Thinking..."):
                try:
                    result = qa.run(query)
                    st.write("**Answer:**")
                    st.write(result)
                except Exception as e:
                    st.error(f"Error processing your question: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred while processing your PDF: {str(e)}")
        st.write("Please try uploading a different PDF or check if the file is corrupted.")

else:
    st.info("Please upload a PDF file to get started!")
