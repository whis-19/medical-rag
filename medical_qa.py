import os
import pandas as pd
from getpass import getpass
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Load environment variables from .env file
load_dotenv()

# API Key setup
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("Enter your Gemini API Key: ")
print("API Key set.")

# Constants
DATA_FILE_PATH = "datasets/mtsamples.csv" 
CHROMA_PATH = "chroma_db_medical"

print("Setup complete.") 

# Check if Chroma DB already exists
if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
    print(f"Found existing Chroma DB at {CHROMA_PATH}. Loading...")
    print("\n1. Initializing Gemini Embedding Model (text-embedding-004)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
    print("2. Loading existing vector store...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    print("Vector Store loaded successfully!")
else:
    print(f"No existing Chroma DB found. Creating new one at {CHROMA_PATH}...")
    print(f"1. Loading documents from {DATA_FILE_PATH}...")
    loader = CSVLoader(file_path=DATA_FILE_PATH, encoding="utf-8", csv_args={
        'delimiter': ',',
        'quotechar': '"'
    })

    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    transcription_docs = []
    print("2. Processing documents...")
    for doc in tqdm(docs, desc="Processing documents"):
        new_doc = Document(
            page_content=doc.page_content, 
            metadata={"source": doc.metadata.get("source"), "row": doc.metadata.get("row")} 
        )
        transcription_docs.append(new_doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,     # This size is a good balance for medical factoids.
        chunk_overlap=50,   # Small overlap to maintain context across chunk boundaries.
        separators=["\n\n", "\n", ". ", " ", ""] 
    )

    print("3. Splitting documents into chunks...")
    documents = text_splitter.split_documents(transcription_docs)
    print(f"Total documents loaded: {len(docs)}")
    print(f"Total chunks created after splitting: {len(documents)}")

    print("\n4. Initializing Gemini Embedding Model (text-embedding-004)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004") 
    
    print("5. Creating Chroma vector store and running embeddings...")
    print("This may take a few minutes depending on your data size...")
    
    # Create vector store with progress indication
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Vector Store creation complete! The medical knowledge base is ready for RAG.")

print("1. Initializing Gemini LLM...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

print("2. Setting up the Chroma Retriever...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

system_prompt_template = (
    "You are a helpful and extremely safe medical assistant. Your primary task is to answer the user's question "
    "**ONLY** based on the provided clinical context below. You must not use any external knowledge. "
    "If the context does not contain the answer, you must state clearly: 'I cannot provide an answer based on the provided medical context.' "
    "For every piece of information you provide, you **MUST** include a citation, referencing the original source document "
    "using the format: [Source: Row X]. The context documents contain a 'row' metadata field that identifies the source."
    "\n\nContext:\n{context}"
)

print("3. Defining the Citation-Enforcing Prompt...")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("human", "{input}"),
    ]
)

print("4. Creating the Document Combination Chain...")
document_chain = create_stuff_documents_chain(llm, prompt)

print("5. Creating the final Retrieval Chain...")
rag_chain = create_retrieval_chain(retriever, document_chain)

print("RAG Pipeline built successfully! Ready for first test query.")