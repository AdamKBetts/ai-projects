# Import Libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Indexing

# Load the document
loader = PyPDFLoader("knowledge_base.pdf")
documents = loader.load()

# Split the doc into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Create embeddings and store in a vector database
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")

# Retrieval & Generation

# Load the vector store and LLM
new_vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Load small opensource LLM
llm_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=500, model_kwargs={"temperature": 0.0, "do_sample": False},)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    new_vectorstore.as_retriever(search_kwargs={"k": 2}),
    chain_type="stuff",
    return_source_documents=True
)

# Interact with the chatbot
chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    result = qa_chain.invoke({"question": query, "chat_history": chat_history})
    print("Bot:", result["answer"])

    # Update chathistory for context
    chat_history.append((query, result["answer"]))