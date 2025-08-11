Conversational AI Chatbot with a Knowledge Base
This project demonstrates a conversational AI chatbot built on a Retrieval-Augmented Generation (RAG) pipeline. The chatbot is designed to answer questions based on a specific, private knowledge base provided as a PDF document. This showcases the ability to create intelligent systems that can process and understand information from a custom source.

Features ✨
Custom Knowledge Base: Uses a local PDF file as the source of all information.

Retrieval-Augmented Generation (RAG): Implements a full RAG pipeline to retrieve relevant information before generating a response, which helps prevent the model from "hallucinating."

Vector Search: Utilizes a vector database (FAISS) to perform fast and efficient semantic searches for relevant document chunks.

Instruction-Tuned Model: Uses a fine-tuned, open-source large language model (google/flan-t5-base) to generate coherent and context-aware answers.

Conversational Interface: Provides a simple command-line interface for natural language interaction.

Core Components 🧩
This project is built using a modern AI engineering stack:

LangChain: A framework for building LLM applications, used to orchestrate the entire RAG pipeline.

Hugging Face Transformers: Provides access to the open-source embedding and large language models.

FAISS: A library from Facebook AI for efficient similarity search, acting as our vector database.

PyPDF2: A Python library for extracting text from PDF documents.

Installation 🛠️
To set up this project, you will need a PDF file to use as your knowledge base. Ensure your virtual environment is active before running the commands.

Clone the repository:

Bash

git clone https://github.com/AdamKBetts/ai-projects.git
cd ai-projects/chatbot-with-knowledge-base
Install the required libraries:

Bash

pip install langchain-community langchain-huggingface faiss-cpu pypdf2 transformers
Add your PDF: Place your chosen PDF file in this directory and rename it to knowledge_base.pdf.

Usage ▶️
With your virtual environment active and your knowledge_base.pdf in place, run the rag_bot.py script.

Bash

python rag_bot.py
The first time you run the script, it will download the necessary models from Hugging Face and process your PDF. This may take a few minutes. Once the process is complete, you can start asking questions based on the content of your PDF.

To exit the chatbot, simply type exit and press Enter.

Customization ⚙️
You can easily customize the chatbot's behavior by adjusting a few parameters in rag_bot.py:

chunk_size: In the RecursiveCharacterTextSplitter section, you can modify the chunk_size to control how large the text segments are.

search_kwargs: In the new_vectorstore.as_retriever line, you can change the k value to adjust how many document chunks the chatbot retrieves to answer each question. A smaller number makes the bot more concise, while a larger number provides more context.
