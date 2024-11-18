# Conversational PDF Query Application

This application allows users to interact with a PDF document through conversational queries using natural language processing. It employs LangChain for text processing and retrieval, OpenAI for language modeling and embeddings, Pinecone for vector storage, and Streamlit for creating an interactive user interface.

---

## Features

- **PDF Parsing**: Reads and processes the content of a PDF document.
- **Text Splitting**: Splits the PDF content into manageable chunks using a recursive character text splitter.
- **Vector Storage**: Creates and stores vector embeddings for efficient retrieval.
- **Conversational Interface**: Provides a chat-like interface for interacting with the document.
- **Contextual Query Response**: Retrieves relevant content from the document and generates responses to user queries.
- **Session State Management**: Maintains chat history and session information.

---

## Requirements

### Python Libraries
- `os`
- `langchain_openai`
- `langchain_community.document_loaders`
- `langchain_text_splitters`
- `langchain_pinecone`
- `streamlit`
- `streamlit_chat`

### Services
- **OpenAI API**: For generating embeddings and conversational responses. Requires an OpenAI API key.
- **Pinecone**: For vector storage and retrieval. Requires an account and an index.

---


