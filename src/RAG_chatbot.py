import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import streamlit as st
from streamlit_chat import message




loader = PyPDFLoader("../documents/Samplepdf.pdf")
pages = []
for page in loader.load():
    pages.append(page)

text_splitter = RecursiveCharacterTextSplitter(
chunk_size=1000, chunk_overlap=200, add_start_index=True)
                                                 
all_splits = text_splitter.split_documents(pages)

embeddings = OpenAIEmbeddings(api_key= os.environ['OPENAI_API_KEY'])

index_name ="vectorindex"

vectorstore = PineconeVectorStore.from_documents(all_splits,embeddings,index_name=index_name)

    # Define a prompt template to structure the LLM input
# prompt_template = """
# Based on the given context provide 3 lines of  response to the query

# Query:
# {query}

# Response:
# """

# # Fill in the prompt with the retrieved content and the query
# prompt = PromptTemplate(
# input_variables=["query"],
# template=prompt_template
# ).format(query=query)
    
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(
        temperature=0.5,
        model_name="gpt-4o-mini"
    ),
    retriever=vectorstore.as_retriever()
    )


def converse_history(query):
    result = chain(
        {"question":query,
         "chat_history":st.session_state['history']})
    st.session_state['history'].append((query,result["answer"]))

    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history']=[]
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello, How can I help you?"]
if 'past' not in st.session_state:
    st.session_state['past']=['Hi']
    
response_container=st.container()
container=st.container()

with container:
    with st.form(key='my_website',clear_on_submit=True):
        customer_input=st.text_input("Query:",placeholder="How can I help you?")
        submit_button=st.form_submit_button(label='Enter')
    if submit_button and customer_input:
        output = converse_history(customer_input)
        
        st.session_state['past'].append(customer_input)
        st.session_state['generated'].append(output)
        
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i],is_user=True,key=str(i)+'_user')
            message(st.session_state['generated'][i],key=str(i))