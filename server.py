from fastapi import FastAPI
from langserve import add_routes
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_cohere import CohereEmbeddings

app = FastAPI(
    title="eva",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

cohere_api_key = os.getenv("API")
store = {}
app = FastAPI()
session_memories = {}

embeddings = CohereEmbeddings(
    cohere_api_key=cohere_api_key,
    model="embed-english-v3.0",
)

retriever = Chroma(persist_directory="pocs/chroma_db",collection_name='patient1', embedding_function=embeddings).as_retriever()
chat = ChatCohere(cohere_api_key=cohere_api_key)
str_out = StrOutputParser()

prompt_template = PromptTemplate(
    input_variables=["history", "context", "question"],
    template="""
You are a really sympathetic and caring medical assistant for question-answering tasks. 
Your name is 'Eva'. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 

### Conversation History:
{history}

### Retrieved Context:
{context}

### User Question:
{question}

### Eva's Response:
"""
)

from langchain.memory import ConversationBufferMemory

prompt_template = PromptTemplate(
    input_variables=["history", "context", "question"],
    template="""
You are a really sympathetic and caring medical assistant for question-answering tasks. 
Your name is 'Eva'. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Also just stick to the point and give concise and short answers.

### Conversation History:
{history}

### Retrieved Context:
{context}

### User Question:
{question}

### Eva's Response:
"""
)
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
chain = (
    {
        "history": RunnableLambda(lambda x: memory.load_memory_variables(x)["history"]),
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | chat
    | str_out
)
   

add_routes(
    app,
    chain,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn
    port = os.getenv("PORT",10000)
    uvicorn.run(app, host="0.0.0.0", port=port)
