import uvicorn
import websockets
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import dotenv
from langchain_cohere import ChatCohere
import os
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
import json
from langchain_cohere import CohereEmbeddings


cohere_api_key ='cZWxyHPX5B72hYVgeLK45bTrwiM05v8lQ5dHGIXS'
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

class WebSocketServer:
    def __init__(self, host, port, chain):
        self.host = host
        self.port = port
        self.chain = chain
    
    async def handle_connection(self, websocket, path):
            """
            Handles a WebSocket connection by receiving messages from the client,
            processing them using the AI model, and sending back the response.

            Parameters:
            - websocket: The WebSocket connection object.
            - path: The path of the WebSocket connection.

            Returns:
            None
            """
            async for message in websocket:
                try:
                    data = json.loads(message)
                    session_id = data.get("session_id", "default_session")
                    input_text = data["input"]

                    answer = self.chain.invoke(input_text)
                    
                    response = {"answer": answer, "id":session_id}
                    memory.save_context({"question": input_text}, {"response": answer})
                    await websocket.send(json.dumps(response))
                except Exception as e:
                    error_response = {"error": str(e)}
                    await websocket.send(json.dumps(error_response))
    
    async def start_server(self):
            """
            Starts the server and listens for incoming connections.

            This method uses the websockets.serve function to create a WebSocket server
            and binds it to the specified host and port. It then waits for incoming
            connections and handles each connection using the handle_connection method.

            Note: This method runs indefinitely until the program is terminated.

            Parameters:
                self (object): The instance of the class.

            Returns:
                None
            """
            async with websockets.serve(self.handle_connection, self.host, self.port):
                await asyncio.Future()  # run forever


import tomli      
class ConfigLoader:
    def __init__(self, filepath):
        self.config = self.load_config(filepath)
    
    def load_config(self, filepath):
        """
        Loads the configuration from the given file.

        Args:
            filepath (str): The path to the TOML file.

        Returns:
            dict: The loaded configuration.

        """
        with open(filepath, "rb") as params:
            return tomli.load(params)
        

        
if __name__ == "__main__":
    config_loader = ConfigLoader("parameters.toml")
    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    server = WebSocketServer("localhost", config_loader.config["general"]["port"], chain)
    
    asyncio.run(server.start_server())