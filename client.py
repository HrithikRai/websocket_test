from langserve import RemoteRunnable
joke_chain = RemoteRunnable("http://localhost:8000/joke/")
print(joke_chain.invoke("what questions have i asked?")
)