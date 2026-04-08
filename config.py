from langchain_ollama import ChatOllama, OllamaEmbeddings

#加载LLM和embedding
llm = ChatOllama(model="qwen3:4b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")