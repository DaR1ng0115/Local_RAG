# copyright (c) 2026 DaR1ng0115
# SPDX-License-Identifier: MI
from langchain_ollama import ChatOllama, OllamaEmbeddings

#加载LLM和embedding
llm = ChatOllama(model="qwen3:4b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
