# copyright (c) 2026 DaR1ng0115
# SPDX-License-Identifier: MI
import sqlite3
import streamlit as st
import config as cg
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from performance_monitor import PerformanceMonitorHandler
from file_uploader import FileUploader
from file_uploader import Fileimport

#-------------------------RAG部分-------------------------

@st.cache_resource
def load_vector_db():
    #加载向量数据库
    fileimport = Fileimport()
    db = fileimport.creat_db()
    return db,fileimport

def load_rag_chain(db):
    llm = cg.llm
    embeddings = cg.embeddings

#prompt提示词
    prompt_template = ChatPromptTemplate.from_template("""
收到用户的问题后，先判断是否需要查找信息，如果需要，请基于以下上下文回答问题。如果不需要，正常回答。这个判断过程请在后台隐式进行，
不需要明确告知用户本次回答没有查找上下文，但如果使用了上下文内容，请告知用户本次回答查找了上下文。                                   

上下文：
{context}                                     
                                     
问题：{question}                                    
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

#建立chain
    rag_chain = (
        {
            "context": (lambda x: x["question"]) | db.as_retriever(k=3) | format_docs,
            "question": lambda x: x["question"],
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain

#初始化对话数据库
def init_chat_db(db_path="chat_history.db") :
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

#-------------------------streamlit部分-------------------------

#性能监视器
monitor = PerformanceMonitorHandler()

#标题
st.set_page_config(page_title="本地RAG问答",page_icon="📚")
st.title("本地RAG问答")
st.caption("基于你的文档，使用 Qwen3-4B 本地模型")

#定义向量库
db, fileimport = load_vector_db()

#创建文件上传实例（对应file_uploader文件中的创建文件上传的前端UI方法)
file_loader = FileUploader(fileimport)
file_loader.fileupload()

#创建RAG链实例
rag_chain = load_rag_chain(db)

#创建对话数据库
conn = init_chat_db()

if "messages" not in st.session_state:
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()
    st.session_state.messages = [{"role" : row[0], "content" : row[1]} for row in reversed(rows)]
else:
    pass

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("输入你的问题:")
if prompt and prompt.strip():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in rag_chain.stream({"question" : prompt}, config={"callbacks" : [monitor]}):
            full_response += chunk
            placeholder.markdown(full_response + "|")
        placeholder.markdown(full_response)
        answer = full_response

    with st.sidebar:
        st.subheader("⚡ 实时性能指标")
        col1, col2, col3 = st.columns(3)
        col1.metric("📄 检索耗时", f"{monitor.metrics['retrieval_latency']:.3f}s")
        col2.metric("⌛ 首 Token (TTFT)", f"{monitor.metrics['ttft']:.3f}s")
        col3.metric("⚡ 生成速度 (TPS)", f"{monitor.metrics['tps']:.1f} tok/s")

    st.session_state.messages.append({"role": "assistant", "content": answer})

    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ("user", prompt))
    cursor.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ("assistant", answer))
    conn.commit()
