# copyright (c) 2026 DaR1ng0115
# SPDX-License-Identifier: MIT
import os
from uuid import uuid4
from langchain_chroma import Chroma
import config as cg
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class FileUploader():
    """用于创建上传文件的入口"""
    def __init__(self, fileimport_instance) -> None:
        self.upload_dir = Path('./docs')
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.fileimport = fileimport_instance
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()

    def unique_filename(self, ori_filename : str) -> str:
        #为每个文件生成一个单独的文件名
        unique_id = uuid4()
        file_extension = ori_filename.split(".")[-1]
        return f"{unique_id}.{file_extension}"

    def fileupload(self):
        #创建上传文件的前端UI
        with st.sidebar:
            st.header("📁 文档管理")
            upload_file = st.file_uploader("Choose a file")
            if upload_file is not None:
                file_key = f"{upload_file.name}_{upload_file.size}"
                if file_key not in st.session_state.processed_files:
                    newname = self.unique_filename(upload_file.name)
                    filepath = self.upload_dir / newname
                    with open(filepath, "wb") as f:
                        f.write(upload_file.getbuffer())
                        self.fileimport.add_file(filepath)
                        st.session_state.processed_files.add(file_key)
                    st.success(f"文件'{upload_file.name}'已成功保存到'{self.upload_dir}'")
                else:
                    st.info(f"文件{upload_file.name}已上传过了，无需重复上传！")
            
class Fileimport():
    """读取现存文件，处理文件内容存入向量数据库"""
    def __init__(self):
        self.docs_dir = "./docs"
        self.llm = cg.llm
        self.embeddings = cg.embeddings
    
    def fileimport(self):
        #读取文件夹中的文件
        self.loader = DirectoryLoader(
        self.docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding':'utf-8'}
    )
        return self.loader

    def textsplitter(self):
        #对文档内容进行分词
        self.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 70,
        length_function = len,
        separators=["\n\n","\n",",",".","。","，"]
    )
        return self.text_splitter
    
    def creat_db(self):
        #创建向量数据库
        loader = self.fileimport()
        raw_documents = loader.load()
        text_splitter = self.textsplitter()
        if(os.path.exists("./chroma_db") and os.listdir("./chroma_db")) :
            self.db = Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)
            print("加载已有向量数据库")

        else:
            doc_splits = text_splitter.split_documents(raw_documents)
            self.db = Chroma.from_documents(
                documents=doc_splits,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
        )
            print("建立新的向量数据库")

        return self.db
    
    
    def add_file(self, filepath : Path):
        #添加新的文件
        loader = TextLoader(str(filepath), encoding='utf-8')
        new_raw_doc = loader.load()
        splitter = self.textsplitter()
        new_splitter = splitter.split_documents(new_raw_doc)
        self.db.add_documents(new_splitter)
        print(f"已添加文件:{filepath.name}")
