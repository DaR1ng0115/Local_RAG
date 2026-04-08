# 项目名称 ：一个简单的本地RAG项目

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](./LICENSE) 文件。

## 项目简介

本项目是一个**基于本地大模型的智能文档问答系统**。你可以上传自己的文档（目前仅支持txt文件），大模型将会根据文档内容回答你的问题。整个过程完全在本地进行，无需联网，保障隐私安全。

## 核心功能

-**文档上传与管理**：支持上传'.txt'文件，并自动构建数据库。
-**智能问答**：你可以显式指明"根据文档内容进行回答"，大模型将根据你上传的文档进行解答，若问题与文档无关，大模型将仅运用训练数据进行回答。
-**流式输出**：前端页面将逐字输出回答内容。
-**性能监测**：每一次回答都将显示检索耗时，首字延迟(TTFT)和生成速度(TPS)。

## 技术栈

-**Langchain**：编排整个RAG工作流
-**ollama**：本地运行并管理LLM(大语言模型)
-**Chroma**：向量数据库，用于存储和检索文档
-**Streamlit**：构建简洁的前端Web交互页面
-**Ollama Embeddings**：提供文本嵌入模型

## 快速开始

-确保你的python版本 >= 3.10

1.克隆项目
git clone https://github.com/DaR1ng0115/Local_RAG.git
cd Local_RAG

2.安装依赖
pip install -r requirement.txt

3.启动应用
streamlit run main.py
应用将会自动在你的默认浏览器中打开

## 使用指南

1.上传文档：在左侧边栏中上传你的'.txt'文档

2.开始提问：在对话框中输入与文档相关的问题，或显式指定LLM根据文档回答问题

3.查看结果：LLM将结合文档内容回答你的问题

## 项目结构

├── main.py                  # Streamlit 主程序
├── file_uploader.py        # 文件上传与处理模块
├── performance_monitor.py  # 性能监控模块
├── config.py               # 配置文件
├── requirements.txt        # 项目依赖
└── docs/                   # 存放用户上传的文档
