# agent-engineering-notes
## agent学习知识笔记

这是一个关于大语言模型（LLM）、AI Agent以及LLM应用工程实践的技术知识库。 

本仓库主要记录在学习和开发过程中，对大模型原理、Agent架构、RAG检索增强、Tool Calling、LangChain、LangGraph、MCP以及LLMOps等相关技术的理解、源码分析和工程实践经验。 

随着大语言模型技术的发展，AI应用已经从传统的问答系统逐渐演进为具备任务规划、工具调用、知识检索和自主决策能力的智能Agent系统。 

本仓库希望通过系统化的文章整理，帮助开发者深入理解： 

- 大语言模型的工作原理  
- Agent系统的核心设计思想
- LLM应用工程落地方法 
- 企业级AI应用架构设计

为什么创建这个仓库 

大模型技术发展迅速，从模型能力到应用工程之间存在大量工程实践问题。

本仓库希望沉淀个人在AI Agent开发过程中的技术思考，将零散的学习内容整理成体系化知识，形成一套面向工程实践的Agent开发路线。  

目标 通过持续更新，希望建立一个：

- Agent工程师学习路线 
- LLM应用开发参考手册 
- AI系统架构实践文档 

帮助更多开发者理解并构建真正可落地的智能Agent应用。



# 目录

## 大语言模型RAG应用开发基础

 [大语言模型出现幻觉的原因及缓解方案](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-development-fundamentals/hallucination.md)

 [检索增强生成RAG基础架构介绍](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-development-fundamentals/enhanced-retrieval.md)

[向量数据库的介绍与用途](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-development-fundamentals/vector-database.md)

[传统数据库与向量数据库的使用差异](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-development-fundamentals/diff_trad_vs_vector_db.md)

[Embedding嵌入模型介绍](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-development-fundamentals/embedding.md)

## LangChain RAG 应用开发组件

[Document 组件与文档加载器的使用](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/langchain-rag-application-development/document-components-and-document-loaders.md)

[自定义 LangChain 文档加载器使用技巧](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/langchain-rag-application-development/custom-langchain-document-loader.md)

[Blob 与 BlobParser 代替文档加载器](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/langchain-rag-application-development/blob-and-blobparser.md)

[文档转换器与字符分割器组件的使用](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/langchain-rag-application-development/doc-converter-character-splitter-component.md)

## Embedding嵌入模型

[OpenAI Embedding 嵌入模型](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/embedding-model/openai-embedding.md)

[CacheBackEmbedding 组件的使用](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/embedding-model/CacheBackEmbedding.md)

[其他Embedding嵌入模型](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/embedding-model/other-embedding.md)

## 文本分割器

[递归字符文本分割器的使用与运行流程](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/text-splitter/recursive-text-splitter.md)

[语义文档分割器与其他文档分割器的使用](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/text-splitter/semantic-text-splitter.md)

[自定义 LangChain 文档分割器技巧](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/text-splitter/custom-text-splitter.md)

[非分割类型的文档转换器使用技巧](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/text-splitter/non%E2%80%91chunking-doc-transformer.md)

## RAG优化策略

[使用检索器逻辑路由缩减检索范围](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-strategies/LR‑Retriever.md)
