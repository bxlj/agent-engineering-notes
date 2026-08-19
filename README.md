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

## 向量数据库

[Faiss 向量数据库的配置与使用](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/vector-database/faiss.md)

[Pinecone 向量数据库的配置与使用](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/vector-database/pinecone.md)

[TCVectorDB 向量数据库的配置与使用](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/vector-database/TCVectorDB.md)

[Weaviate 向量数据库的配置与使用](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/vector-database/weaviate.md)

[对接自定义向量数据库的配置与使用](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/vector-database/custom_vector_store.md)

[VectorStore 组件深入学习与检索方法](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/vector-database/vectorstore.md)

## RAG优化策略

[使用检索器逻辑路由缩减检索范围](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-strategies/LR_Retriever.md)

[使用语义路由选择不同的 Prompt 模板](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-strategies/srpt_prompt_routing.md)

[自查询检索器实现动态元数据过滤](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-strategies/self_query_retriever_dynamic_metadata_filter.md)

[MultiVector 实现多向量检索检索文档](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-strategies/multi_vector_retrieval.md)

[父文档检索器实现拆分和存储平衡](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-strategies/parent_doc_retriever_splitting_storage_balance.md)

[递归文档树检索实施高级 RAG 优化理解](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-strategies/raptor_recursive_tree_rag_optimization.md)

[ReRank 重排序提升 RAG 系统效果](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-strategies/rag_rerank_reordering.md)

[纠正性索引增强生成 CRAG 优化策略](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-strategies/crag_optimization_strategy.md)

## 大语言模型函数调用与Agent开发

[函数调用快速提取结构化数据使用技巧](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/llm-function-call-agent-dev/function‑call‑struct‑extract.md)

[函数调用出错处理提升程序健壮性](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/llm-function-call-agent-dev/function-call-error-handling.md)

[多模态 LLM 执行函数调用的技巧](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/llm-function-call-agent-dev/multimodal-llm-function-call.md)

[基于 ReACT 架构的 Agent 智能体设计与实现](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/llm-function-call-agent-dev/ReACT_Agent_Design_And_Implementation.md)

[基于工具调用的智能体设计与实现](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/llm-function-call-agent-dev/tool_calling_agent_design.md)

## LangChain RAG 开发多阶段优化

[RAG 开发 6 个阶段优化策略分析](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-and-knowledge-base/rag‑six‑stages‑optimization.md)

[多查询重写策略提升检索准确性](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-and-knowledge-base/multi‑query‑rewrite‑retrieval‑accuracy.md)

[RAG 多查询结果融合策略](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-and-knowledge-base/rag‑multi‑query‑fusion‑rrf.md)

[问题分解策略提升复杂问题检索正确率](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-and-knowledge-base/rag‑question‑decomposition‑strategy.md)

[Step‑Back 回答回退策略扩大检索范围](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-and-knowledge-base/step‑back‑retrieval‑strategy.md)

[多检索器集成多种算法实现混合检索](https://github.com/bxlj/agent-engineering-notes/blob/main/docs/rag-optimization-and-knowledge-base/ensemble-retriever-hybrid-search.md)
