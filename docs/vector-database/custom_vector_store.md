# 对接自定义向量数据库的配置与使用

## 01. 对接自定义向量数据库

向量数据库的发展非常迅猛，几乎间隔几天就有新的向量数据库发布，LangChain 不可能将所有向量数据库都进行集成，亦或者封装的包存在这一些 bug 或错误，这个时候就需要考虑创建自定义向量数据库，去实现特定的方法。

在 LangChain 实现自定义向量数据库的类有两种模式，一种是继承封装好的数据库类，一种是继承基类`VectorStore`。前一种一般继承后重写部分方法进行扩展或者修复 bug，后面一种是对接新的向量数据库。

在 LangChain 中，继承 `VectorStore` 只需实现最基础的 3 个方法即可正常使用：

1. **add_texts**：将对应的数据添加到向量数据库中。
2. **similarity_search**：最基础的相似性搜索。
3. **from_texts**：从特定的文本列表、元数据列表中构建向量数据库。

其他方法因为使用频率并不高，`VectorStore` 并没有设置成虚拟方法，但是再没有实现的情况下，直接调用会报错，涵盖：

1. `delete()`：删除向量数据库中的数据。
2. `_select_relevance_score_fn()`：根据距离计算相似性得分函数。
3. `similarity_search_with_score()`：携带得分的相似性搜索函数。
4. `similarity_search_by_vector()`：传递向量进行相似性搜索。
5. `max_marginal_relevance_search()`：最大边界相似性搜索。
6. `max_marginal_relevance_search_by_vector()`：传递向量进行最大边界相关性搜索。

## 02. 自定义 VectorStore 示例

要在 LangChain 中对接自定义向量数据，本质上就是将向量数据库提供的方法集成到 `add_texts`、`similarity_search`、`from_texts` 方法下，例如自建一个基于内存 + 欧几里得距离的 “向量数据库”，示例如下：

```python
import uuid
from typing import List, Optional, Any, Iterable, Type

import dotenv
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings


class MemoryVectorStore(VectorStore):
    """自定义向量数据库"""
    store: dict = {}  # 在内存中开辟位置存储向量

    def __init__(self, embedding: Embeddings, **kwargs):
        self._embedding = embedding

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[str]:
        """将数据添加到内存向量数据库中"""
        # 1.判断metadatas和texts的长度是否保持一致
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("元数据格式必须和文本数据保持一致")

        # 2.将文本转换为向量
        embeddings = self._embedding.embed_documents(texts)

        # 3.生成uuid
        ids = [str(uuid.uuid4()) for text in texts]

        # 4.将原始文本、向量、元数据、id构建字典并存储
        for idx, text in enumerate(texts):
            self.store[ids[idx]] = {
                "id": ids[idx],
                "vector": embeddings[idx],
                "text": text,
                "metadata": metadatas[idx] if metadatas is not None else {}
            }
        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """执行相似性搜索"""
        # 1.将query转换成向量
        embedding = self._embedding.embed_query(query)

        # 2.循环遍历记忆存储，计算欧几里得距离
        result: list = []
        for key, record in self.store.items():
            distance = self._euclidean_distance(embedding, record["vector"])
            result.append({
                "distance": distance,
                **record,
            })

        # 3.找到欧几里得距离最小的k条记录
        sorted_result = sorted(result, key=lambda x: x["distance"])
        result_k = sorted_result[:k]

        # 4.循环构建文档列表并返回
        documents = [
            Document(page_content=item["text"], metadata={**item["metadata"], "score": item["distance"]})
            for item in result_k
        ]
        return documents

    @classmethod
    def from_texts(cls: Type["MemoryVectorStore"], texts: List[str], embedding: Embeddings,
                   metadatas: Optional[List[dict]] = None,
                   **kwargs: Any) -> "MemoryVectorStore":
        """通过文本、嵌入模型、元数据构建向量数据库"""
        memory_vector_store = cls(embedding=embedding, **kwargs)
        memory_vector_store.add_texts(texts, metadatas)
        return memory_vector_store

    @classmethod
    def _euclidean_distance(cls, vec1, vec2) -> float:
        """计算两个向量的欧几里得距离"""
        return np.linalg.norm(np.array(vec1) - np.array(vec2))


dotenv.load_dotenv()

# 1.创建初始数据与嵌入模型
texts = [
    "笨笨是一只很喜欢睡觉的猫咪",
    "我喜欢在夜晚听音乐，这让我感到放松。",
    "猫咪在窗台上打盹，看起来非常可爱。",
    "学习新技能是每个人都应该追求的目标。",
    "我最喜欢的食物是意大利面，尤其是番茄酱的那种。",
    "昨晚我做了一个奇怪的梦，梦见自己在太空飞行。",
    "我的手机突然关机了，让我有些焦虑。",
    "阅读是我每天都会做的事情，我觉得很充实。",
    "他们一起计划了一次周末的野餐，希望天气能好。",
    "我的狗喜欢追逐球，看起来非常开心。",
]

metadatas = [
    {"page": 1},
    {"page": 2},
    {"page": 3},
    {"page": 4},
    {"page": 5},
    {"page": 6, "account_id": 1},
    {"page": 7},
    {"page": 8},
    {"page": 9},
    {"page": 10},
]

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 2.构建自定义向量数据库
db = MemoryVectorStore.from_texts(texts, embedding, metadatas)

# 3.执行检索
print(db.similarity_search("我养了一只猫，叫笨笨"))
```