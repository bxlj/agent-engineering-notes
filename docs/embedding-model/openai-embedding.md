# OpenAI Embedding 接口使用实践测试

## 01. OpenAI Embedding 嵌入模型

OpenAI 服务提供商提供了线上的 Embeddings 服务，涵盖了 3 种模型，信息对比如下：

|          模型          | 1 美元支持嵌入的文档数每个文档 800 个 Token | 性能评估 | 最大输入 |     向量维度     |
| :--------------------: | :-----------------------------------------: | :------: | :------: | :--------------: |
| text‑embedding‑3‑small |                   62,500                    |  62.3%   |   8191   | 1536（支持修改） |
| text‑embedding‑3‑large |                    9,615                    |  64.6%   |   8191   | 3072（支持修改） |
| text‑embedding‑ada‑002 |                   12,500                    |  61.0%   |   8191   |       1536       |

OpenAI 的 Embeddings 嵌入模型虽然是市面上效果最好的嵌入模型，虽然价格非常低廉，不过由于没有本地版本，而且接口响应速度相对较慢，在需要对大量文档进行嵌入时，效率非常低，所以国内使用得并不多（国内一般会使用对应的本地或者本地服务提供商的嵌入模型）。

OpenAI Embeddings 官网文档：[https://platform.openai.com/docs/guides/embeddings](https://link.wtturl.cn/?target=https%3A%2F%2Fplatform.openai.com%2Fdocs%2Fguides%2Fembeddings&scene=im&aid=497858&lang=zh)

## 02.Embedding 组件使用

在 LangChain 中，Embeddings 类是一个专为与文本嵌入模型交互而设计的类，这个类为许多嵌入模型提供商（如 OpenAI、Cohere、Hugging Face 等）提供一个标准的接口。

LangChain 中 Embeddings 类提供了两种方法：

1. `embed_documents`：用于嵌入文档列表，传入一个文档列表，得到这个文档列表对应的向量列表。
2. `embed_query`：用于嵌入单个查询，传入一个字符串，得到这个字符串对应的向量。

并且 Embeddings 类并不是一个 Runnable 组件，所以并不能直接接入到 Runnable 序列链中，需要额外的 RunnableLambda 函数进行转换，核心基类代码也非常简单：

```python
class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        return await run_in_executor(None, self.embed_documents, texts)
    
    async def aembed_query(self, text: str) -> List[float]:
       """Asynchronous Embed query text."""
       return await run_in_executor(None, self.embed_query, text)
```

如果想对接自定义的 Embedding 嵌入模型，其实非常简单，只需实现 `embed_documents` 和 `embed_query` 这两个方法即可。

LangChain 使用 OpenAI Embeddings 嵌入模型示例：

```python
import dotenv
import numpy as np
from langchain_openai import OpenAIEmbeddings
from numpy.linalg import norm

dotenv.load_dotenv()


def cosine_similarity(vector1: list, vector2: list) -> float:
    """计算传入两个向量的余弦相似度"""
    # 1.计算内积/点积
    dot_product = np.dot(vector1, vector2)

    # 2.计算向量的范数/长度
    norm_vec1 = norm(vector1)
    norm_vec2 = norm(vector2)

    # 3.计算余弦相似度
    return dot_product / (norm_vec1 * norm_vec2)


embeddings = OpenAIEmbeddings()

query_vector = embeddings.embed_query("你好，我是慕小课，我喜欢打篮球")
documents_vector = embeddings.embed_documents([
    "你好，我是慕小课，我喜欢打篮球",
    "这个喜欢打篮球的人叫慕小课",
    "求知若渴，虚心若愚"
])

print(query_vector)
print(len(query_vector))

print("============")

print(len(documents_vector))
print("vector1与vector2的余弦相似度:", cosine_similarity(documents_vector[0], documents_vector[1]))
print("vector2与vector3的余弦相似度:", cosine_similarity(documents_vector[0], documents_vector[2]))
```

