# 其他Embedding嵌入模型

## 01. Hugging Face 文本嵌入模型

### 1.1 Hugging Face 本地模型

在某些对数据保密要求极高的场合下，数据不允许传递到外网，这个时候就可以考虑使用本地的文本嵌入模型 ——Hugging Face 本地嵌入模型，配置的技巧也非常简单，安装 `langchain‑huggingface` 与 `sentence‑transformers` 包，命令如下：

```
pip install -U langchain-huggingface sentence-transformers
```

其中 `langchain‑huggingface` 是 langchain 团队基于 huggingface 封装的第三方社区包，`sentence‑transformers` 是一个用于生成和使用预训练的文本嵌入，基于 transformer 架构，也是目前使用量最大的本地文本嵌入模型。

配置好后，就可以像正常的文本嵌入模型一样使用了，示例：

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

query_vector = embeddings.embed_query("你好，我是张强，我喜欢打篮球")

print(query_vector)
print(len(query_vector))
```

`sentence‑transformers` 除此加载的时候，会将模型从本地应用加载到内存中，首次加载相对会慢，后续调用速度即可恢复正常。

除此之外，使用 Hugging Face 文本嵌入还可以加载任意本地的文本嵌入模型，传递 `model_name` 与 `cache_folder` 参数即可，示例：

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="neuml/pubmedbert-base-embeddings",
    cache_folder="./embeddings/"
)

query_vector = embeddings.embed_query("你好，我是张强，我喜欢打篮球")

print(query_vector)
print(len(query_vector))
```

### 1.2 HuggingFace 远程嵌入模型

部分模型的文件比较大，如果只是短期内调试，可以考虑使用 HuggingFace 提供的远程嵌入模型，首先安装对应的依赖：

```
pip install huggingface_hub
```

然后在 Hugging Face 官网（[https://huggingface.co/](https://link.wtturl.cn/?target=https%3A%2F%2Fhuggingface.co%2F&scene=im&aid=497858&lang=zh)）的 setting 中添加对应的访问秘钥，并配置到 `.env` 文件中：

```
HUGGINGFACEHUB_API_TOKEN=xxx
```

接下来就可以使用 Hugging Face 提供的推理服务，这样在本地服务器上就无需配置对应的文本嵌入模型了。

```python
import dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

dotenv.load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L12-v2")

query_vector = embeddings.embed_query("你好，我是张强，我喜欢打篮球")

print(query_vector)
print(len(query_vector))
```

## 02. 百度千帆文本嵌入模型

如果想对接国内的文本嵌入模型提供商，可以考虑百度千帆，是目前国内生态最好，支持的模型最多（Embedding‑V1、bge‑large‑zh、bge‑large‑en、tao‑8k），速度最快的 AI 应用开发平台。

由于目前百度千帆并没有单独封装到独立的包，可以直接从 `langchain_community` 中导入，使用示例如下：

```python
import dotenv
from langchain_community.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint

dotenv.load_dotenv()

embeddings = QianfanEmbeddingsEndpoint()

query_vector = embeddings.embed_query("我叫张强，我喜欢打篮球游泳")

print(query_vector)
print(len(query_vector))
```

相关资料信息：

1. 百度千帆预设模型列表：https://cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu
2. 百度千帆嵌入文档：https://python.langchain.com/v0.2/docs/integrations/text_embedding/baidu_qianfan_endpoint/
3. 百度千帆嵌入翻译文档：http://imooc-langchain.shortvar.com/docs/integrations/text_embedding/baidu_qianfan_endpoint/