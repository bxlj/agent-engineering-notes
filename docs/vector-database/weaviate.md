# Weaviate 向量数据库的配置与使用

## 01. Weaviate 介绍

Weaviate 是完全使用 Go 语言构建的开源向量数据库，提供了强大的数据存储和检索功能。并且 Weaviate 提供了多种部署方式，以满足不同用户和用例的需求，部署方式如下：

1. **Weaviate 云**：使用 Weaviate 官方提供的云服务，支持数据复制、零停机更新、无缝扩容等功能，适用于评估、开发和生产场景。
2. **Docker 部署**：使用 Docker 容器部署 Weaviate 向量数据库，适用于评估和开发等场景。
3. **K8s 部署**：在 K8s 上部署 Weaviate 向量数据库，适用于开发和生产场景。
4. **嵌入式 Weaviate**：基于本地文件的方式构建 Weaviate 向量数据库，适用于评估场景，不过嵌入式 Weaviate 只适用于 Linux、macOS 系统，在 Windows 下不支持。

Weaviate 和 Pinecone/TCVectorDB 一样，也存在着集合的概念，在 Weaviate 中集合类似传统关系型数据库中的表，负责管理一类数据 / 数据对象，要使用 Weaviate 的流程其实也非常简单：

1. 创建部署 Weaviate 数据库（使用 Weaviate 云、Docker 部署）。
2. 安装 Python 客户端 / LangChain 集成包。
3. 连接 Weaviate（本地连接、云端连接）。
4. 创建数据集 / 集合（代码创建、可视化管理界面创建），在 Weaviate 中，集合的名字必须以大写字母开头，并且只能包含字母、数字和下划线，否则创建的时候会出错，和 Python 的类名规范几乎一致。
5. 添加数据 / 向量。
6. 相似性搜索 / 带过滤器的相似性搜索。

**参考资料：**

1. Weaviate 官网：https://weaviate.io/
2. Weaviate 快速上手指南：https://weaviate.io/developers/weaviate/quickstart
3. LangChain Weaviate 集成包翻译文档：https://imooc-langchain.shortvar.com/docs/integrations/vectorstores/weaviate

## 02. Weaviate 云服务

Weaviate 官方为所有注册登录的账号提供了无限量的 Weaviate 云服务（免费账号每个实例使用时间最大为 14 天，付费账户不限），通过邮箱注册登录 Weaviate 后，找到后台管理系统的 `Clusters`(集群) 即可快速创建 Weaviate 向量数据库实例。

Weaviate 后台管理面板：https://console.weaviate.cloud/dashboard

创建好 Weaviate 云服务器集群后，平台提供了 REST 和 gRPC 两种链接方式的地址与 API 秘钥，在客户端中即可快速连接到云服务。

## 03. Docker 部署 Weaviate 向量数据库

在 Docker 上部署 Weaviate 向量数据库非常简单，如果使用默认值，则不需要 `docker‑compose.yml` 文件来运行镜像（适用于开发场景），安装好 Docker 之后，执行如下命令：

```
docker run -d --name weaviate-dev -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.24.20
```

上述的命令就会快速创建一个叫 `weaviate‑dev` 的容器并在后台运行，该容器暴露了两个端口，8080 和 50051，其中 8080 端口为 REST 接口连接端口、50051 为 gRPC 服务连接端口。

除此之外，使用 Docker 部署的 Weaviate 向量数据库服务，还有以下几个常见命令：

```
# 启动 weaviate 服务
docker start weaviate

# 关闭 weaviate 服务
docker stop weaviate

# 移除 weaviate 容器
docker rm weaviate-dev

# 查看当前 docker 容器所有镜像
docker images

# 移除 weaviate 镜像
docker rmi cr.weaviate.io/semitechnologies/weaviate

# 查看当前运行的 docker 服务
docker ps

# 查看所有 docker 容器（涵盖启动和未启动）
docker ps -a
```

## 04. Weaviate 向量数据库使用技巧

创建好 Weaviate 数据库服务后，接下来就可以安装 Python 客户端 / LangChain 集成包，命令如下：

```
pip install -Uqq langchain-weaviate
```

下一步如果使用的是 Weaviate 云服务，可以直接从可视化界面创建 `collection`，亦或者在使用时 LangChain 自动检测对应的数据集是否存在，如果不存在则直接创建。

然后就可以考虑连接 Weaviate 服务了，Weaviate 框架针对不同的部署方式提供的不同的连接方法：

1. `weaviate.connect_to_local()`：连接到本地的部署服务，需配置连接 URL、端口号。
2. `weaviate.connect_to_wcs()`：连接到远程的 Weaviate 服务，需配置连接 URL、连接秘钥。

示例如下：

```
import weaviate

# 连接192.168.2.120:8080并创建weaviate客户端
client = weaviate.connect_to_local("192.168.2.120", "8080")
```

连接到远程的 Weaviate 服务代码如下：

```
import weaviate
from weaviate.auth import AuthApiKey

client = weaviate.connect_to_wcs(
    cluster_url="https://2j9jgyhprd2yej3c3rwnog.c0.us‑west3.gcp.weaviate.cloud",
    auth_credentials=AuthApiKey("BAn9bGZdZbdGCMUyfdegQoKFctyMmxQdDFb")
)
```

创建好客户端后，接下来可以基于客户端创建 LangChain 向量数据库实例，在实例化 LangChain VectorDB 时，需要传递 `client`（客户端）、`index_name`（集合名字）、`text`（原始文本的存储键）、`embedding`（文本嵌入模型），如下：

```
import dotenv
import weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore

dotenv.load_dotenv()

# 1.连接weaviate向量数据库
client = weaviate.connect_to_local("192.168.2.120", "8080")

# 2.实例化WeaviateVectorStore
embedding = OpenAIEmbeddings(model="text‑embedding‑3‑small")
db = WeaviateVectorStore(client=client, index_name="DatasetTest", text_key="text", embedding=embedding)
```

实例化 LangChain VectorDB 后，就可以像 Faiss、Pinecone、TCVectorDB 一样去使用了，例如执行新增数据后完成检索示例如下：

```
import dotenv
import weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore

dotenv.load_dotenv()

# 1.连接weaviate向量数据库
client = weaviate.connect_to_local("192.168.2.120", "8080")

# 2.实例化WeaviateVectorStore
embedding = OpenAIEmbeddings(model="text‑embedding‑3‑small")
db = WeaviateVectorStore(client=client, index_name="dataset‑test", text_key="text", embedding=embedding)

# 3.新增数据
ids = db.add_texts([
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
])

# 4.检索数据
print(db.similarity_search_with_score("笨笨"))
```

**输出内容：**

```
[(Document(page_content='笨笨是一只很喜欢睡觉的猫咪'), 0.699999988079071),
(Document(page_content='猫咪在窗台上打盹，看起来非常可爱。'), 0.209039822208023),
(Document(page_content='我的狗喜欢追逐球，看起来非常开心。'), 0.19787956774234772),
(Document(page_content='我的手机突然关机了，让我有些焦虑。'), 0.114359922707084)]
```

在 Weaviate 中，也支持带过滤器的相似性筛选，并且 LangChain Weaviate 社区包并没有对筛选过滤器进行二次封装，所以直接传递原生的 weaviate 过滤器即可，参考文档：https://weaviate.io/developers/weaviate/search/filters

例如需要检索 `page` 属性大于等于 5 的所有数据，可以构建一个 filters 后传递给检索方法，如下：

```
from weaviate.classes.query import Filter

filters = Filter.by_property("page").greater_or_equal(5)
print(db.similarity_search_with_score("笨笨", filters=filters))
```

**输出结果如下：**

```
[(Document(page_content='我的狗喜欢追逐球，看起来非常开心。', metadata={'page': 10.0,
'account_id': None}), 0.699999988079071), (Document(page_content='我的手机突然关机
了，让我有些焦虑。', metadata={'page': 7.0, 'account_id': None}),
0.4045487940311432), (Document(page_content='昨晚我做了一个奇怪的梦，梦见自己在太空飞
行。', metadata={'page': 6.0, 'account_id': 1.0}), 0.318904846906662), (Document(page_content='我最喜欢的食物是意大利面，尤其是番茄酱的那种。', metadata=
{'page': 5.0, 'account_id': None}), 0.2671944797039032)]
```

如果想获取 Weaviate 原始集合的实例，可以通过 `db._collection` 快速获得，从而去执行一些原始操作，例如：

```python
from weaviate.classes.query import MetadataQuery

collection = db._collection
response = reviews.query.near_text(
    query="a sweet German white wine",
    limit=2,
    target_vector="title_country",  # Specify the target vector for named vector collections
    return_metadata=MetadataQuery(distance=True)
)

for o in response.objects:
    print(o.properties)
    print(o.metadata.distance)
```