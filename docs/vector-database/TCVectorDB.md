# TCVectorDB 向量数据库的配置与使用

## 01. TCVectorDB 向量数据库简介

虽然目前绝大部分开源向量数据库都是海外的，配置起来也非常简单，性能也很高，但是因为网络的原因，如果将向量数据库部署到海外，而产品面向国内，网络延迟是一个必要考虑的问题，所以考虑国内服务提供商的云向量数据库往往是最佳的选择。

腾讯云向量数据库（TCVectorDB）是一款全托管的自研企业级分布式数据库服务，专用于存储、检索、分析多维向量数据，该数据库支持多种索引类型和相似度计算方法，索引支持千亿级向量规模，可支持百万级 QPS 及毫秒级查询延迟。而且目前认证过的腾讯云账号可以免费使用 TCVectorDB 一个月时间，相关文档：

1. TCVectorDB 产品链接：https://cloud.tencent.com/product/vdb
2. TCVectorDB 产品文档：https://cloud.tencent.com/document/product/1709
3. LangChain TCVectorDB 翻译文档：http://imooc-langchain.shortvar.com/docs/integrations/vectorstores/tencentvectordb/

TCVectorDB 和 Pinecone 的设计理念非常接近，在腾讯云向量数据库中也有对应的**数据库、集合、记录**等概念：

1. **数据库**：分为普通向量数据库和 AI 数据库，其中 AI 数据库无需外部配置文本分割、Embedding、文档解析等功能，底层全部由腾讯云实现，而普通向量数据库则需要外部程序处理，它只能接收传递的数据，并没有处理功能，但是功能更加可定制化。
2. **集合**：集合是数据库的下一个单位，类似与传统数据库中的**表**，在集合中，需要设置集合名称、分片数、索引等信息。
3. **记录**：集合的每一条数据就是记录。

TCVectorDB 默认只能在内网中链接使用，在生产环境中，也尽可能不将数据库暴露到外网中，但是在开发中，则需要配置并开启外网访问功能，配置好外网访问功能、API 秘钥后，需要在项目中导入对应的环境变量。

### 环境变量

```
TC_VECTOR_DB_URL=xxx
TC_VECTOR_DB_USERNAME=root
TC_VECTOR_DB_KEY=xxx
TC_VECTOR_DB_DATABASE=llmops
TC_VECTOR_DB_TIMEOUT=30
```

接下来安装对应的 Python 包，命令如下：

```
pip install tcvectordb
```

接下来就可以像 Faiss、Pinecone 一样正常使用即可，整体功能和 Pinecone 几乎一模一样，只是`filter`、`namespace`等概念的操作有些许差异。

## 02. TCVectorDB 向量数据库使用技巧

### 2.1 内置 Embedding 导入数据与查询

TCVectorDB 向量数据库内置了 Embedding 模型，并支持 5 种嵌入方式，如下：

| 模型名                 | 适用语言类型 | 维度 | 最大 Token 数量 | 分类得分 | 聚类得分 | 检索得分 |
| ---------------------- | ------------ | ---- | --------------- | -------- | -------- | -------- |
| bge-base‑zh（推荐）    | 中文         | 768  | 512             | 67.06    | 47.64    | 69.53    |
| m3e‑base               | 中文         | 768  | 512             | 67.52    | 47.68    | 56.91    |
| text2vec‑large‑chinese | 中文         | 1024 | 512             | 60.66    | 30.02    | 41.94    |
| e5‑large‑v2            | 英文         | 1024 | 512             | 75.24    | 44.49    | 50.56    |
| multilingual‑e5‑base   | 多语言       | 768  | 514             | 63.35    | 40.68    | 40.68    |

使用内置的 Embedding 模型速度会更快（而且目前 LangChain 封装的 TCVectorDB 使用内置 Embedding 没有 bug），不过限制也比较多，迁移数据库时，所有的文本全部需要重新生成。

内置 Embedding 模型文档说明 & 计费：https://cloud.tencent.com/document/product/1709/98014

**示例代码**

```python
import os
import dotenv
from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import (
    ConnectionParams,
)
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()

embedding = OpenAIEmbeddings(model="text‑embedding‑3‑small")
db = TencentVectorDB(
    embedding=None,
    connection_params=ConnectionParams(
        url=os.environ.get("TC_VECTOR_DB_URL"),
        username=os.environ.get("TC_VECTOR_DB_USERNAME"),
        key=os.environ.get("TC_VECTOR_DB_KEY"),
        timeout=int(os.environ.get("TC_VECTOR_DB_TIMEOUT")),
    ),
    database_name=os.environ.get("TC_VECTOR_DB_DATABASE"),
    collection_name="dataset‑builtin",
)

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

ids = db.add_texts(texts)
print("添加文档id列表:", ids)
print(db.similarity_search_with_score("我养了一只猫，叫笨笨"))
```

**输出示例**

```
添加文档id列表: ['1721204162283324200‑8452797147690134545‑0', '1721204162283324200‑4748609318240602995‑1', '1721204162283324200‑5687737768520280782‑2', '1721204162283324200‑‑327905253905429573‑3', '1721204162283324200‑‑4217514370195306218‑4', '1721204162283324200‑5510932568003309936‑5', '1721204162283324200‑7945499047074485108‑6', '1721204162283324200‑8077162730065074156‑7', '1721204162283324200‑36196153077512649‑8', '1721204162283324200‑3472252428185052217‑9']

[(Document(page_content='笨笨是一只很喜欢睡觉的猫咪'), 0.874507),
(Document(page_content='我的狗喜欢追逐球，看起来非常开心。'), 0.757709),
(Document(page_content='猫咪在窗台上打盹，看起来非常可爱。'), 0.748665),
(Document(page_content='昨晚我做了一个奇怪的梦，梦见自己在太空飞行。'), 0.735823)]
```

### 2.2 外部 Embedding 导入数据与查询

除了使用内置的 Embedding 模型，TCVectorDB 也支持传递自定义的文本嵌入模型，但是在 LangChain 中封装的 TencentVectorDB 存在一个 bug，当传递外部 Embedding 模型时，必须配置`meta_fields`属性，并且指定字段名称为`text`，然后将 metadatas 中的数据添加上`text`这个字段，向量数据库才会记录原文信息。

`meta_fields`的字段是告知 TCVectorDB 需要存储 metadata 中的哪些字段，如果没有设置，metadata 中的所有字段均不会被保存。

如果没有记录原文信息，在进行检索操作时，会因为找不到原文，从而没法创建 Document 文档，引发报错。

**出现 bug 的源码片段**

```
# langchain_community/vectorstores/tencentvectordb.py->TencentVectorDB::add_texts
def add_texts(
    self,
    texts: Iterable[str],
    metadatas: Optional[List[dict]] = None,
    timeout: Optional[int] = None,
    batch_size: int = 1000,
    ids: Optional[List[str]] = None,
    **kwargs: Any,
) -> List[str]:
    ...
    if embeddings:
        doc_attrs["vector"] = embeddings[id]
    else:
        doc_attrs["text"] = texts[id]
    ...
```

**修复示例代码**

```python
import os
import dotenv
from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import (
    ConnectionParams,
    MetaField,
    META_FIELD_TYPE_STRING,
)
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()

embedding = OpenAIEmbeddings(model="text‑embedding‑3‑small")
db = TencentVectorDB(
    embedding=embedding,
    connection_params=ConnectionParams(
        url=os.environ.get("TC_VECTOR_DB_URL"),
        username=os.environ.get("TC_VECTOR_DB_USERNAME"),
        key=os.environ.get("TC_VECTOR_DB_KEY"),
        timeout=int(os.environ.get("TC_VECTOR_DB_TIMEOUT")),
    ),
    database_name=os.environ.get("TC_VECTOR_DB_DATABASE"),
    collection_name="dataset‑external",
    meta_fields=[
        # 配置text字段，防止LangChain在检索时，找不到字段构建Document文档，从而触发错误
        MetaField(name="text", data_type=META_FIELD_TYPE_STRING),
    ]
)

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

# 将原始文本数据合并到metadata中，
metadata = [{"text": text, "page": index} for index, text in enumerate(texts)]

ids = db.add_texts(texts, metadata)
print("添加文档id列表:", ids)
print(db.similarity_search("我养了一只猫，叫笨笨"))
```

**输出内容**

```
添加文档id列表: ['1721204290046443100‑‑7979142186803955240‑0',
'1721204290046443100‑‑2991801275249710134‑1', '1721204290046443100‑7229055024046586038‑2', '1721204290046443100‑6820010242590464795‑3',
'1721204290046443100‑1761951776563722534‑4', '1721204290046443100‑7105227574091896456‑5', '1721204290046443100‑385681627517092469‑6',
'1721204290046443100‑2611196434988287836‑7', '1721204290046443100‑5925412479147078471‑8', '1721204290046443100‑‑5880598133876375674‑9']

[Document(page_content='笨笨是一只很喜欢睡觉的猫咪', metadata={'text': '笨笨是一只很喜欢睡觉的猫咪'}), Document(page_content='猫咪在窗台上打盹，看起来非常可爱。', metadata={'text': '猫咪在窗台上打盹，看起来非常可爱。'}), Document(page_content='我的狗喜欢追逐球，看起来非常开心。', metadata={'text': '我的狗喜欢追逐球，看起来非常开心。'}), Document(page_content='我的手机突然关机了，让我有些焦虑。', metadata={'text': '我的手机突然关机了，让我有些焦虑。'})]
```

### 2.3 TCVectorDB 带过滤的相似性搜索

TCVectorDB 也支持为相似性搜索添加过滤器，表达式的格式为 `<field_name><operator><value>`，多个表达式之间支持`and`（与）、`or`（或）、`not`（非）关系。其中：

1. `<field_name>`：表示要过滤的字段名。
2. `<operator>`：要使用的运算符。
   - string：匹配单个字符串值（=）、排除单个字符串值（!=）、匹配任意一个字符串值（in）、排除所有字符串值（not in）。其对应的 Value 必须使用英文双引号括起来。
   - uint64：大于（>）、大于等于（>=）、等于（=）、小于（<）、小于等于（<=）、不等于（!=）。
3. `<value>`：表示要匹配的值。

例如：`uint64_val > 30`、`game_tag="Detective"` 等。

并且在 TCVectorDB 使用非向量字段进行检索时，必须在构建 Collection 时添加上对应的索引，否则无法检索会报错。

详细文档: https://cloud.tencent.com/document/product/1222/98734#2e2e982a‑a487‑4a05‑8410‑0‑58c6bda12180

**代码示例**

```python
import os
import dotenv
from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import (
    ConnectionParams,
    MetaField,
    META_FIELD_TYPE_UINT64,
)
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()

db = TencentVectorDB(
    embedding=None,
    connection_params=ConnectionParams(
        url=os.environ.get("TC_VECTOR_DB_URL"),
        username=os.environ.get("TC_VECTOR_DB_USERNAME"),
        key=os.environ.get("TC_VECTOR_DB_KEY"),
        timeout=int(os.environ.get("TC_VECTOR_DB_TIMEOUT")),
    ),
    database_name=os.environ.get("TC_VECTOR_DB_DATABASE"),
    collection_name="dataset-filter",
    meta_fields=[
        MetaField(name="page", data_type=META_FIELD_TYPE_UINT64),
    ]
)

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

ids = db.add_texts(texts, metadatas)
print("添加文档id列表:", ids)

print(db.similarity_search_with_score("我养了一只猫，叫笨笨", expr="page>=9"))

```

输出内容：

```
添加文档id列表: ['1721205246238323600--5707869341580869917-0',
'1721205246238323600-977715369524907179-1', '1721205246238323600-
6780760367391118320-2', '1721205246238323600-9214408441339088766-3',
'1721205246238323600--5359114623033048472-4', '1721205246238323600-
5350375289397780096-5', '1721205246238323600-1143848279187318822-6',
'1721205246238323600-1555278682065468720-7', '1721205246238323600-
-8454962785458887831-8', '1721205246238323600--7242128883936657791-9']
[(Document(page_content='我的狗喜欢追逐球，看起来非常开心。', metadata={'page': 10}),
0.757709), (Document(page_content='他们一起计划了一次周末的野餐，希望天气能好。',
metadata={'page': 9}), 0.675508)]
```