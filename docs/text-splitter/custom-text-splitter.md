# 自定义 LangChain 文档分割器技巧

## 01. 自定义文档分割器

在 LangChain 中，如果内置的文档分割器均没办法完成需求，还可以根据特定的需求实现自定义文档分割器（一般极少），实现的方法也非常简单，继承文本分割器基类 `TextSplitter`，在构造函数中传递相关参数，然后实现 `split_text()` 方法即可。

例如，实现一个根据传递的分隔符实现对文档进行片段划分，并且将分割出来的文档片段转换成 N 个关键词的分割器，代码示例：

```python
from typing import List

import jieba.analyse
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import TextSplitter


class CustomTextSplitter(TextSplitter):
    """自定义文本分割器"""

    def __init__(self, seperator: str, top_k: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)
        self._seperator = seperator
        self._top_k = top_k

    def split_text(self, text: str) -> List[str]:
        """分割传入的文本为字符串列表"""
        split_texts = text.split(self._seperator)
        text_keywords = []
        for split_text in split_texts:
            text_keywords.append(
                jieba.analyse.extract_tags(split_text, self._top_k)
            )

        return [",".join(keywords) for keywords in text_keywords]


# 1.创建加载器与分割器
loader = UnstructuredFileLoader("./科幻短篇.txt")
text_splitter = CustomTextSplitter("\n\n")

# 2.加载文档并分割
documents = loader.load()
chunks = text_splitter.split_documents(documents)

# 3.循环遍历文档信息
for chunk in chunks:
    print(f"文档库块内容:{chunk.page_content}")
```

生成内容：

```
文档库块内容:星际,穿越,标题
文档库块内容:概要
文档库块内容:星系,李昂,文明,这个,高度发达,宇航员,星际,灭绝,一个,穿越
文档库块内容:迷航,星际,第一章
文档库块内容:飞船,李昂,观察窗,浩瀚,星空,宇航员,经验丰富,星系,与众不同,穿越
文档库块内容:李昂,星空,睁开眼,眩晕,飞船,星系,扭曲,闪过,发现自己,陌生
文档库块内容:异星,第二章,文明
文档库块内容:李昂,文明,行星,一颗,高度发达,小行星,星球,灭绝,星系,这颗
文档库块内容:赛跑,第三章,时间
文档库块内容:寻找,李昂,科技知识,小行星,飞船,星系,撞击,紧迫,自己,阻止
文档库块内容:火花,第四章,智慧
文档库块内容:能量,李昂,场来,计划,小行星,解决方案,撞击,这个,轨道,科学家
文档库块内容:第五章,危机,希望
文档库块内容:星球,矿物,能量,李昂,夜以继日,小行星,蕴藏,罕见,深处,科学家
文档库块内容:第六章,星际,救援
文档库块内容:小行星,成功,矿物,能量,李昂,撞击,带回,采集,勇敢,实验室
文档库块内容:归途,第七章
文档库块内容:李昂,遗迹,热烈欢迎,星系,隐藏,感激,解除,通往,这个,探索
文档库块内容:第八章,重逢,星际
文档库块内容:李昂,归属感,归途,星系,穿越,踏上,冒险,告别,前所未有,平静
文档库块内容:第九章,开始
文档库块内容:李昂,星球,星际,之间,一个,开启,激励,全新,无数,英雄
文档库块内容:第十章,星际,使者
文档库块内容:文明,李昂,旅程,使者,美好,宇宙,探索,交流,挑战,面对
文档库块内容:慕课
文档库块内容:梦工厂,程序员
```

## 02. RAG 文档分割 / 分块总结

在前面的文章中，我们学习过字符文本分割器、递归字符文本分割器、Html 标题 / 段分割器、语义分割器等多种文本分割器类型，这也是目前 RAG 分块 Chunk 的 4 种策略：

1. **固定大小分块**：这是最常见的分块方法，通过设定块的大小和是否有重叠来决定分块。这种方法简单直接，不需要使用任何 NLP 库，因此计算成本低且易于使用，例如 `CharacterTextSplitter`，亦或者直接循环遍历固定大小拆分。
2. **基于结构的分块**：常见的 HTML、MARKDOWN 格式，或者其他可以有明确结构格式的文档。这种可以借助 “结构感知” 对文档分块，充分利用文档文本意外的信息，类似 LangChain 中的 `HTMLHeaderTextSplitter` 等。
3. **基于语义的分块**：这种策略旨在确保每个分块包含尽可能多的语义独立信息。可以采用不同的方法，如标点符号、自然段落、或者 NLTK、Spicy 等工具包来实现语义分块，或者 Embedding‑based 方法，例如 LangChain 的 `SemanticChunker` 等。
4. **递归分块**：递归分块使用一组分隔符，以分层和迭代的方式将输入文本划分为更小的块。如果最初分割文本没有产生所需大小或结构的块，则该方法会继续递归地分割直到满足条件，例如 LangChain 中的 `RecursiveCharacterTextSplitter` 等。

这些策略各有优势和适用场景，选择合适的分块策略取决于具体的应用需求和数据特性。很遗憾，到目前为止还没有什么是最优的策略，但这也是很难有一个产品一统天下的原因。同时策略可以组合使用，并不是一类文档只能用一种策略。

> [!IMPORTANT]
>
> 对于一个 RAG 场景，分成四个主要阶段：预检索、检索、后检索 和 生成，其中 分块 是 预检索阶段的策略，如果在 分块 阶段尝试了上述 4 种策略均没有很好的效果，或许就不应该采用 RAG 的策略，而是使用 微调 的方式，让这部分知识成为模型永久的记忆，效果可能会更好！