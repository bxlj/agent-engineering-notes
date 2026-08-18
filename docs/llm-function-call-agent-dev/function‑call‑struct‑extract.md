# 函数调用快速提取结构化数据使用技巧

## 01. 结构化输出策略与选择

让 LLM 返回结构化输出非常重要，这是因为 LLM 应用程序的输出通常用于下游程序，这些应用程序需要特定的参数，目前常见的几种让 LLM 结构化输出的策略有：

1. **Prompt**：通过 prompt 让 LLM 输出特定结构的内容，兼容所有 LLM，但是输出不稳定。
2. **函数 / 工具调用**：让 LLM 绑定函数，并设置选择模式为强制，让 LLM 强制调用函数，从而获取结构化输出数据。
3. **JSON 模式**：对于支持 JSON 模式输出的 LLM，还可以通过设置输出结构为 JSON 模式，从而获取结构化数据。

其中后两种输出模式会更稳定一些，在 LangChain 中为后两种方法封装了`.with_structured_output()`方法，这也是获取结构化输出的最简单和最可靠的方法，在`.with_structured_output()`的底层会使用 LLM 原生的函数调用或 JSON 模式。

该方法接受一个`BaseModel`子类作为输入，该子类需要指定输出属性的名称、描述和类型，该方法返回的是一个 Runnable 可运行对象，但并不是输出字符串或者 AI 消息，而是输出与给定模式对应的对象，其中模式可以指定为 JSON 模式（返回一个字典）或 Pydantic 类（返回一个 Pydantic 对象）。

另外检测一个 LLM 是否支持`.with_structured_output()`可以通过查看源码或者在 LangChain 的大语言模型 高级功能列表中查看，链接：https://imooc-langchain.shortvar.com/docs/integrations/chat/

我们学习的`DocTran`在底层就是使用函数调用来实现结构化输出，如果使用这节课所学的知识来完成一个 QA 问答数据的提取，代码示例如下：

```python
import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


class QAExtra(BaseModel):
    """一个问答键值对工具，传递对应的假设性问题+答案"""
    question: str = Field(description="假设性问题")
    answer: str = Field(description="假设性问题对应的答案")


llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
structured_llm = llm.with_structured_output(QAExtra)

prompt = ChatPromptTemplate.from_messages([
    ("system", "请从用户传递的query中提取出假设性的问题+答案。"),
    ("human", "{query}")
])

chain = {"query": RunnablePassthrough()} | prompt | structured_llm

print(chain.invoke("我叫张三，我喜欢打篮球，游泳"))
```

输出内容：

```
question='你喜欢打篮球吗？' answer='是的，我喜欢打篮球。'
```

在`.with_structured_output()`的底层会优先使用函数调用模式，如果 LLM 支持 JSON 模式，还可以在函数内传递多一个参数`method="json_mode"`，即使用大语言模型的 JSON 输出模式来操作，修改代码如下：

```python
import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


class QAExtra(BaseModel):
    """一个问答键值对工具，传递对应的假设性问题+答案"""
    question: str = Field(description="假设性问题")
    answer: str = Field(description="假设性问题对应的答案")


llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(QAExtra, method="json_mode")

prompt = ChatPromptTemplate.from_messages([
    ("system", "请从用户传递的query中提取出假设性的问题+答案。响应格式为JSON，并携带`question`和`answer`两个字段。"),
    ("human", "{query}")
])

chain = {"query": RunnablePassthrough()} | prompt | structured_llm

print(chain.invoke("我叫张三，我喜欢打篮球，游泳。"))
```

输出内容：

```
question='你叫什么名字？你喜欢做什么？' answer='我叫慕小课，我喜欢打篮球，游泳。'
```

在上面的代码中修改了几个部分：

1. **gpt‑4o 模型**：在 OpenAI 提供的模型中，并不是所有模型都支持 JSON 模式的，需要查看文档确认。
2. **prompt 提示模板**：在 JSON 模式中，需要在 Prompt 中告知 LLM 输出 JSON 的结构信息，要不然会报错。

不过因为支持 JSON 模式的大语言模型较少，所以使用起来限制也比较多，还会对原始的 Prompt 产生干扰，所以尽可能使用函数调用的形式来结构化数据，会更稳定。

> Important 在创建`BaseModel`子类时，类的名称、文档字符串以及参数的名称和提供的描述也非常重要，因为在该函数的底层大部分使用的都是函数调用，只有添加上这些信息后，才可以将对应的 LLM 绑定函数调用，获取正确的结果。

## 02. .with_structured_output () 源码解析

在`.with_structured_output()`底层，会通过传递的`method`来执行不同的运行时参数绑定，例如使用函数调用亦或者是 JSON 模式输出，核心代码：

```python
def with_structured_output(
    self,
    schema: Optional[_DictOrPydanticClass] = None,
    *,
    method: Literal["function_calling", "json_mode"] = "function_calling",
    include_raw: bool = False,
    **kwargs: Any,
) -> Runnable[LanguageModelInput, _DictOrPydantic]:
    if kwargs:
        raise ValueError(f"Received unsupported arguments {kwargs}")
    is_pydantic_schema = _is_pydantic_class(schema)
    if method == "function_calling":
        if schema is None:
            raise ValueError(
                "schema must be specified when method is 'function_calling'. "
                "Received None."
            )
        llm = self.bind_tools([schema], tool_choice="any")
        if is_pydantic_schema:
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[schema], first_tool_only=True
            )
        else:
            key_name = convert_to_openai_tool(schema)["function"]["name"]
            output_parser = JsonOutputKeyToolsParser(
                key_name=key_name, first_tool_only=True
            )

    elif method == "json_mode":
        llm = self.bind(response_format={"type": "json_object"})
        output_parser = (
            PydanticOutputParser(pydantic_object=schema)
            if is_pydantic_schema
            else JsonOutputParser()
        )
    else:
        raise ValueError(
            f"unrecognized method argument. Expected one of 'function_calling' or "
            f"'json_mode'. Received: '{method}'"
        )

    if include_raw:
        parser_assign = RunnablePassthrough.assign(
            parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
        )
        parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
        parser_with_fallback = parser_assign.with_fallbacks(
            [parser_none], exception_key="parsing_error"
        )
        return RunnableMap(raw=llm) | parser_with_fallback
    else:
        return llm | output_parser
```

所以哪怕不使用 LangChain，该思路也非常值得借鉴，在需要使用 LLM 获取结构化数据时，只需要构建一个虚假的函数，让 LLM 绑定该函数，并且设置`tool_choice="any"`即强制调用所有函数，这样就可以在最大程度上确保 LLM 能稳定地输出结构化数据。