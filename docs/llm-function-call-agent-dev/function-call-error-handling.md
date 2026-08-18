# 函数调用出错处理提升程序健壮性

在执行函数调用的过程中，错误几乎是不可避免的，无论是大语言模型错误地传递了参数，还是工具内部抛出的错误，如果错误不做任何处理操作，都会让程序变得异常脆弱，所以我们可以考虑在链中构建错误处理 / 捕获来减轻这些故障出现的概率。

## 01. try/except 捕获工具错误

首先我们先来构建一个复杂工具，并让 LLM 尝试调用这个工具，并故意引发一些错误，示例代码如下：

```python
import dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

@tool
def complex_tool(int_arg: int, float_arg: float, dict_arg: dict) -> int:
    """使用复杂工具进行复杂计算操作"""
    return int_arg * float_arg

# 1.创建大语言模型并绑定工具
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
llm_with_tools = llm.bind_tools([complex_tool])

# 2.创建链并执行工具
chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | complex_tool

# 3.调用链
print(chain.invoke("使用复杂工具，对应参数为5和2.1"))
```

输出内容如下：

```
Traceback (most recent call last):
  File "D:\Project\llmops\llmops-api\study\41-工具调用\1.错误捕获.py", line 29, in <module>
    print(chain.invoke("使用复杂工具，对应参数为5和2.1"))
  File "D:\Project\llmops\llmops-api\venv\lib\site-packages\langchain_core\runnables\base.py", line 2875, in invoke
    input = step.invoke(input, config)
  File "D:\Project\llmops\llmops-api\venv\lib\site-packages\langchain_core\tools.py", line 427, in invoke
    return self.run(tool_input, **kwargs)
  File "D:\Project\llmops\llmops-api\venv\lib\site-packages\langchain_core\tools.py", line 615, in run
    raise error_to_raise
  File "D:\Project\llmops\llmops-api\venv\lib\site-packages\langchain_core\tools.py", line 578, in run
    tool_args, tool_kwargs = self._to_args_and_kwargs(tool_input)
  File "D:\Project\llmops\llmops-api\venv\lib\site-packages\langchain_core\tools.py", line 501, in _to_args_and_kwargs
    tool_input = self._parse_input(tool_input)
  File "D:\Project\llmops\llmops-api\venv\lib\site-packages\langchain_core\tools.py", line 454, in _parse_input
    result = input_args.parse_obj(tool_input)
  File "D:\Project\llmops\llmops-api\venv\lib\site-packages\pydantic\v1\main.py", line 526, in parse_obj
    return cls(**obj)
  File "D:\Project\llmops\llmops-api\venv\lib\site-packages\pydantic\v1\main.py", line 341, in __init__
    raise validation_error
pydantic.v1.error_wrappers.ValidationError: 1 validation error for complex_toolSchema
dict_arg
  field required (type=value_error.missing)
```

在上面的示例中，因为工具的描述并不清晰，大语言模型只针对前 2 个参数生成了对应的数值，第 3 个参数并没有生成，所以引发了 pydantic 数据校验错误，对于这类错误，通用的处理方法是在工具调用步骤上使用 `try/except` 进行错误的捕获，并在出错时返回游泳的提示信息，修正代码部分如下：

```python
def try_except_tool(tool_args: dict, config: RunnableConfig) -> Any:
    try:
        return complex_tool.invoke(tool_args, config)
    except Exception as e:
        return f"调用工具时使用以下参数:\n\n{tool_args}\n\n引发了以下错误:\n\n{type(e)}: {e}"


chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | try_except_tool
```

输出内容：

```
调用工具时使用以下参数：

{'int_arg': 5, 'float_arg': 2.1}

引发了以下错误：

<class 'pydantic.v1.error_wrappers.ValidationError'>: 1 validation error for complex_toolSchema
dict_arg
  field required (type=value_error.missing)
```

在这种模式下，如果将工具返回的错误消息重新返回给 LLM 时，LLM 会重新判断并继续执行函数调用，从而将参数补全，让程序正常执行，也是最推荐的一种错误处理方案，即在工具内部进行错误的捕获，并将错误信息独立返回。

## 02. 回退与重试处理

在前面的课时中，学习 Runnable 可运行组件时，我们学习了如果 Runnable 可运行组件出错，可以执行回退和重试两种策略，在函数调用中也可以使用这两种策略来处理，例如：

1. 在函数调用参数生成错误时，可以考虑回退到一个更好的模型，例如平时使用 `gpt-3.5-turbo-16k` 模型，在出错时，回退到 `gpt-4o` 模型上进行重新试验。
2. 亦或者是触发重试机制，并且在重试的时候，携带上错误信息，让 LLM 强大的自然语言处理功能进行自我纠正。

使用更好的模型进行回退，操作起来非常简单，只需要定义一个更强大的模型，并创建一条新链，然后使用 Runnable 可运行组件的 `.with_fallbacks()` 即可绑定对应的回退策略，示例代码如下：

```python
import dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

@tool
def complex_tool(int_arg: int, float_arg: float, dict_arg: dict) -> int:
    """使用复杂工具进行复杂计算操作"""
    return int_arg * float_arg

# 1.创建大语言模型并绑定工具
llm = ChatOpenAI(model="gpt-3.5-turbo-16k").bind_tools([complex_tool])
better_llm = ChatOpenAI(model="gpt-4o").bind_tools(
    [complex_tool], tool_choice="complex_tool",
)

# 2.创建链并执行工具
better_chain = better_llm | (lambda msg: msg.tool_calls[0]["args"]) | complex_tool
chain = (llm | (lambda msg: msg.tool_calls[0]["args"]) | complex_tool).with_fallbacks([better_chain])

# 3.调用链
print(chain.invoke("使用复杂工具，对应参数为5和2.1"))
```

回退策略并不是万能的，有的时候哪怕使用了参数更大的模型，生成的函数调用参数依旧不符合规范，这种情况就要考虑下工具的相关描述、Prompt 编写得是否存在问题。

另外一种优化策略是携带错误信息的重试策略，让 LLM 纠正其行为，只需要在抛出错误时，将错误信息一起携带给 LLM，让其重新操作即可，示例代码如下：

```python
from typing import Any
import dotenv
from langchain_core.messages import ToolCall, AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


class CustomToolException(Exception):
    """自定义的工具错误异常"""
    def __init__(self, tool_call: ToolCall, exception: Exception) -> None:
        super().__init__()
        self.tool_call = tool_call
        self.exception = exception


@tool
def complex_tool(int_arg: int, float_arg: float, dict_arg: dict) -> int:
    """使用复杂工具进行复杂计算操作"""
    return int_arg * float_arg


def tool_custom_exception(msg: AIMessage, config: RunnableConfig) -> Any:
    try:
        return complex_tool.invoke(msg.tool_calls[0]["args"], config)
    except Exception as e:
        raise CustomToolException(msg.tool_calls[0], e)


def exception_to_messages(inputs: dict) -> dict:
    # 1.从输入中提取错误信息
    exception = inputs.pop("exception")
    # 2.将历史消息添加到原始输入中，以便模型直到它在上一次工具调用中犯了一个错误
    messages = [
        AIMessage(content="", tool_calls=[exception.tool_call]),
        ToolMessage(tool_call_id=exception.tool_call["id"], content=str(exception.exception)),
        HumanMessage(content="最后一次工具调用引发了异常，请尝试使用更正的参数再次调用该工具，不要重复犯错。")
    ]
    inputs["last_output"] = messages
    return inputs


# 1.创建prompt，并预留占位符，用于存储错误输出信息
prompt = ChatPromptTemplate.from_messages([
    ("human", "{query}"),
    ("placeholder", "{last_output}"),
])

# 2.创建大语言模型并绑定工具
llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools=[complex_tool])

# 3.创建链并执行工具
chain = prompt | llm | tool_custom_exception
self_correcting_chain = chain.with_fallbacks(
    [exception_to_messages | chain], exception_key="exception",
)

# 4.调用自我纠正链完成任务
print(self_correcting_chain.invoke({"query": "使用复杂工具，对应参数为5和2.1"}))
```

输出内容：

```
10.5
```

