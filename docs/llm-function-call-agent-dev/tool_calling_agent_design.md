# 基于工具调用的智能体设计与实现

## 01. 工具调用智能体

基于 ReACT 架构的智能体会将 tools (工具描述)、agent_scratchpad (智能体草稿)、工具结果、推理等内容全部放到同一个 prompt 中，并通过提取 LLM 的规范输出来决定下一步的操作，这种模式会随着 LLM 输出的随机性，不同 LLM 性能的差异让程序变得异常脆弱。

而且 ReACT 架构早期设计之初是针对 LLM (文本补全模型) 进行设计的，即传入一段话，让 LLM 补全其后续的文本，随着 LLM 的发展，消息设计更友好、结构化输出更稳定的函数调用、性能更强大的 ChatModel 发布了，可以考虑将 ReACT 迁移到基于聊天消息 + 工具调用的架构上，思想不变，但是使用更稳定的消息列表 + 工具调用。

迁移流程图如下：

![image-20260818220747394](images/image-20260818220747394.png)

在上述的工具调用智能体 Prompt 中，输出规范会通过检测 LLM 是输出文本内容还是工具调用参数来判断下一步是什么，这样性能更加稳定，而且对于绝大部分 LLM 来说，工具调用支持一次性调用生成多个工具的参数，性能会更强。

在 LangChain 中，其实也为基于工具调用的 Agent 封装了一个快速创建的方法 `create_tool_calling_agent()` 和预设 Prompt。

- LangChain hub 工具调用 Prompt 链接：https://smith.langchain.com/hub/hwchase17/openai-tools-agent
- 基于工具调用的智能体文档：https://python.langchain.com/v0.1/docs/modules/agents/agent_types/tool_calling/

## 02. 实现示例

在 LangChain 中，要实现工具调用 Agent 其实也非常简单，步骤其实和 ReACT‑Agent 一模一样，创建好工具列表、Prompt、LLM (支持工具调用)，然后使用 `create_tool_calling_agent()` 创建智能体，接下来创建智能体执行者完成包装即可。

例如实现一个可以根据用户输入实现自主选择【联网搜索 + 文生图】的智能体，示例代码如下：

```python
import dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import GoogleSerperRun
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.dalle_image_generator import DalleAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


class GoogleSerperArgsSchema(BaseModel):
    query: str = Field(description="执行谷歌搜索的查询语句")


class DalleArgsSchema(BaseModel):
    query: str = Field(description="输入应该是生成图像的文本提示(prompt)")


# 1.定义工具与工具列表
google_serper = GoogleSerperRun(
    name="google_serper",
    description=(
        "一个低成本的谷歌搜索API。"
        "当你需要回答有关时事的问题时，可以调用该工具。"
        "该工具的输入是搜索查询语句。"
    ),
    args_schema=GoogleSerperArgsSchema,
    api_wrapper=GoogleSerperAPIWrapper(),
)

dalle = OpenAIDALLEImageGenerationTool(
    name="openai_dalle",
    api_wrapper=DalleAPIWrapper(model="dall‑e‑3"),
    args_schema=DalleArgsSchema,
)

tools = [google_serper, dalle]

# 2.定义工具调用agent提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 2.创建大语言模型
llm = ChatOpenAI(model="gpt‑4o‑mini")

# 3.创建agent与agent执行者
agent = create_tool_calling_agent(
    prompt=prompt,
    llm=llm,
    tools=tools,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent_executor.invoke({"input": "请帮我绘制一张老爷爷爬山的图片"}))
```

输出内容：

```
> Entering new AgentExecutor chain...

Invoking: `openai_dalle` with `{'query': 'An elderly man climbing a mountain, with a determined expression, wearing typical hiking gear. The background shows a scenic mountain landscape with trees and a clear sky. The scene conveys a sense of adventure and resilience.'}`

https://dalleproduse.blob.core.windows.net/private/images/6fa42821‑1331‑4b18‑90dc‑12364b6fb390/generated_00.png?se=2024‑08‑19T13%3A32%3A20Z&sig=vro%2BFrpJbsEbrMxeQnV8rbkE5GKU5tTqlCEvc908V2E%3D&ske=2024‑08‑23T22%3A55%3A33Z&skoid=09ba021e‑c417‑441c‑b203‑c81e5dcd7b7f&sks=b&skt=2024‑08‑16T22%3A22%3A3A55%3A33Z&sktid=33e01921‑4d64‑4f8c‑a055‑5bdaffd5e33d&skv=2020‑10‑02&sp=r&spr=https&sr=b&sv=2020‑10‑02这是我为您绘制的图片：



希望您喜欢这幅作品！

> Finished chain.

{'input': '请帮我绘制一张老爷爷爬山的图片', 'output': '这是我为您绘制的老爷爷爬山的图片：\n\n\n\n希望您喜欢这幅作品！'}
```

修改成提问「马拉松的世界纪录是多少？」，该智能体的回复如下：

```
> Entering new AgentExecutor chain...

Invoking: `google_serper` with `{'query': '马拉松世界纪录 2023'}`

2023年4月的伦敦马拉松赛，他以2小时1分25秒夺冠，这个成绩仅比当时的世界纪录慢了16秒。 2023年10月的芝加哥马拉松赛，以2小时00分35秒打破世界纪录。 仅有的三次马拉松经历，各个都是世界前六的好成绩。
截至2023年10月，马拉松的世界纪录是由选手在芝加哥马拉松上创造的，时间为2小时00分35秒。

> Finished chain.

{'input': '马拉松的世界纪录是多少？', 'output': '截至2023年10月，马拉松的世界纪录是由选手在芝加哥马拉松上创造的，时间为2小时00分35秒。'}
```