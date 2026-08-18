# 多模态 LLM 执行函数调用的技巧

目前市面上的 LLM 除了能接收文本数据，还出现了不少多模态 LLM，这些 LLM 不仅能接收文本数据，还能接收图像、音频等内容，并且这类多模态 LLM 也支持函数调用功能，要想使用这些模型调用工具，只需要按照正常的方式将工具绑定到模型上，在传递消息给大语言模型时，按照多模态 LLM 的特定规则即可。

## 01. GPT‑4o 多模态输入

OpenAI 提供的 GPT‑4o 模型是一个多模态的模型（输入多模态），在传递消息列表时，可以在人类消息中添加图片地址，这样即可将特定的图片上传给 GPT‑4o 模型进行识别，从而实现多模态输入。

- GPT‑4o 多模态输入文档链接：https://platform.openai.com/docs/api-reference/chat/create

官方提供原生示例如下：

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0])
```

在 LangChain 中，如果消息也是多模态的，只需要按照特定的模型对应的消息结构历史创建 LangChain 消息即可，例如在上面的示例中，人类消息传递了一个列表，包含了文本和图片链接，转换成 LangChain 消息 / 提示模板如下：

```python
prompt = ChatPromptTemplate.from_messages([
    ("human", [
        {"type": "text", "text": "{query}"},
        {"type": "image_url", "image_url": {"url": "{image_url}"}}
    ])
])

print(prompt.invoke({
    "query": "这张图片所在的地址的天气怎么样",
    "image_url": "https://img1.baidu.com/it/u=644490943,1781886584&fm=253&fmt=auto&app=138&f=JPEG"
}).to_messages())
```

使用技巧和普通的提示模板没有差异，只是将原本的字符串替换成了列表字典的格式，输出内容：

```
[HumanMessage(content=[{'type': 'text', 'text': '这张图片所在的地址的天气怎么样'}, {'type': 'image_url', 'image_url': {'url': 'https://img1.baidu.com/it/u=644490943,1781886584&fm=253&fmt=auto&app=138&f=JPEG'}}])]
```

这里我们以 GPT‑40 + 天气预报查询工具 + 多模态输入 三个组件构建一个 LLM 应用，让该 LLM 应用能识别上传图片城市所在的天气信息，运行流程如下：

![image-20260818213332915](images/image-20260818213332915.png)

示例代码：

```python
import json
import os
from typing import Type, Any

import dotenv
import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


class GaodeWeatherArgsSchema(BaseModel):
    city: str = Field(description="需要查询天气预报的目标城市，例如：广州")


class GaodeWeatherTool(BaseTool):
    """根据传入的城市名查询天气"""
    name = "gaode_weather"
    description = "当你想询问天气或与天气相关的问题时的工具。"
    args_schema: Type[BaseModel] = GaodeWeatherArgsSchema

    def _run(self, *args: Any, **kwargs: Any) -> str:
        """运行工具获取对应城市的天气预报"""
        try:
            # 1.获取高德API秘钥，如果没有则抛出错误
            gaode_api_key = os.getenv("GAODE_API_KEY")
            if not gaode_api_key:
                return f"高德开放平台API秘钥未配置"

            # 2.提取传递的城市名字并查询行政编码
            city = kwargs.get("city", "")
            session = requests.session()
            api_domain = "https://restapi.amap.com/v3"
            city_response = session.request(
                method="GET",
                url=f"{api_domain}/config/district?keywords={city}&subdistrict=0&extensions=all&key={gaode_api_key}",
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            city_response.raise_for_status()
            city_data = city_response.json()

            # 3.提取行政编码调用天气预报查询接口
            if city_data.get("info") == "OK":
                if len(city_data.get("districts")) > 0:
                    ad_code = city_data["districts"][0]["adcode"]

                    weather_response = session.request(
                        method="GET",
                        url=f"{api_domain}/weather/weatherInfo?city={ad_code}&extensions=all&key={gaode_api_key}&output=json",
                        headers={"Content-Type": "application/json; charset=utf-8"},
                    )
                    weather_response.raise_for_status()
                    weather_data = weather_response.json()
                    if weather_data.get("info") == "OK":
                        return json.dumps(weather_data)

            session.close()
            return f"获取{kwargs.get('city')}天气预报信息失败"
            # 4.整合天气预报信息并返回
        except Exception as e:
            return f"获取{kwargs.get('city')}天气预报信息失败"


# 1.构建prompt
prompt = ChatPromptTemplate.from_messages([
    ("human", [
        {"type": "text", "text": "请获取下上传图片对应城市的天气信息。"},
        {"type": "image_url", "image_url": {"url": "{image_url}"}}
    ]),
])

weather_prompt = ChatPromptTemplate.from_template("""请整理下传递的城市的天气预报信息，并以用户友好的方式输出。
<weather>
{weather}
</weather>""")

# 2.构建LLM并绑定工具
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools=[GaodeWeatherTool()], tool_choice="gaode_weather")

# 3.创建链应用并执行
chain = (
    {
        "weather": (
            {"image_url": RunnablePassthrough()}
            | prompt
            | llm_with_tools
            | (lambda msg: msg.tool_calls[0]["args"])
            | GaodeWeatherTool()
        )
    }
    | weather_prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke("https://xxxx.com/guangzhou.jpg"))
```

输出内容：

```
以下是广州未来几天的天气预报信息：

### 广州市天气预报
**发布时间：2024年7月11日 12:00**

### 2024年7月11日（星期四）
- **白天天气：** 大雨
- **夜间天气：** 大雨
- **白天温度：** 33℃
- **夜间温度：** 25℃
- **白天风向：** 西南
- **夜间风向：** 西南
- **风力等级：** 1‑3级

### 2024年7月12日（星期五）
- **白天天气：** 大雨
- **夜间天气：** 中雨‑大雨
- **白天温度：** 33℃
- **夜间温度：** 24℃
- **白天风向：** 北
- **夜间风向：** 北
- **风力等级：** 1‑3级

### 2024年7月13日（星期六）
- **白天天气：** 中雨‑大雨
- **夜间天气：** 中雨‑大雨
- **白天温度：** 32℃
- **夜间温度：** 25℃
- **白天风向：** 西南
- **夜间风向：** 西南
- **风力等级：** 1‑3级

### 2024年7月14日（星期日）
- **白天天气：** 中雨‑大雨
- **夜间天气：** 大雨
- **白天温度：** 33℃
- **夜间温度：** 25℃
- **白天风向：** 北
- **夜间风向：** 北
- **风力等级：** 1‑3级

请注意天气变化，做好防雨准备。
```

## 02. GPT‑4o 接入 DALL‑E 文生图

除了输入的多模态，大语言模型的输出其实也可以通过曲线救国的方式实现**多模态输出**，有使用过 ChatGPT‑Plus 的小伙伴肯定了解，当我们和 ChatGPT 对话时，可以让 ChatGPT 帮忙生成对应的图片，该功能底层本质上也是通过**函数调用**来实现的，运行原理非常简单：

1. 创建一个根据**文本生成图片**工具，例如可以使用 DALLE，工具的调用参数为生成图片的 Prompt。
2. 将工具绑定到 LLM 上，并预设特定的 Prompt，让 LLM 将用户输入的 query 转换成绘图的 Prompt。
3. 调用工具，将 Prompt 生成图片，并将**工具消息、AI 消息**叠加到历史消息中，再次提问，获得最终答复。

运行流程和刚刚我们创建的图片获取天气预报应用非常接近，如下：

![image-20260818214049359](images/image-20260818214049359.png)

```python
import dotenv
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities.dalle_image_generator import DALL EAPIWrapper
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

dalle = OpenAIDALLEImageGenerationTool(
    name="openai_dalle",
    api_wrapper=DALLEAPIWrapper(model="dall‑e‑3")
)

llm = ChatOpenAI(model="gpt‑4o")
llm_with_tools = llm.bind_tools(tools=[dalle], tool_choice="openai_dalle")

chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | dalle

print(chain.invoke("帮我绘制一幅老爷爷的图片"))
```

输出内容：

```
https://oaidalleapiprodscus.blob.core.windows.net/private/org‑8mmokaDgFXOdOsE9HC1PNBZM/user‑XbaFwYqigWuDrIeZS6l61HgI/img‑NxKXd4xBMJZdIixcxYQe5dmd.png?st=2024‑08‑15T05%3A02%3A53Z&se=2024‑08‑15T07%3A02%3A53Z&sp=r&sv=2024‑08‑04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d‑d6c1‑4a0a‑bac7‑5c84a87759f8&sktid=a48cca56‑e6da‑484e‑a814‑9c849652bcb3&skt=2024‑08‑14T17%3A52%3A40Z&ske=2024‑08‑15T17%3A52%3A40Z&sks=b&skv=2024‑08‑04&sig=WHJ0%2BKW9OKRSKOaYLXL%2Btataexl5Vf9lJAGVxvmO9kM%3D
```