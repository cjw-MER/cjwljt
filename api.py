import os
from openai import OpenAI
from zhipuai import ZhipuAI
class LLMAPI:
    def __init__(self, model="qwen-plus"):
        self.model = model
    def chat(self, message):
        client = OpenAI(
            api_key="sk-a1ce4ad357354e50b9a03fe33ebcacd2",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=message,
            temperature=0.3
        )
        
        return completion.choices[0].message.content

class ZHI:
    def __init__(self, model="qwen-plus"):
        self.model = model
    def chat(self, message):
        # 1. 填写你的 API Key
        client = ZhipuAI(api_key="a8a611f66132c411c94bae2f8a598805.BQ2kPXL1BOshgxIb") 

        # 2. 创建聊天请求
        response = client.chat.completions.create(
            model="glm-4-flash",  # 指定使用免费的 Flash 模型
            messages=message,
            response_format={"type": "json_object"},
            temperature=0
        )
        return response.choices[0].message.content



class DEEPSEEK:
    def __init__(self, model="qwen-plus"):
        self.model = model
    def chat(self, message):
        client = OpenAI(
            api_key="sk-fdad79bedb7d4c378771bb48731422e8", # deepseek api
            base_url="https://api.deepseek.com",
        )
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=message,
            temperature=0
        )
        print(f"输入token: {completion.usage.prompt_tokens}")
        print(f"输出token: {completion.usage.completion_tokens}")
        print(f"总token: {completion.usage.total_tokens}")
        
        return completion.choices[0].message.content