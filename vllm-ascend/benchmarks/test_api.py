from openai import OpenAI

# 连接本地大模型服务
client = OpenAI(
    base_url="http://127.0.0.1:8001/v1",  # 大部分服务默认拼接/v1
    api_key="dummy-key"  # 本地部署无密钥随便填
)

# 对话调用
resp = client.chat.completions.create(
    model="/home/trainee/models/Qwen3-8B",  # 本地模型名随意填
    messages=[
        {"role": "user", "content": "你好，简单介绍自己"}
    ],
    temperature=0.7
)

print(resp.choices[0].message.content)