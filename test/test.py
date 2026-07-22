# import os
# # 在 import vllm 之前设置 NPU 可见设备
# os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "6"
# # 如有需要也设 vllm-ascend 插件
# # os.environ["VLLM_PLUGINS"] = "ascend"
# # os.environ["VLLM_USE_V1"] = "0"

# from vllm import LLM, SamplingParams

# MODEL_PATH = "/home/trainee/models/Qwen3-8B"
# DRAFT_MODEL_PATH = "/home/trainee/models/Qwen3-8B-DFlash-b16"

# llm = LLM(
#     model=MODEL_PATH,
#     trust_remote_code=True,
#     gpu_memory_utilization=0.9,
#     max_num_batched_tokens=16384,
#     # 推测解码配置（vllm 较新版本支持在 LLM 中传 dict）
#     speculative_config={
#         "num_speculative_tokens": 15,
#         "method": "dflash",
#         "model": DRAFT_MODEL_PATH,
#         "enforce_eager": True,
#     },
#     # Ascend 单卡可不设 tensor_parallel_size，默认 1
#     # tensor_parallel_size=1,
# )

# sampling_params = SamplingParams(
#     temperature=0.6,
#     top_p=0.95,
#     top_k=20,
#     max_tokens=2048,
# )

# messages = [[
#     {"role": "user", "content": "简单介绍一下大型语言模型。"}
# ]]

# outputs = llm.chat(
#     messages,
#     sampling_params,
#     chat_template_kwargs={"enable_thinking": False},   # 对应你原命令 enable_thinking=False
# )

# for out in outputs:
#     print(out.outputs[0].text)

prompt = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"


import requests
import json

def test_vllm_llm_api(base_url="http://127.0.0.1:8080/v1"):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "/home/trainee/models/Qwen3-8B",  # 替换你启动vLLM时加载的模型名
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 128
    }
    try:
        resp = requests.post(
            url=f"{base_url}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=99999
        )
        if resp.status_code == 200:
            res = resp.json()
            print("✅ vLLM 服务推理正常！")
            print("模型返回内容：")
            print(res["choices"][0]["message"]["content"])
            return True
        else:
            print(f"❌ 接口报错，状态码：{resp.status_code}")
            print("返回详情：", resp.text)
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接vLLM服务，端口未开放/服务未启动")
        return False
    except Exception as e:
        print(f"⚠️ 请求异常：{e}")
        return False

if __name__ == "__main__":
    # 改成你的部署地址
    test_vllm_llm_api(base_url="http://127.0.0.1:8080/v1")

# import multiprocessing

# # 获取CPU逻辑核心数
# cpu_num = multiprocessing.cpu_count()
# print(f"CPU核心数：{cpu_num}")