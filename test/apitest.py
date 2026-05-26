import os
import json
from urllib import request, error

# 配置API密钥和端点（替换为你的API-KEY）

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODEL_NAME = "qwen-turbo-latest"
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


def list_qwen_model_codes():
    """
    列出可用模型 code（OpenAI 兼容模型列表接口）
    :return: (codes, raw_body)
    """
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/models"
    req = request.Request(
        url,
        headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
        method="GET",
    )
    with request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    codes = [item.get("id") for item in body.get("data", []) if item.get("id")]
    return codes, body


def call_qwen(prompt, model="qwen-turbo"):
    """
    调用通义千问API
    :param prompt: 用户提问内容
    :param model: 模型版本（qwen-turbo/qwen-plus/qwen-flash）
    :return: 模型回复
    """
    if not DASHSCOPE_API_KEY:
        return "调用失败：请先设置环境变量 DASHSCOPE_API_KEY"
    try:
        payload = {
            "model": model,
            "input": {"messages": [{"role": "user", "content": prompt}]},
            "parameters": {"temperature": 0.7, "max_tokens": 1024},
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        used_model = body.get("model") or model
        return f"[model={used_model}] {body['output']['text']}"
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        return f"调用失败：HTTP {e.code} - {detail}"
    except Exception as e:
        return f"调用失败：{str(e)}"

# 测试调用
if __name__ == "__main__":
    if not DASHSCOPE_API_KEY:
        print("请先设置环境变量 DASHSCOPE_API_KEY 后再运行。")
    else:
        try:
            model_codes, _ = list_qwen_model_codes()
            print("可用模型 code（部分）：")
            for code in model_codes[:30]:
                print("-", code)
        except Exception as e:
            print("获取模型列表失败：", e)

        result = call_qwen("请你告诉我你的模型版本", MODEL_NAME)
        print("模型回复：\n", result)