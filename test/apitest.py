# 安装依赖（首次运行前执行）
# pip install openai

from openai import OpenAI
import os

# 配置API密钥和端点（替换为你的API-KEY）
API_KEY = "sk-61818ac1168b403dbc1e0653710e1f0a"  # 你的API-KEY
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 初始化客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def call_qwen(prompt, model="qwen-turbo"):
    """
    调用通义千问API
    :param prompt: 用户提问内容
    :param model: 模型版本（qwen-turbo/qwen-plus/qwen-flash）
    :return: 模型回复
    """
    try:
        # 发送请求
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # 随机性（0-1，越小越精准）
            max_tokens=1024    # 最大回复长度
        )
        # 返回回复内容
        return response.choices[0].message.content
    except Exception as e:
        return f"调用失败：{str(e)}"

# 测试调用
if __name__ == "__main__":
    result = call_qwen("请你告诉我你的模型版本")
    print("模型回复：\n", result)