import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field
from typing import Optional, List, Tuple
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate


class CharacterRelation(BaseModel):
    """字符关系分类结果"""
    homophone: List[Tuple[str, str]] = Field(default_factory=list, description="同/近音字对列表，可能有一对或多对")
    interchangeable: List[Tuple[str, str]] = Field(default_factory=list, description="异体字对列表，可能有一对或多对")
    function_word: List[str] = Field(default_factory=list, description="虚词列表")


class Agents:
    """基础agent类，封装通用功能"""

    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature
        self.agents = {}
        # 初始化模型连接等通用操作
        self.model = self._initialize_model()
        self.construct_agents()

    def _initialize_model(self):
        """初始化模型连接，子类可重写"""
        api_key = os.getenv("DeepSeek_API_KEY")
        if not api_key:
            # 加载当前目录的 .env 文件
            load_dotenv()
            api_key = os.getenv("DeepSeek_API_KEY")
    
        if not api_key:
            print("警告: 未找到DEEPSEEK_API_KEY环境变量")
            raise Exception("未找到DEEPSEEK_API_KEY环境变量")
        try:
            llm = ChatDeepSeek(
            model="deepseek-chat",
            temperature=self.temperature,
            max_tokens=1000,
            timeout=30,  # 增加超时时间
            max_retries=2,
            api_key=api_key,
        )
            return llm
        except Exception as e:
            print(f"初始化DeepSeek模型失败: {e}")
            raise Exception("未找到DEEPSEEK_API_KEY环境变量")

    def build_template(self, system_prompt, examples):
        # 创建统一的human消息格式模板
        human_message_template = "文本1 {text1} 文本2 {text2} 操作 {operation}"
        
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", human_message_template),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )
        
        final_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                few_shot_prompt,
                ("human", human_message_template),
            ]
        )
        return final_prompt

    def construct_agents(self):
        """构造agents，保持与原始文件一致的结构"""
        self.agents['character_relation'] = self.construct_agent()

    def construct_agent(self):
        """构造单个agent"""
        system_prompt = """你是一个古文研究专家，精通古文字学。你需要查看两个文本对，并关注两者差异。判断每对差别文字是否为以下情况：
        1. 异体：如"为"和"為"、"泛"和"氾"、"惪"和"德"等 
        2. 音同/近：如"知"和"智"、"说"和"悦"、"景"通"影"等。
        3. 虚词：之、乎、者、也、矣、焉、哉、夫、盖、惟、唯、其、而、以等。
单纯含义接近不属于以上情况（"後"和"退", "離"和"遠"）。如果有以上情况，请以列表加元组的形式返回。
        """
        examples = [
            {
                "text1": "大道泛兮其可左右", 
                "text2": "大道氾兮其可左右",
                "operation": "替换泛->氾",
                "output": '{"homophone":[],"interchangeable":[("泛","氾")],"function_word":[]}'
            },
            {
                "text1": "无为而无不为", 
                "text2": "無為而無不為",
                "operation": "替换无->無,替换为->為",
                "output": '{"homophone":[],"interchangeable":[("无","無"),("为","為")],"function_word":[]}'
            },
            {
                "text1": "大邦者下流也",
                "text2": "大國者下流",
                "operation": "替换邦->國, 删除也",
                "output": '{"homophone":[],"interchangeable":[],"function_word":["也"]}'
            },
            {
                "text1": "不遠亓甾重",
                "text2": "不離輜重",
                "operation": "替换遠->離,删除亓,替换甾->輜",
                "output": '{"homophone":[("甾","輜")],"interchangeable":[],"function_word":["亓"]}'
            },
        ]
        template = self.build_template(system_prompt, examples)
        structured_llm = self.model.with_structured_output(CharacterRelation)
        agent = template | structured_llm
        return agent

    def invoke(self, input_data):
        """调用agent的统一接口"""
        return self.agents['character_relation'].invoke(input_data)