
import json
import requests

# ===================== 配置区 =====================
DASHSCOPE_API_KEY = "sk-61818ac1168b403dbc1e0653710e1f0a"  # 
MODEL_NAME = "qwen-turbo"
API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


def build_realign_prompt(
    chapter: int,
    sentence: int,
    original_text: str,
    modified_text: str,
    dp_edit_distance: int,
    dp_operations: list,
) -> str:
    """
    构造让大模型“重排操作序列”的 Prompt。
    """
    # 把 DP 的 operations 也给模型看，说明这是一份“算法给的、可能不自然”的方案
    dp_ops_str = json.dumps(dp_operations, ensure_ascii=False, indent=2)

    prompt = f"""你是古籍校勘专家，正在处理《道德经》的不同版本异文。

【任务】
现在给你一条句子对，以及一组由算法（编辑距离动态规划）自动生成的字符级操作序列。
这些操作在“总步数”上是最少的，但在中文语法和语义上不一定是最自然的。
请你在 **不改变总编辑距离（步数）的前提下**，重新设计一组更符合中文语感的操作序列。

【基本定义】
- 我们在字符级别操作，允许三种操作：
  1. replace：把原句中的一个字符替换成另一个字符
  2. insert：在原句的某个位置插入一个字符
  3. delete：删除原句中的一个字符
- 每个操作的代价为 1，总编辑距离 = 操作步数。
- 位置 position 采用 0-based，下标始终相对于“当前字符串”的索引。从 original_text 出发，按顺序依次执行 operations：
    第一条操作的 position 在 original_text 上取索引；
    第二条操作的 position 在“执行完第一条操作后的字符串”上取索引；
    以此类推。
    请务必保证在每一步中，position 都在当前字符串的合法范围内（0 ≤ position ＜ 当前长度；insert 允许 position == 当前长度，表示在末尾插入）。

【需要满足的硬性约束】
1. 你的输出中，"edit_distance" 必须等于 "operations" 的长度。
2. 如果从 original_text 出发，按你给出的 operations 依次执行，最终字符串必须完全等于 modified_text。
3. 操作类型只能是："replace" / "insert" / "delete"。
4. 每条操作的字段格式：
   - replace:
     {{
       "type": "replace",
       "original_char": "<原来的字>",
       "target_char": "<替换后的字>",
       "position": <int, 0-based>
     }}
   - insert:
     {{
       "type": "insert",
       "char": "<要插入的字>",
       "position": <int, 0-based 或 len(当前字符串)>
     }}
   - delete:
     {{
       "type": "delete",
       "char": "<被删除的字>",
       "position": <int, 0-based>
     }}

【语感要求】
- 在总步数不变的前提下，尽量让每一步操作在中文语法与语义上更自然：
  - 尽量把明显对应的一对字（如“智”和“知”）视为一次 replace，而不是“删一个字 + 把别的字替成它”。
  - 对于明显多余的虚词或语尾助词（如“也”），更倾向于 delete。
  - 不要为了“凑步数”而做绕远路的替换。

【示例（仅示意风格）】
例如：
- 原句：是以聖人自智而不自見也
- 目标句：是以聖人自知不自見
一种较自然的解析是：
- replace: 智 → 知
- delete: 而
- delete: 也

【本次输入】
章节：{chapter}
句号：{sentence}
original_text：{original_text}
modified_text：{modified_text}

算法给出的编辑距离（步数）：{dp_edit_distance}
算法给出的操作序列（仅供参考，可能不自然）：
{dp_ops_str}

【输出要求】
请你只输出一个 JSON，对象结构如下（不要有任何多余文字）：
{{
  "edit_distance": <int>,
  "operations": [
    {{
      "type": "replace" | "insert" | "delete",
      ...
    }},
    ...
  ]
}}
"""
    return prompt


def call_qwen(prompt: str) -> dict:
    """
    调用通义千问 API，返回解析后的 JSON 结果。
    期望模型的最终 content 是一个 JSON 字符串。
    """
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "input": {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        "parameters": {
            "result_format": "json",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 800,
        },
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # 通义千问的 JSON 格式与你在 costume_edit_distance.py 中类似
    try:
        content = data["output"]["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected API response format: {json.dumps(data, ensure_ascii=False)}")

    # content 应该就是一个 JSON 字符串
    try:
        return json.loads(content)
    except Exception:
        raise RuntimeError(f"Model content is not valid JSON: {content}")


def apply_operations(src: str, operations: list) -> str:
    """
    从 src 出发，依次执行 operations，返回最终字符串。
    注意：这里把 position 解释为“相对于原始字符串的下标”，
    因此需要在内部维护一份从原始下标到当前字符串位置的映射。
    """
    # 序列中每个元素为 (当前字符, 原始下标或 None[表示后插入的字符])
    seq = [(ch, idx) for idx, ch in enumerate(src)]

    def find_index_by_original_pos(original_pos: int) -> int:
        """根据原始下标，在当前序列中找到对应位置的索引。"""
        for i, (_ch, orig_idx) in enumerate(seq):
            if orig_idx == original_pos:
                return i
        raise ValueError(f"Cannot find original position {original_pos} in current sequence")

    for op in operations:
        op_type = op.get("type")
        pos = op["position"]

        if op_type == "replace":
            target_char = op["target_char"]
            idx = find_index_by_original_pos(pos)
            ch, orig_idx = seq[idx]
            seq[idx] = (target_char, orig_idx)

        elif op_type == "delete":
            idx = find_index_by_original_pos(pos)
            del seq[idx]

        elif op_type == "insert":
            # 对于插入操作，position 理解为：
            # “在原始下标 >= position 的第一个字符之前插入；
            #  如果不存在这样的字符，则在末尾插入”
            char = op["char"]
            insert_idx = None
            for i, (_ch, orig_idx) in enumerate(seq):
                if orig_idx is not None and orig_idx >= pos:
                    insert_idx = i
                    break
            if insert_idx is None:
                insert_idx = len(seq)
            seq.insert(insert_idx, (char, None))

        else:
            raise ValueError(f"Unknown operation type: {op_type}")

    return "".join(ch for ch, _orig in seq)


def main():
    # ========== 1. 准备“单条示例数据” ==========
    chapter_number = 35
    sentence_number = 6
    original_text = "是以聖人自智而不自見也"
    modified_text = "是以聖人自知不自見"

    # 这里的 dp_operations 请按你现在 sentence_edit_distance.json 中实际的记录来填
    # 下面是你提到的那条（算法版本）的示例结构（请按真实数据调整 position 等）：
    dp_edit_distance = 3
    dp_operations = [
        {
            "type": "delete",
            "char": "智",
            "position": 5
        },
        {
            "type": "replace",
            "original_char": "而",
            "target_char": "知",
            "position": 6
        },
        {
            "type": "delete",
            "char": "也",
            "position": 10
        },
    ]

    # ========== 2. 构造 Prompt ==========
    prompt = build_realign_prompt(
        chapter_number,
        sentence_number,
        original_text,
        modified_text,
        dp_edit_distance,
        dp_operations,
    )

    # ========== 3. 调用通义千问 ==========
    print("调用大模型进行重排，请稍候...")
    result = call_qwen(prompt)

    # ========== 4. 解析与校验 ==========
    print("\n模型返回原始 JSON：")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    edit_distance = result.get("edit_distance")
    operations = result.get("operations", [])

    if not isinstance(edit_distance, int):
        raise ValueError("edit_distance must be an integer")
    if not isinstance(operations, list):
        raise ValueError("operations must be a list")

    if edit_distance != len(operations):
        raise ValueError(
            f"edit_distance({edit_distance}) != len(operations)({len(operations)})"
        )

    # 模拟执行操作，检查是否能从 original_text 得到 modified_text
    final_text = apply_operations(original_text, operations)
    if final_text != modified_text:
        raise ValueError(
            f"Applying operations did not reach target text.\n"
            f"Expected: {modified_text}\nGot:      {final_text}"
        )

    # ========== 5. 打印对比结果 ==========
    print("\n=== 原始 DP 操作序列 ===")
    print(json.dumps(dp_operations, ensure_ascii=False, indent=2))

    print("\n=== 大模型重排后的操作序列 ===")
    print(json.dumps(operations, ensure_ascii=False, indent=2))

    print("\n校验成功：")
    print(f"- 编辑距离（步数）: {edit_distance}")
    print(f"- 执行后得到的句子: {final_text}")


if __name__ == "__main__":
    main()