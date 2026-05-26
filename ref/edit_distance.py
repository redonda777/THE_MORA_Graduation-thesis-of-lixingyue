import textdistance

from agent import Agents, CharacterRelation
from typing import List, Tuple, Dict, Optional, Any, Set
import time


# 编辑距离函数，返回编辑距离和操作路径
def edit_distance(s1: str, s2: str) -> Tuple[float, List[Tuple[str, ...]]]:
    """
    计算两个字符串的原始编辑距离和操作路径
    省略、缺失等情况不应该在此出现
    """
    if s1 == "" or s2 == "":
        raise ValueError("s1或s2字符串不能为空")
        # 字符串为空，返回0.5

    if "……" in s1 or "……" in s2:
        raise ValueError("s1或s2字符串包含省略号")
        # 省略号而直接返回0.4

    m, n = len(s1), len(s2)
    # 创建 (m+1) x (n+1) 的DP表，初始化为0
    dp = [[0.] * (n + 1) for _ in range(m + 1)]
    # 创建操作表，记录每个位置的操作类型
    operations = [[None] * (n + 1) for _ in range(m + 1)]

    # 边界条件：第一行和第一列
    for i in range(m + 1):
        dp[i][0] = i  # s2为空，删除i个字符
        if i > 0:
            operations[i][0] = ('删除', s1[i - 1])
    for j in range(n + 1):
        dp[0][j] = j  # s1为空，插入j个字符
        if j > 0:
            operations[0][j] = ('插入', s2[j - 1])

    # 填充DP表和操作表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # s1[i],s2[j] 匹配时，代价为0
            # 这里对方框做特殊处理
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                if s2[j - 1] != "#":
                    # 匹配，且匹配的字符不是方框
                    operations[i][j] = ('匹配', s1[i - 1])
                else:
                    # 匹配，且匹配的为方框
                    operations[i][j] = ('替换', s1[i - 1], s2[j - 1])
            else:
                # 计算替换、删除、插入的代价
                replace_cost = dp[i - 1][j - 1] + 1
                delete_cost = dp[i - 1][j] + 1
                insert_cost = dp[i][j - 1] + 1

                # 选择最小代价的操作
                min_cost = min(replace_cost, delete_cost, insert_cost)
                dp[i][j] = min_cost

                # 记录操作类型
                if min_cost == replace_cost:
                    operations[i][j] = ('替换', s1[i - 1], s2[j - 1])
                elif min_cost == delete_cost:
                    operations[i][j] = ('删除', s1[i - 1])
                else:
                    operations[i][j] = ('插入', s2[j - 1])

    # 回溯生成操作路径
    path = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            op = operations[i][j]
            if op[0] == '匹配' or op[0] == '替换':
                path.append(op)
                i -= 1
                j -= 1
            elif op[0] == '删除':
                path.append(op)
                i -= 1
            else:  # 插入
                path.append(op)
                j -= 1
        elif i > 0:
            path.append(operations[i][j])
            i -= 1
        else:  # j > 0
            path.append(operations[i][j])
            j -= 1

    # 反转路径，使其从开始到结束
    path.reverse()

    return dp[m][n], path


def analyze_character_relations_with_agent(operations: List[Tuple[str, ...]],
                                           agent: Any,
                                           s1: str,
                                           s2: str,
                                           homophone_cost: float,
                                           interchangeable_cost: float,
                                           function_word_cost: float,
                                           missing_char_cost: float,
                                           private_char_cost: float,
                                           cache_agent_result: Optional[Dict[Tuple[str, str], Any]] = None) -> Tuple[
    List[Tuple[str, ...]], Optional[Any], bool]:
    """
    遍历编辑距离返回的操作，调用大模型agent判断特殊字符关系，并为特殊关系操作赋予自定义代价
    """
    if s1 == "" or s2 == "":
        raise ValueError("s1和s2字符串不能为空")

    if "……" in s1 or "……" in s2:
        raise ValueError("s1或s2字符串包含省略号")

    # 准备需要分析的字符对
    replace_pairs = []
    delete_chars = []
    insert_chars = []

    def is_private_char(c):
        return '\uE000' <= c <= '\uE0FF'

    for op in operations:
        if op[0] == '替换':
            if not (is_private_char(op[1]) or is_private_char(op[2])):
                replace_pairs.append((op[1], op[2]))
        elif op[0] == '删除':
            if not is_private_char(op[1]):
                delete_chars.append(op[1])
        elif op[0] == '插入':
            if not is_private_char(op[1]):
                insert_chars.append(op[1])
        else:
            # 匹配
            assert op[0] == '匹配'
            # print(f"未处理操作类型: {op}")

    try:
        # 构建操作描述
        operation_descriptions = []

        for pair in replace_pairs:
            operation_descriptions.append(f"替换{pair[0]}->{pair[1]}")

        if delete_chars:
            operation_descriptions.append(f"删除字符：{', '.join(delete_chars)}")
        if insert_chars:
            operation_descriptions.append(f"插入字符：{', '.join(insert_chars)}")

        operation = '; '.join(operation_descriptions) if operation_descriptions else "无特殊操作"

        cache_key = (s1, s2) if cache_agent_result is not None and (s1, s2) in cache_agent_result else (s2, s1)
        used_cache = False
        # 调用agent，添加超时和重试
        if cache_agent_result is not None:
            if cache_key in cache_agent_result:
                results = cache_agent_result[cache_key]
                used_cache = True
            else:
                results = None
        else:
            results = None
        api_ok = False
        if results is None:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    results = agent.invoke({
                        "text1": s1,
                        "text2": s2,
                        "operation": operation
                    })
                    api_ok = True
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Agent调用失败，第{attempt + 1}次重试...")
                        time.sleep(2)  # 等待2秒后重试
                    else:
                        print(f"Agent调用失败，使用默认值: {e}")
                        # 返回默认的CharacterRelation对象
                        results = CharacterRelation(homophone=[], interchangeable=[], function_word=[])
                        api_ok = False

        if cache_agent_result is not None and results is not None and api_ok:
            # 这里必须判断is not None，如果不显式写，会因为大小为0的错误而无法进入
            cache_agent_result[cache_key] = results
        # 安全地访问结果
        homophone_pairs = set(getattr(results, 'homophone', []) or [])
        interchangeable_pairs = set(getattr(results, 'interchangeable', []) or [])
        function_words = set(getattr(results, 'function_word', []) or [])

        # 为每个操作分配代价
        operations_with_cost = []
        for op in operations:
            if op[0] == '匹配':
                assert len(op) == 2, op
                operations_with_cost.append(op + (0.,))
            elif op[0] == '替换':
                assert len(op) == 3, op
                char1, char2 = op[1], op[2]

                if char1 == '#' or char2 == '#':
                    operations_with_cost.append(op + (missing_char_cost,))
                elif is_private_char(char1) or is_private_char(char2):
                    operations_with_cost.append(op + (private_char_cost,))
                elif (char1, char2) in interchangeable_pairs:
                    operations_with_cost.append(op + (interchangeable_cost,))
                elif (char1, char2) in homophone_pairs:
                    operations_with_cost.append(op + (homophone_cost,))
                else:
                    operations_with_cost.append(op + (1.,))
            elif op[0] == '删除' or op[0] == '插入':
                char = op[1]
                assert len(op) == 2, op
                if char == '#':
                    # todo: 这里有问题，插入或删除的缺失字符代价应该是0.5
                    operations_with_cost.append(op + (0.5,))
                elif is_private_char(char):
                    operations_with_cost.append(op + (private_char_cost,))
                elif char in function_words:
                    operations_with_cost.append(op + (function_word_cost,))
                else:
                    operations_with_cost.append(op + (1.,))
            else:
                print(f"未处理操作类型: {op[0]}")
                operations_with_cost.append(op + (1.,))
        '''
        for i in range(len(operations_with_cost)):
            op = operations_with_cost[i]
            if type(op[-1]) != float:
                operations_with_cost[i] = op + (1.,)
        '''
        return operations_with_cost, results, used_cache

    except Exception as e:
        print(f"分析字符关系时出错: {e}")
        # 出错时返回原始操作，所有操作代价为1
        operations_with_cost = [op + (1.,) if op[0] != '匹配' else op + (0.,) for op in operations]
        return operations_with_cost, None, False


def calculate_adjusted_edit_distance(s1: str, s2: str, agent: Agents,
                                     homophone_cost: float = 0.25,
                                     interchangeable_cost: float = 0.1,
                                     function_word_cost: float = 0.2,
                                     missing_char_cost: float = 0.3,
                                     private_char_cost: float = 0.3, do_length_normalization: bool = False,
                                     cache_agent_result: Optional[Dict[Tuple[str, str], Any]] = None,
                                     use_agent_analysis: bool = True) -> Tuple[
    float, list, object, float, bool]:
    """
    计算考虑特殊字符关系后的调整编辑距离，支持自定义代价值
    值越大，相似度越低
    """

    if s1 == "" or s2 == "":
        if s1 == "" or s2 == "":
            # 两个字符串都为空，返回0.5
            return 0.5, [("特殊操作", 0.5)], None, 0, False
        else:
            # 只有一个字符串为空，另一个非空，代价较大
            return 0.9, [("特殊操作", 0.9)], None, 0, False

    if "……" in s1 or "……" in s2:
        # 省略号查看剩余字符相似度，如果不相似返回0.4，如果相似则减去相似
        similarity = (len(textdistance.lcsseq(s1, s2)) - 1.) / (min(len(s1), len(s2)) - 1.)  # -1排除省略号影响
        cost = max(0.0, min(0.2, 0.4 - similarity))
        operations_with_cost = [("特殊操作", cost)]
        return cost, operations_with_cost, None, 0, False

    original_distance, operations = edit_distance(s1, s2)
    if int(original_distance) == 0:
        # 原始距离为0，直接返回
        operations_with_cost = [op + (0.,) for op in operations]
        return 0., operations_with_cost, None, original_distance, False
    # 检查是否有缓存结果
    if not use_agent_analysis:
        # 不使用大模型调整，直接返回原始编辑距离
        operations_with_cost = [op + (1.,) if op[0] != '匹配' else op + (0.,) for op in operations]
        adjusted_distance = sum(op[-1] for op in operations_with_cost)
        assert adjusted_distance == original_distance, (adjusted_distance, original_distance)
        used_cache = False
        analysis_results = None

    else:
        # 根据参数决定是否使用大模型调整
        if cache_agent_result is not None:
            # 调用agent获取结果并缓存
            operations_with_cost, analysis_results, used_cache = analyze_character_relations_with_agent(
                operations,
                agent,
                s1,
                s2,
                homophone_cost,
                interchangeable_cost,
                function_word_cost,
                missing_char_cost,
                private_char_cost,
                cache_agent_result
            )
        else:
            # 没有缓存，直接调用agent
            operations_with_cost, analysis_results, used_cache = analyze_character_relations_with_agent(
                operations,
                agent,
                s1,
                s2,
                homophone_cost,
                interchangeable_cost,
                function_word_cost,
                missing_char_cost,
                private_char_cost,
                None
            )
        for op in operations_with_cost:
            assert type(op[-1]) == float, (operations_with_cost, op)
        adjusted_distance = sum(op[-1] for op in operations_with_cost)

    # 归一化：除以两个字符串的最大长度
    max_len = max(len(s1), len(s2))
    if max_len > 0:
        if do_length_normalization:
            normalized_distance = adjusted_distance / max_len
        else:
            normalized_distance = adjusted_distance
    else:
        normalized_distance = 0.0

    return normalized_distance, operations_with_cost, analysis_results, original_distance, used_cache


# 主函数示例
if __name__ == "__main__":
    try:
        # 创建agent实例
        agent_instance = Agents()

        print("=== 中文示例测试 ===")
        s1_cn = "学而时习之"
        s2_cn = "学而时習之"

        # 使用默认代价值计算
        adjusted_distance, ops_with_cost, analysis, _, used_cache = calculate_adjusted_edit_distance(
            s1_cn, s2_cn, agent_instance
        )
        print(f"调整后编辑距离: {adjusted_distance}")
        print("带代价值的操作路径:")
        for op in ops_with_cost:
            if op[0] == '匹配':
                print(f"  {op[0]}: '{op[1]}', 代价: {op[-1]}")
            elif op[0] == '替换':
                print(f"  {op[0]}: '{op[1]}' -> '{op[2]}', 代价: {op[-1]}")
            else:
                print(f"  {op[0]}: '{op[1]}', 代价: {op[-1]}")
        if analysis:
            print("\n字符关系分析结果（默认代价值）:")
            if analysis.homophone:
                print(f"同音字对: {analysis.homophone}")
            if analysis.synonym:
                print(f"同义字对: {analysis.synonym}")
            if analysis.interchangeable:
                print(f"通假字对: {analysis.interchangeable}")
            if analysis.function_word:
                print(f"虚词列表: {analysis.function_word}")

        # 使用自定义代价值计算
        print("\n=== 使用自定义代价值 ===")
        custom_distance, custom_ops, custom_analysis, _, _ = calculate_adjusted_edit_distance(
            s1_cn, s2_cn, agent_instance,
            homophone_cost=0.2,
            synonym_cost=0.3,
            interchangeable_cost=0.05,
            function_word_cost=0.2,
            missing_char_cost=0.3
        )
        print(f"使用自定义代价值的编辑距离: {custom_distance}")

        # 安全地访问分析结果
        if custom_analysis:
            print("\n字符关系分析结果:")
            # 使用getattr安全访问属性
            homophone = getattr(custom_analysis, 'homophone', None)
            synonym = getattr(custom_analysis, 'synonym', None)
            interchangeable = getattr(custom_analysis, 'interchangeable', None)
            function_word = getattr(custom_analysis, 'function_word', None)

            if homophone:
                print(f"同音字对: {homophone}")
            if synonym:
                print(f"同义字对: {synonym}")
            if interchangeable:
                print(f"通假字对: {interchangeable}")
            if function_word:
                print(f"虚词列表: {function_word}")

        print("\n=== 测试用例: 特殊方括号 ===")
        s1_bracket = "其用不"
        s2_bracket = "亓甬不"

        adjusted_distance, ops_with_cost, analysis, _, used_cache = calculate_adjusted_edit_distance(
            s1_bracket, s2_bracket, agent_instance)
        print(f"文本1: {s1_bracket}")
        print(f"文本2: {s2_bracket}")
        print(f"调整后编辑距离: {adjusted_distance}")
        print("带代价值的操作路径:")
        for op in ops_with_cost:
            if op[0] == '匹配':
                print(f"  {op[0]}: '{op[1]}', 代价: {op[-1]}")
            elif op[0] == '替换':
                print(f"  {op[0]}: '{op[1]}' -> '{op[2]}', 代价: {op[-1]}")
            else:
                print(f"  {op[0]}: '{op[1]}', 代价: {op[-1]}")

        if analysis:
            print("\n字符关系分析结果:")
            if analysis.homophone:
                print(f"同音字对: {analysis.homophone}")
            if analysis.synonym:
                print(f"同义字对: {analysis.synonym}")
            if analysis.interchangeable:
                print(f"通假字对: {analysis.interchangeable}")
            if analysis.function_word:
                print(f"虚词列表: {analysis.function_word}")

        # 测试归一化效果
        print("\n=== 归一化测试 ===")
        s1_test = "学而时习之"
        s2_test = "学而时習之"
        normalized_distance, _, _, _, used_cache = calculate_adjusted_edit_distance(s1_test, s2_test, agent_instance)
        print(f"文本1: {s1_test}")
        print(f"文本2: {s2_test}")
        print(f"归一化后的编辑距离: {normalized_distance}")
        print(f"最大长度: {max(len(s1_test), len(s2_test))}")

    except Exception as e:
        print(f"计算调整后编辑距离时出错: {e}")
