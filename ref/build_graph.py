import argparse
import json
from datetime import datetime
import time

import networkx as nx
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm
from edit_distance import calculate_adjusted_edit_distance, Agents

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'SimHei'


class BookGraph:
    def __init__(self, agent_instance=None, use_all_version=True):
        self.G = nx.Graph()
        self.agent = agent_instance if agent_instance else Agents()
        self.sentence_nodes = {}
        self.version_sentences = {}
        if use_all_version:
            self.versions = ['hj', 'gd', 'gd1', 'gd2', 'gd3', 'ba', 'bb', 'wb', 'hs', 'yz', 'xr', 'fy']
        else:
            self.versions = ['hj', 'ba', 'bb', 'wb', 'hs', 'yz', 'xr', 'fy']

    def build_from_dataframe(self, df, similarity_threshold, cache_file="similarity_cache.pkl",
                             resume_from_cache=False, save_interval=50, include_empty_nodes=False, do_length_normalization=False, use_agent_analysis=True):
        """
        从DataFrame构建图结构，支持相似度计算的缓存和进度恢复
        修改：1. 在构造chapter_node时判断是否包含非空statement节点
             2. 只有当text非空时才创建statement类型节点
        """
        # 初始化每个版本的句子字典
        for version in self.versions:
            self.version_sentences[version] = {}

        # 用于跟踪每个章节是否包含非空语句节点
        chapter_has_nonempty_statements = set()

        # 第一遍：收集数据并标记包含非空语句的章节
        print("正在收集数据并标记有效章节...")
        for index, row in tqdm(df.iterrows()):
            seg = row['seg']
            ln = row['ln']

            for version in self.versions:
                # 记录句子信息
                text = ""
                if version in row and pd.notna(row[version]):
                    text = row[version]

                # 标记包含非空语句的章节
                chapter_node = f"{version}_{seg}"
                if text.strip() != '~' or include_empty_nodes:  # 非空语句
                    chapter_has_nonempty_statements.add(chapter_node)

                    # 记录句子信息到字典
                    node_id = f"{seg}_{ln}"
                    if node_id not in self.sentence_nodes:
                        self.sentence_nodes[node_id] = {}
                    statement_node = f"{version}_{seg}_{ln}"
                    self.sentence_nodes[node_id][version] = {'node_id': statement_node, 'text': text}

                if seg not in self.version_sentences[version]:
                    self.version_sentences[version][seg] = {}
                self.version_sentences[version][seg][ln] = {
                    'node_id': f"{version}_{seg}_{ln}",
                    'text': text
                }

        print(f"共标记 {len(chapter_has_nonempty_statements)} 个包含非空语句的章节")

        # 第二遍：构建节点和边
        print("正在构建图节点和边...")
        statement_node_count = 0  # 计数创建的非空语句节点

        for index, row in tqdm(df.iterrows()):
            seg = row['seg']
            ln = row['ln']

            for version in self.versions:
                # 版本节点
                version_node = f"{version}"
                if not self.G.has_node(version_node):
                    self.G.add_node(version_node, type="version", text=version_node, version=version)

                chapter_node = f"{version}_{seg}"
                # 只创建包含非空语句的章节节点
                if chapter_node in chapter_has_nonempty_statements and not self.G.has_node(chapter_node):
                    self.G.add_node(chapter_node, type="chapter", text=chapter_node, seg=seg, version=version)
                    self.G.add_edge(version_node, chapter_node, relation="contains")

                # 语句节点 - 只有当text非空时才创建
                text = ""
                if version in row and pd.notna(row[version]):
                    text = row[version]

                # 只有当text非空时才创建statement节点
                if text.strip() != '~' or include_empty_nodes:
                    statement_node = f"{version}_{seg}_{ln}"
                    if self.G.has_node(statement_node):
                        continue
                    self.G.add_node(statement_node, type="statement", text=text, version=version, seg=seg, ln=ln)
                    statement_node_count += 1

                    # 只有当章节节点存在时才添加边
                    assert self.G.has_node(chapter_node)
                    self.G.add_edge(chapter_node, statement_node, relation="contains")
                    self.G.add_edge(version_node, statement_node, relation="contains")

        print(f"共创建 {statement_node_count} 个非空语句节点")

        # 计算相似度并添加相似边
        self._calculate_similarities(similarity_threshold, cache_file, resume_from_cache, save_interval, do_length_normalization, use_agent_analysis)

    def _calculate_similarities(self, similarity_threshold, cache_file="similarity_cache.pkl",
                                resume_from_cache=False, save_interval=50, do_length_normalization=False, use_agent_analysis=True):
        """
        计算版本间句子相似度，支持结果缓存和进度恢复

        参数:
            similarity_threshold: 相似度阈值
            cache_file: 缓存文件路径
            resume_from_cache: 是否从缓存恢复进度
            save_interval: 每处理多少个句子对保存一次缓存
            do_length_normalization: 是否归一化距离
            use_agent_analysis: 是否使用大模型分析
        """
        similarity_edges_count = 0
        processed_pairs = set()  # 已处理的句子对集合
        cache_agent_result = {}  # 大模型结果缓存
        start_time = time.time()
        last_save_time = start_time

        cache_hits = 0  # 缓存命中次数
        cache_miss = 0  # 缓存未命中次数

        # 尝试从缓存恢复
        if resume_from_cache and os.path.exists(cache_file) and use_agent_analysis:
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    cache_agent_result = cache_data.get('cache_agent_result', {})
            except Exception as e:
                print(f"加载缓存失败，将从头开始: {e}")
                cache_agent_result = {}

        total_pairs = 0
        # 首先计算总共有多少个句子对需要处理
        for node_id, version_dict in self.sentence_nodes.items():

            versions = list(version_dict.keys())
            total_pairs += len(versions) * (len(versions) - 1) // 2

        print(f"总共需要处理 {total_pairs} 个句子对")
        processed_count = len(processed_pairs)

        # 开始处理句子对
        for node_id, version_dict in tqdm(self.sentence_nodes.items()):
            seg, ln = node_id.split('_')
            versions = list(version_dict.keys())

            for i in range(len(versions)):
                for j in range(i + 1, len(versions)):
                    version1 = versions[i]
                    version2 = versions[j]

                    # 构建唯一的句子对标识符
                    pair_id = (f"{version1}_{seg}_{ln}", f"{version2}_{seg}_{ln}")
                    # 确保顺序一致，避免重复处理 (A,B) 和 (B,A)
                    if pair_id[0] > pair_id[1]:
                        pair_id = (pair_id[1], pair_id[0])

                    # 检查是否已经处理过
                    if pair_id in processed_pairs:
                        print(f"已处理过 {pair_id}，跳过")
                        continue

                    node1 = version_dict[version1]
                    node2 = version_dict[version2]

                    text1 = node1['text']
                    text2 = node2['text']

                    # 处理特殊字符和乱序情况
                    if '%' in text1 or '%' in text2:
                        adjusted_distance, used_cache = self._handle_disordered_sentences(
                            version1, version2, seg, ln, text1, text2, cache_agent_result=cache_agent_result, do_length_normalization=
                            do_length_normalization, use_agent_analysis=use_agent_analysis
                        )
                    else:
                        try:
                            adjusted_distance, _, _, _, used_cache = calculate_adjusted_edit_distance(
                                text1, text2, self.agent, cache_agent_result=cache_agent_result, do_length_normalization=do_length_normalization
                                , use_agent_analysis=use_agent_analysis
                            )
                            if used_cache:
                                cache_hits += 1
                            else:
                                cache_miss += 1
                                #print(f"缓存未命中: {version1}_{seg}_{ln} ↔ {version2}_{seg}_{ln}")
                        except Exception as e:
                            print(f"计算相似度时出错 ({version1}↔{version2} seg{seg}ln{ln}): {e}")
                            # 即使出错也标记为已处理，避免重复尝试
                            processed_pairs.add(pair_id)
                            continue

                    similarity = round(max(0., 1. - adjusted_distance), 3)
                    assert similarity >= 0 and similarity <= 1, (adjusted_distance, similarity)

                    # 添加相似边
                    if similarity >= similarity_threshold:
                        assert node1['node_id'] in self.G.nodes, f"node1 {node1['node_id']} not in graph"
                        assert node2['node_id'] in self.G.nodes, f"node2 {node2['node_id']} not in graph"
                        self.G.add_edge(
                            node1['node_id'],
                            node2['node_id'],
                            relation="similar",
                            similarity=similarity,
                            adjusted_distance=round(adjusted_distance, 3),
                            version1=version1,
                            version2=version2,
                            seg=seg,
                            ln=ln
                        )
                        similarity_edges_count += 1

                    # 标记为已处理
                    processed_pairs.add(pair_id)
                    processed_count += 1

                    # 定期保存缓存
                    current_time = time.time()
                    # 每处理 save_interval 个句子对或每30秒保存一次缓存
                    if processed_count % save_interval == 0 or current_time - last_save_time > 30 and use_agent_analysis:
                        self._save_progress(cache_file, cache_agent_result)
                        last_save_time = current_time
                        elapsed_time = current_time - start_time
                        pairs_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
                        remaining_pairs = total_pairs - processed_count
                        remaining_time = remaining_pairs / pairs_per_second if pairs_per_second > 0 else 0

                        print(f"已处理: {processed_count}/{total_pairs} ({processed_count / total_pairs * 100:.2f}%), "
                              f"预计剩余时间: {remaining_time / 60:.2f}分钟", flush= True)
                        # if cache_hits + cache_miss > 0:
                        #     self.print_hit_rate(cache_hits, cache_miss)

        # 处理完成后，保存最终缓存
        if use_agent_analysis:
            self._save_progress(cache_file, cache_agent_result, final=True)
        print(f"相似度计算完成，共添加 {similarity_edges_count} 条相似边")
        # if cache_hits + cache_miss > 0:
        #     self.print_hit_rate(cache_hits, cache_miss)
    @staticmethod
    def print_hit_rate(cache_hits, cache_miss):
        final_hit_rate = (cache_hits / (cache_hits + cache_miss)) * 100
        print(f"\n缓存使用最终统计:")
        print(f"- 缓存命中: {cache_hits} 次")
        print(f"- 缓存未命中: {cache_miss} 次")
        print(f"- 缓存命中率: {final_hit_rate:.1f}%")

    def _save_progress(self, cache_file, cache_agent_result, final=False):
        """
        保存当前处理进度到缓存文件，只有在有新内容时才覆盖原文件
        """
        try:
            # 检查是否存在旧的缓存文件
            has_new_content = True  # 默认为有新内容
            if os.path.exists(cache_file):
                try:
                    # 加载旧的缓存数据
                    with open(cache_file, 'rb') as f:
                        old_cache = pickle.load(f)
                        # 比较是否有新内容
                        # 获取旧缓存中的结果
                    old_cache_result = old_cache.get('cache_agent_result', {})

                    # 比较key集合来确定是否有新内容
                    current_keys = set(cache_agent_result.keys())
                    old_keys = set(old_cache_result.keys())

                    # 检查是否有新的key添加
                    new_keys = current_keys.difference(old_keys)
                    # 如果没有新key且没有修改的值，则认为没有新内容
                    if not new_keys:
                        has_new_content = False
                        print(f"缓存未更新: 没有新的结果添加或修改")
                    else:
                        # 记录更新情况
                        has_new_content = True
                        if new_keys:
                            print(f"检测到{len(new_keys)}个新的缓存结果")
                except Exception as e:
                    print(f"读取旧缓存失败，将创建新缓存: {e}")
                    has_new_content = True  # 读取失败时仍创建新缓存

            # 只有在有新内容或final=True时才保存
            if has_new_content or final:
                # 构建缓存数据
                cache_data = {
                    'timestamp': datetime.now().isoformat(),
                    'cache_agent_result': cache_agent_result,
                    'final': final
                }

                # 保存到临时文件，然后重命名，防止写入过程中断导致文件损坏
                temp_file = f"{cache_file}.tmp"
                with open(temp_file, 'wb') as f:
                    pickle.dump(cache_data, f)

                # 原子性替换文件
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                os.rename(temp_file, cache_file)

                status = "最终" if final else "中间"
                print(f"{status}缓存已保存到 {cache_file}，包含{len(cache_agent_result)} 条数据")

                # 保存JSON格式的大模型结果
                json_file = cache_file.replace('.pkl', '.json')
                # 转换为JSON可序列化的格式
                json_serializable_result = {}
                for key, value in cache_agent_result.items():
                    # 将元组键转换为字符串
                    if isinstance(key, tuple):
                        str_key = f"({key[0]},{key[1]})"
                    else:
                        str_key = str(key)

                    # 将对象值转换为字典
                    if hasattr(value, '__dict__'):
                        # 对于有__dict__的对象，转换其属性为字典
                        json_serializable_result[str_key] = vars(value)
                    elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
                        # 已经是可序列化类型
                        json_serializable_result[str_key] = value
                    else:
                        # 其他类型转换为字符串
                        json_serializable_result[str_key] = str(value)

                # 构建JSON数据
                json_data = {
                    'timestamp': datetime.now().isoformat(),
                    'cache_agent_result': json_serializable_result,
                    'total_entries': len(cache_agent_result),
                    'final': final
                }
                # 保存JSON文件
                temp_json_file = f"{json_file}.tmp"
                with open(temp_json_file, 'w', encoding='utf-8') as f:
                    # 使用indent=2使JSON文件更易读
                    json.dump(json_data, f, ensure_ascii=False, indent=2)

                # 原子性替换JSON文件
                if os.path.exists(json_file):
                    os.remove(json_file)
                os.rename(temp_json_file, json_file)

                status = "最终" if final else "中间"
                print(f"{status}JSON数据已保存到 {json_file}，包含{len(json_serializable_result)} 条数据")

        except Exception as e:
            print(f"保存缓存失败: {e}")

    def _handle_disordered_sentences(self, version1, version2, seg, ln, text1, text2, cache_agent_result=None, do_length_normalization=False, use_agent_analysis=True):
        """处理带有'%'标记的乱序句子"""
        clean_text1 = text1.replace('%', '')
        clean_text2 = text2.replace('%', '')

        version1_sentences = self._get_sentences_in_segment(version1, seg)
        version2_sentences = self._get_sentences_in_segment(version2, seg)
        used_cache = False

        if version1_sentences is None or version2_sentences is None:
            try:
                adjusted_distance, _, _, _, used_cache = calculate_adjusted_edit_distance(
                    clean_text1, clean_text2, self.agent, cache_agent_result=cache_agent_result, do_length_normalization=do_length_normalization, use_agent_analysis=use_agent_analysis)
                return adjusted_distance, used_cache
            except Exception:
                return 1.0, used_cache

        max_ln = max(max(version1_sentences.keys()), max(version2_sentences.keys()))
        distances = []

        # 两个句子都有"%"的情况
        if '%' in text1 and '%' in text2:
            try:
                direct_distance, _, _, _, used_cache = calculate_adjusted_edit_distance(
                    clean_text1, clean_text2, self.agent, cache_agent_result=cache_agent_result, do_length_normalization=do_length_normalization, use_agent_analysis=use_agent_analysis)
                distances.append(direct_distance)
            except Exception:
                return 1.0, used_cache

        # version1有"%"的情况
        if '%' in text1:
            version2_candidates = self._get_candidate_sentences(version2, seg, ln, max_ln)
            for candidate_ln, candidate_text in version2_candidates:
                clean_candidate = candidate_text.replace('%', '')
                try:
                    distance, _, _, _, used_cache = calculate_adjusted_edit_distance(
                        clean_text1, clean_candidate, self.agent, cache_agent_result=cache_agent_result, do_length_normalization=do_length_normalization, use_agent_analysis=use_agent_analysis)
                    distances.append(distance)
                except Exception:
                    return 1.0, used_cache

        # version2有"%"的情况
        if '%' in text2:
            version1_candidates = self._get_candidate_sentences(version1, seg, ln, max_ln)
            for candidate_ln, candidate_text in version1_candidates:
                clean_candidate = candidate_text.replace('%', '')
                try:
                    distance, _, _, _, used_cache = calculate_adjusted_edit_distance(
                        clean_text2, clean_candidate, self.agent, cache_agent_result=cache_agent_result, do_length_normalization=do_length_normalization, use_agent_analysis=use_agent_analysis)
                    distances.append(distance)
                except Exception:
                    return 1.0, used_cache

        if not distances:
            try:
                adjusted_distance, _, _, _, used_cache = calculate_adjusted_edit_distance(
                    clean_text1, clean_text2, self.agent, cache_agent_result=cache_agent_result, do_length_normalization=do_length_normalization, use_agent_analysis=use_agent_analysis)
                return adjusted_distance
            except Exception:
                return 1.0, used_cache

        return min(distances), used_cache

    def _get_candidate_sentences(self, target_version, seg, start_ln, max_ln):
        """获取候选句子"""
        candidates = []
        normal_count = 0
        current_ln = start_ln

        while current_ln <= max_ln and normal_count < 2:
            target_sentence_info = self._get_sentence_by_version_seg_ln(target_version, seg, current_ln)
            if not target_sentence_info:
                break

            target_text = target_sentence_info['text']
            candidates.append((current_ln, target_text))

            if '%' not in target_text:
                normal_count += 1
            else:
                normal_count = 0

            current_ln += 1

        return candidates

    def _get_sentence_by_version_seg_ln(self, version, seg, ln):
        """获取特定句子"""
        if (version in self.version_sentences and
                seg in self.version_sentences[version] and
                ln in self.version_sentences[version][seg]):
            return self.version_sentences[version][seg][ln]
        return None

    def _get_sentences_in_segment(self, version, seg):
        """获取章节中的所有句子"""
        if version in self.version_sentences and seg in self.version_sentences[version]:
            return self.version_sentences[version][seg]
        return None

    def save_to_pickle(self, filename="book_graph.pkl"):
        """保存图到pkl文件"""
        try:
            graph_data = {
                'graph': self.G,
                'sentence_nodes': self.sentence_nodes,
                'version_sentences': self.version_sentences,
                'versions': self.versions,
                'graph_info': self.get_graph_info()
            }

            with open(filename, 'wb') as f:
                pickle.dump(graph_data, f)

            print(f"图数据已成功保存到: {filename}")
            print("图信息:")
            for key, value in graph_data['graph_info'].items():
                print(f"  {key}: {value}")
            return filename

        except Exception as e:
            print(f"保存pkl文件时出错: {e}")
            return None

    def load_from_pickle(self, filename="book_graph.pkl"):
        """从pkl文件加载图"""
        try:
            with open(filename, 'rb') as f:
                graph_data = pickle.load(f)

            self.G = graph_data['graph']
            self.sentence_nodes = graph_data['sentence_nodes']
            self.version_sentences = graph_data['version_sentences']
            self.versions = graph_data['versions']

            print(f"图数据已从 {filename} 成功加载")
            return True

        except Exception as e:
            print(f"加载pkl文件时出错: {e}")
            return False

    def get_graph_info(self):
        """获取图的基本信息"""
        return {
            "节点数量": len(self.G.nodes),
            "边数量": len(self.G.edges),
            "版本数量": len(self.versions),
            "句子节点数量": len([n for n, attr in self.G.nodes(data=True) if attr.get('type') == 'statement']),
            "相似边数量": len([e for e in self.G.edges(data=True) if e[2].get('relation') == 'similar'])
        }

    def print_similar_pairs(self):
        """打印相似句子对"""
        print("跨版本相似句子对：")
        similar_edges = [(u, v, d) for u, v, d in self.G.edges(data=True) if d.get("relation") == "similar"]
        for u, v, d in similar_edges:
            print(f"{u} 与 {v} 相似度：{d['similarity']}")


# 修改 build_and_save_graph 函数，支持传递缓存参数
def build_and_save_graph(data_file='CleanData.xlsx', similarity_threshold=0.3, output_file='book_graph.pkl',
                         cache_file="similarity_cache.pkl", resume_from_cache=False, save_interval=50, include_empty_nodes=False, do_length_normalization=False, use_all_version=True, use_agent_analysis=True):
    """
    构建图并保存为pkl文件的主函数
    """
    # 创建agent实例
    agent_instance = Agents()

    # 读取数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, data_file)

    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        print(f"成功读取数据文件: {data_file}")
        print(f"数据形状: {df.shape}")
    except Exception as e:
        print(f"读取数据文件时出错: {e}")
        return None

    # 构建图
    print("开始构建图...")
    graph = BookGraph(agent_instance=agent_instance, use_all_version=use_all_version)

    # 如果需要恢复进度，检查是否存在缓存文件
    if resume_from_cache and not os.path.exists(cache_file):
        print(f"未找到缓存文件 {cache_file}，将从头开始构建")
        resume_from_cache = False

    graph.build_from_dataframe(
        df,
        similarity_threshold,
        cache_file=cache_file,
        resume_from_cache=resume_from_cache,
        save_interval=save_interval,
        do_length_normalization=do_length_normalization,
        include_empty_nodes=include_empty_nodes,
        use_agent_analysis=use_agent_analysis
    )

    # 打印图信息
    graph_info = graph.get_graph_info()
    print("\n图构建完成，基本信息:")
    for key, value in graph_info.items():
        print(f"  {key}: {value}")

    # 打印相似句子对
    graph.print_similar_pairs()

    # 保存为pkl文件
    print(f"\n正在保存图到 {output_file}...")
    saved_file = graph.save_to_pickle(output_file)

    if saved_file:
        print(f"图已成功保存到: {saved_file}")
        return graph
    else:
        print("保存图失败")
        return None


def load_graph_from_pickle(pkl_file='book_graph.pkl'):
    """
    从pkl文件加载图
    
    参数:
        pkl_file: pkl文件名
    """
    graph = BookGraph()
    if graph.load_from_pickle(pkl_file):
        print("图加载成功")
        graph_info = graph.get_graph_info()
        print("图基本信息:")
        for key, value in graph_info.items():
            print(f"  {key}: {value}")
        return graph
    else:
        print("图加载失败")
        return None


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='构建文本变体图网络')

    # 添加命令行参数
    parser.add_argument('--data_file', type=str, default='CleanData.xlsx',
                        help='数据文件路径 (默认: CleanData.xlsx)')
    parser.add_argument('--similarity_threshold', type=float, default=0.3,
                        help='相似度阈值 (默认: 0.3)')
    parser.add_argument('--output_file', type=str, default='book_graph.pkl',
                        help='输出的图数据文件路径 (默认: book_graph.pkl)')
    parser.add_argument('--cache_file', type=str, default='similarity_cache.pkl',
                        help='相似度计算缓存文件路径 (默认: similarity_cache.pkl)')
    parser.add_argument('--resume_from_cache', action='store_true', default=True,
                        help='从缓存恢复进度 (默认: True)')
    parser.add_argument('--no_resume_from_cache', action='store_false', dest='resume_from_cache',
                        help='不从缓存恢复进度')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='缓存保存间隔 (默认: 50)')
    # 添加控制是否包含空文本节点的参数
    parser.add_argument('--include_empty_nodes', action='store_true',
                        help='包含空文本的语句节点 (默认: False)')
    parser.add_argument('--do_normalize_length', action='store_true',
                        help='按照文本长度进行归一化 (默认: False)')
    parser.add_argument('--use_all_version', action='store_true', default=True,
                        help='使用所有版本 (默认: True)')
    parser.add_argument('--no_use_all_version', action='store_false', dest='use_all_version',
                        help='不使用所有版本')
    parser.add_argument('--use_agent_analysis', action='store_true', default=True,
                        help='使用大模型分析编辑距离 (默认: True)')
    parser.add_argument('--no_use_agent_analysis', action='store_false',
                        dest='use_agent_analysis',
                        help='不使用大模型分析编辑距离 (默认: 使用)')

    # 解析命令行参数
    args = parser.parse_args()

    # 打印参数信息
    print(f"运行参数:")
    print(f"  数据文件: {args.data_file}")
    print(f"  相似度阈值: {args.similarity_threshold}")
    print(f"  输出文件: {args.output_file}")
    print(f"  缓存文件: {args.cache_file}")
    print(f"  恢复进度: {args.resume_from_cache}")
    print(f"  保存间隔: {args.save_interval}")
    print(f"  归一化长度: {args.do_normalize_length}")
    print(f"  使用所有版本: {args.use_all_version}")
    print(f"  包含空文本节点: {args.include_empty_nodes}")
    print(f"  使用大模型分析编辑距离: {args.use_agent_analysis}")

    # 使用命令行参数构建图并保存
    graph = build_and_save_graph(
        data_file=args.data_file,
        similarity_threshold=args.similarity_threshold,
        output_file=args.output_file,
        cache_file=args.cache_file,
        resume_from_cache=args.resume_from_cache,
        save_interval=args.save_interval,
        do_length_normalization=args.do_normalize_length,
        use_all_version=args.use_all_version,
        include_empty_nodes=args.include_empty_nodes,
        use_agent_analysis=args.use_agent_analysis
    )

    if graph:
        print("\n图构建和保存成功完成！")

        # 可选：测试加载功能
        print("\n测试加载功能...")
        loaded_graph = load_graph_from_pickle(args.output_file)
    else:
        print("图构建失败")
