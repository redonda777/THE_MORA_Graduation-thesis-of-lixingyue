import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class TokenState:
    char: str
    source_position: Optional[int]
    is_inserted: bool = False
    was_replaced: bool = False


class LLMRelationClient:
    """规则判定后的大模型补判客户端。"""

    def __init__(self, use_llm: bool, llm_config: Optional[Dict[str, Any]] = None):
        self.config = llm_config or {}
        self.enabled = bool(use_llm)
        self.provider = str(self.config.get("provider", "dashscope"))
        self.model = str(self.config.get("model", "qwen-turbo-latest"))
        self.base_url = str(self.config.get("base_url", "")).strip()
        self.api_key_env = str(self.config.get("api_key_env", "")).strip()
        self.api_key_direct = str(self.config.get("api_key", "")).strip()
        self.timeout_seconds = int(self.config.get("timeout_seconds", 45))
        self.max_retries = int(self.config.get("max_retries", 2))
        self.retry_sleep_seconds = float(self.config.get("retry_sleep_seconds", 1.0))
        self.retry_backoff_multiplier = float(self.config.get("retry_backoff_multiplier", 2.0))
        self.retry_max_sleep_seconds = float(self.config.get("retry_max_sleep_seconds", 8.0))
        self.request_interval_seconds = float(self.config.get("request_interval_seconds", 0.0))
        self.temperature = float(self.config.get("temperature", 0.0))
        self.max_tokens = int(self.config.get("max_tokens", 200))
        self.cache: Dict[str, Tuple[str, str]] = {}
        self.last_request_ts = 0.0
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "success": 0,
            "failures": 0,
            "failure_types": {},
        }

        if self.enabled:
            self._validate_config()
            self.api_key = self._resolve_api_key()
            if not self.api_key:
                raise ValueError(
                    "已启用 --use-llm，但未获取到 API Key。请在 llm_config.json 填 api_key，"
                    "或将 api_key_env 设为环境变量名并在系统环境中设置。"
                )
        else:
            self.api_key = ""

    def _validate_config(self) -> None:
        if not self.base_url:
            raise ValueError("llm_config.json 缺少 base_url")
        if self.provider not in ("dashscope", "openai_compatible"):
            raise ValueError("provider 仅支持 dashscope / openai_compatible")
        if self.request_interval_seconds < 0:
            raise ValueError("request_interval_seconds 不能小于 0")
        if self.retry_sleep_seconds < 0:
            raise ValueError("retry_sleep_seconds 不能小于 0")
        if self.retry_backoff_multiplier < 1.0:
            raise ValueError("retry_backoff_multiplier 不能小于 1.0")
        if self.retry_max_sleep_seconds <= 0:
            raise ValueError("retry_max_sleep_seconds 必须大于 0")

    def _resolve_api_key(self) -> str:
        # 1) 显式 api_key 字段（优先）
        if self.api_key_direct:
            return self.api_key_direct
        # 2) api_key_env 被误填成 sk-... 时，按直填 key 处理
        if self.api_key_env.startswith("sk-"):
            return self.api_key_env
        # 3) api_key_env 作为环境变量名
        if self.api_key_env:
            return os.getenv(self.api_key_env, "").strip()
        # 4) 最后兜底常见变量
        return os.getenv("DASHSCOPE_API_KEY", "").strip()

    @staticmethod
    def _safe_json_loads(raw: str) -> Dict[str, Any]:
        text = raw.strip()
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            loaded = json.loads(text[start : end + 1])
            if isinstance(loaded, dict):
                return loaded
        raise ValueError("模型返回内容不是可解析的 JSON 对象")

    @staticmethod
    def _extract_message_text(message_content: Any) -> Optional[str]:
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            chunks: List[str] = []
            for item in message_content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str):
                        chunks.append(txt)
            if chunks:
                return "\n".join(chunks)
        return None

    def _throttle_if_needed(self) -> None:
        if self.request_interval_seconds <= 0:
            return
        now = time.time()
        if self.last_request_ts <= 0:
            return
        elapsed = now - self.last_request_ts
        remain = self.request_interval_seconds - elapsed
        if remain > 0:
            time.sleep(remain)

    def _record_failure(self, err: Exception) -> None:
        self.stats["failures"] += 1
        key = type(err).__name__
        if hasattr(err, "response") and getattr(err, "response", None) is not None:
            status_code = getattr(err.response, "status_code", None)
            if status_code is not None:
                key = f"{key}:{status_code}"
        failure_types = self.stats["failure_types"]
        failure_types[key] = failure_types.get(key, 0) + 1

    def _dashscope_request(self, prompt: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": {"messages": [{"role": "user", "content": prompt}]},
            "parameters": {
                "result_format": "text",
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        }
        self._throttle_if_needed()
        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout_seconds)
        self.last_request_ts = time.time()
        resp.raise_for_status()
        data = resp.json()
        output = data.get("output", {})
        content = None
        # DashScope 兼容两种返回结构：
        # 1) output.text
        # 2) output.choices[0].message.content
        if isinstance(output, dict):
            text_val = output.get("text")
            if isinstance(text_val, str):
                content = text_val
            else:
                choices = output.get("choices")
                if isinstance(choices, list) and choices:
                    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                    content = self._extract_message_text(message.get("content"))
        if not isinstance(content, str):
            raise ValueError(f"DashScope 返回结构不符合预期: {json.dumps(data, ensure_ascii=False)[:300]}")
        return self._safe_json_loads(content)

    def _openai_compatible_request(self, prompt: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        self._throttle_if_needed()
        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout_seconds)
        self.last_request_ts = time.time()
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        content = self._extract_message_text(message.get("content"))
        if not isinstance(content, str):
            raise ValueError("OpenAI 兼容接口返回 content 非字符串")
        return self._safe_json_loads(content)

    def _request_model(self, prompt: str) -> Dict[str, Any]:
        if self.provider == "dashscope":
            return self._dashscope_request(prompt)
        return self._openai_compatible_request(prompt)

    def classify_operation(
        self,
        op_type: str,
        original_text: str,
        modified_text: str,
        op: Dict[str, Any],
    ) -> Tuple[str, str]:
        if op_type == "replace":
            allowed_relations = ["interchangeable", "homophone", "synonym", "default"]
            op_desc = (
                f"操作: replace, original_char={op['original_char']}, "
                f"target_char={op['target_char']}, position={op['position']}"
            )
        else:
            allowed_relations = ["function_word", "default"]
            op_desc = f"操作: {op_type}, char={op['char']}, position={op['position']}"

        cache_key = json.dumps(
            {
                "op_type": op_type,
                "original_text": original_text,
                "modified_text": modified_text,
                "op": op,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[cache_key]

        prompt = (
            "你是古文校勘专家。请判断下述编辑操作的字符关系类型。\n"
            "只允许从 allowed_relations 中选一个。\n"
            "如果不确定，必须返回 default。\n"
            f"allowed_relations: {allowed_relations}\n"
            f"原句: {original_text}\n"
            f"目标句: {modified_text}\n"
            f"{op_desc}\n"
            "只输出 JSON 对象，格式为：\n"
            '{"relation":"<allowed_relations之一>","reason":"<一句简短理由>"}'
        )

        self.stats["requests"] += 1
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                result = self._request_model(prompt)
                relation = str(result.get("relation", "default")).strip()
                reason = str(result.get("reason", "LLM 未提供原因")).strip()
                if relation not in allowed_relations:
                    relation = "default"
                    reason = "LLM 返回关系不在允许集合，降级 default"
                self.cache[cache_key] = (relation, reason)
                self.stats["success"] += 1
                return relation, reason
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    sleep_seconds = min(
                        self.retry_max_sleep_seconds,
                        self.retry_sleep_seconds * (self.retry_backoff_multiplier ** attempt),
                    )
                    time.sleep(sleep_seconds)
                else:
                    self._record_failure(e)
        fallback = ("default", f"LLM 调用失败，降级 default: {last_error}")
        self.cache[cache_key] = fallback
        return fallback


def is_private_char(ch: str) -> bool:
    return "\uE000" <= ch <= "\uF8FF"


def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_pair_sets(config: Dict[str, Any]) -> Dict[str, set]:
    lexicons = config.get("pair_lexicons", {})
    pair_sets: Dict[str, set] = {}
    for name in ("interchangeable", "homophone", "synonym"):
        pair_sets[name] = {
            canonical_pair(str(x[0]), str(x[1]))
            for x in lexicons.get(name, [])
            if isinstance(x, list) and len(x) == 2
        }
    return pair_sets


def normalize_operation(raw_op: Dict[str, Any]) -> Dict[str, Any]:
    op_type = raw_op.get("type")
    if op_type not in ("replace", "insert", "delete"):
        raise ValueError(f"不支持的操作类型: {op_type}")

    if "position" not in raw_op:
        raise ValueError(f"操作缺少 position 字段: {raw_op}")

    position = int(raw_op["position"])
    if position < 0:
        raise ValueError(f"position 不能为负数: {raw_op}")

    normalized: Dict[str, Any] = {"type": op_type, "position": position}
    if op_type == "replace":
        normalized["original_char"] = str(raw_op.get("original_char", ""))
        normalized["target_char"] = str(raw_op.get("target_char", ""))
    elif op_type == "insert":
        normalized["char"] = str(raw_op.get("char", ""))
    else:
        normalized["char"] = str(raw_op.get("char", ""))
    return normalized


def classify_replace(
    src_char: str,
    tgt_char: str,
    pair_sets: Dict[str, set],
    missing_chars: set,
) -> Tuple[str, str]:
    if src_char in missing_chars or tgt_char in missing_chars:
        return "missing_char", "包含缺失字符占位符"
    if is_private_char(src_char) or is_private_char(tgt_char):
        return "private_char", "包含私有区字符"

    pair = canonical_pair(src_char, tgt_char)
    if pair in pair_sets["interchangeable"]:
        return "interchangeable", "命中异体/通假关系词表"
    if pair in pair_sets["homophone"]:
        return "homophone", "命中同音/近音关系词表"
    if pair in pair_sets["synonym"]:
        return "synonym", "命中近义关系词表"
    return "default", "未命中特殊关系，使用默认代价"


def classify_insert_delete(ch: str, function_words: set, missing_chars: set) -> Tuple[str, str]:
    if ch in missing_chars:
        return "missing_char", "缺失字符占位符"
    if is_private_char(ch):
        return "private_char", "私有区字符"
    if ch in function_words:
        return "function_word", "命中虚词表"
    return "default", "普通增删操作"


def get_cost(config: Dict[str, Any], op_type: str, relation: str) -> float:
    costs = config.get("costs", {})
    if op_type == "replace":
        return float(costs.get("replace", {}).get(relation, costs.get("replace", {}).get("default", 1.0)))
    if op_type in ("insert", "delete"):
        return float(
            costs.get("insert_delete", {}).get(relation, costs.get("insert_delete", {}).get("default", 1.0))
        )
    return float(costs.get("match", 0.0))


def replay_and_collect_matches(
    original_text: str,
    modified_text: str,
    operations: List[Dict[str, Any]],
) -> Tuple[bool, List[str], str, List[TokenState]]:
    warnings: List[str] = []
    tokens: List[TokenState] = [
        TokenState(char=ch, source_position=idx, is_inserted=False, was_replaced=False)
        for idx, ch in enumerate(original_text)
    ]

    for idx, op in enumerate(operations):
        op_type = op["type"]
        pos = op["position"]
        if op_type == "replace":
            if pos >= len(tokens):
                warnings.append(f"第{idx}步 replace 越界，position={pos}，当前长度={len(tokens)}")
                return False, warnings, "".join(t.char for t in tokens), tokens
            expected = op["original_char"]
            if expected and tokens[pos].char != expected:
                warnings.append(
                    f"第{idx}步 replace 原字符不一致，期望={expected}，实际={tokens[pos].char}，继续按实际字符执行"
                )
            tokens[pos].char = op["target_char"]
            tokens[pos].was_replaced = True
        elif op_type == "insert":
            if pos > len(tokens):
                warnings.append(f"第{idx}步 insert 越界，position={pos}，当前长度={len(tokens)}")
                return False, warnings, "".join(t.char for t in tokens), tokens
            tokens.insert(pos, TokenState(char=op["char"], source_position=None, is_inserted=True, was_replaced=False))
        else:
            if pos >= len(tokens):
                warnings.append(f"第{idx}步 delete 越界，position={pos}，当前长度={len(tokens)}")
                return False, warnings, "".join(t.char for t in tokens), tokens
            expected = op["char"]
            if expected and tokens[pos].char != expected:
                warnings.append(
                    f"第{idx}步 delete 字符不一致，期望={expected}，实际={tokens[pos].char}，继续按当前位置删除"
                )
            tokens.pop(pos)

    replay_text = "".join(t.char for t in tokens)
    replay_ok = replay_text == modified_text
    if not replay_ok:
        warnings.append(f"回放后文本不一致，replay={replay_text}，target={modified_text}")
    return replay_ok, warnings, replay_text, tokens


def build_match_operations(final_tokens: List[TokenState]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for target_position, token in enumerate(final_tokens):
        if token.source_position is None:
            continue
        if token.is_inserted or token.was_replaced:
            continue
        matches.append(
            {
                "type": "match",
                "char": token.char,
                "position": target_position,
                "source_position": token.source_position,
                "relation": "exact",
                "reason": "字符保持不变",
                "cost": 0.0,
            }
        )
    return matches


def score_operations(
    original_text: str,
    modified_text: str,
    operations: List[Dict[str, Any]],
    config: Dict[str, Any],
    pair_sets: Dict[str, set],
    llm_client: Optional[LLMRelationClient] = None,
) -> Dict[str, Any]:
    missing_chars = set(config.get("missing_chars", []))
    function_words = set(config.get("function_words", []))

    normalized_ops = [normalize_operation(op) for op in operations]
    scored_ops: List[Dict[str, Any]] = []
    relation_counts: Dict[str, int] = {
        "interchangeable": 0,
        "homophone": 0,
        "synonym": 0,
        "function_word": 0,
        "missing_char": 0,
        "private_char": 0,
        "default": 0,
    }

    total_cost = 0.0
    llm_judged_operations = 0
    llm_promoted_operations = 0
    for idx, op in enumerate(normalized_ops):
        op_type = op["type"]
        relation_source = "rule"
        if op_type == "replace":
            relation, reason = classify_replace(op["original_char"], op["target_char"], pair_sets, missing_chars)
            if llm_client and llm_client.enabled and relation == "default":
                llm_relation, llm_reason = llm_client.classify_operation(op_type, original_text, modified_text, op)
                llm_judged_operations += 1
                relation_source = "llm"
                if llm_relation != "default":
                    llm_promoted_operations += 1
                relation = llm_relation
                reason = llm_reason
            cost = get_cost(config, op_type, relation)
            item = {
                "index": idx,
                "type": op_type,
                "original_char": op["original_char"],
                "target_char": op["target_char"],
                "position": op["position"],
                "relation": relation,
                "reason": reason,
                "relation_source": relation_source,
                "cost": cost,
            }
        else:
            relation, reason = classify_insert_delete(op["char"], function_words, missing_chars)
            # 增删操作默认不走LLM补判，以控制成本与时延。
            cost = get_cost(config, op_type, relation)
            item = {
                "index": idx,
                "type": op_type,
                "char": op["char"],
                "position": op["position"],
                "relation": relation,
                "reason": reason,
                "relation_source": relation_source,
                "cost": cost,
            }
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
        total_cost += cost
        scored_ops.append(item)

    replay_ok, replay_warnings, replay_text, final_tokens = replay_and_collect_matches(
        original_text=original_text,
        modified_text=modified_text,
        operations=normalized_ops,
    )
    match_ops = build_match_operations(final_tokens) if replay_ok else []

    all_ops = sorted(
        scored_ops + match_ops,
        key=lambda x: (x.get("position", 10**9), 0 if x["type"] == "match" else 1, x.get("index", -1)),
    )

    max_len = max(len(original_text), len(modified_text), 1)
    adjusted_distance = total_cost
    normalized_distance = adjusted_distance / max_len

    return {
        "operations_scored": scored_ops,
        "match_operations": match_ops,
        "operations_all": all_ops,
        "relation_summary": {
            "relation_counts": relation_counts,
            "match_count": len(match_ops),
            "edit_count": len(scored_ops),
        },
        "distance_breakdown": {
            "edit_operation_cost": round(total_cost, 6),
            "match_cost": 0.0,
            "total_cost": round(adjusted_distance, 6),
        },
        "llm_summary": {
            "llm_enabled": bool(llm_client and llm_client.enabled),
            "llm_judged_operations": llm_judged_operations,
            "llm_promoted_operations": llm_promoted_operations,
        },
        "adjusted_edit_distance": round(adjusted_distance, 6),
        "normalized_distance": round(normalized_distance, 6),
        "validation": {
            "replay_ok": replay_ok,
            "replay_result_text": replay_text,
            "warnings": replay_warnings,
        },
    }


def process_item(
    item: Dict[str, Any],
    config: Dict[str, Any],
    pair_sets: Dict[str, set],
    llm_client: Optional[LLMRelationClient] = None,
) -> Dict[str, Any]:
    original_text = str(item.get("original_text", ""))
    modified_text = str(item.get("modified_text", ""))
    operations = item.get("operations_llm")
    operation_source = "operations_llm"

    used_fallback = False
    if not isinstance(operations, list):
        operations = item.get("operations_dp", [])
        operation_source = "operations_dp"
        used_fallback = True

    if not isinstance(operations, list):
        raise ValueError("输入记录缺少可用操作序列（operations_llm / operations_dp）")

    scored = score_operations(
        original_text=original_text,
        modified_text=modified_text,
        operations=operations,
        config=config,
        pair_sets=pair_sets,
        llm_client=llm_client,
    )

    output = {
        "chapter_number": item.get("chapter_number"),
        "sentence_number": item.get("sentence_number"),
        "original_text_version": item.get("original_text_version"),
        "modified_text_version": item.get("modified_text_version"),
        "original_text": original_text,
        "modified_text": modified_text,
        "operation_source": operation_source,
        "source": {
            "used_fallback_operations": used_fallback,
            "relation_source": "rule_plus_llm" if (llm_client and llm_client.enabled) else "rule_only",
        },
    }
    output.update(scored)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按特殊字符关系对 operations_llm 赋予自定义代价（方案A输出）。")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("connect_edit_distance/llm_edit_distance_0410/formal_all_sentence_edit_distance_llm_v2_merged.json"),
        help="输入 JSON 数组文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("adjus_distance/formal_all_sentence_adjusted_distance_aggressive.json"),
        help="输出 JSON 文件路径",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("adjus_distance/config_aggressive.json"),
        help="配置文件路径（关系词表和代价）",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="启用大模型对 default 关系进行补判",
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=Path("adjus_distance/llm_config.json"),
        help="大模型配置文件路径",
    )
    parser.add_argument("--start", type=int, default=0, help="起始下标（含）")
    parser.add_argument("--end", type=int, default=-1, help="结束下标（不含），-1 表示到末尾")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    pair_sets = build_pair_sets(config)
    llm_config: Dict[str, Any] = {}
    if args.use_llm:
        llm_config = load_json(args.llm_config)
    llm_client = LLMRelationClient(use_llm=args.use_llm, llm_config=llm_config)
    data = load_json(args.input)
    if not isinstance(data, list):
        raise ValueError("输入文件必须是 JSON 数组")

    start = max(args.start, 0)
    end = len(data) if args.end < 0 else min(args.end, len(data))
    if start >= end:
        raise ValueError(f"无有效处理区间: start={start}, end={end}, total={len(data)}")

    outputs: List[Dict[str, Any]] = []
    for idx in range(start, end):
        try:
            outputs.append(process_item(data[idx], config, pair_sets, llm_client=llm_client))
        except Exception as e:
            outputs.append(
                {
                    "index": idx,
                    "error": str(e),
                    "chapter_number": data[idx].get("chapter_number"),
                    "sentence_number": data[idx].get("sentence_number"),
                    "original_text": data[idx].get("original_text"),
                    "modified_text": data[idx].get("modified_text"),
                }
            )

    dump_json(args.output, outputs)
    print(f"已输出 {len(outputs)} 条到 {args.output}")
    if llm_client.enabled:
        print(
            f"LLM统计 requests={llm_client.stats['requests']} "
            f"cache_hits={llm_client.stats['cache_hits']} "
            f"success={llm_client.stats['success']} "
            f"failures={llm_client.stats['failures']}"
        )
        if llm_client.stats["failure_types"]:
            failure_detail = ", ".join(
                f"{k}:{v}" for k, v in sorted(llm_client.stats["failure_types"].items(), key=lambda kv: kv[0])
            )
            print(f"LLM失败类型统计 {failure_detail}")


if __name__ == "__main__":
    main()
