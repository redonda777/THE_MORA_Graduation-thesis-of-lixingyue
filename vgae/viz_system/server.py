from __future__ import annotations

import csv
import json
import mimetypes
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from xml.etree import ElementTree as ET


ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parent
STATIC_DIR = PROJECT_DIR / "static"
GVNM_DIR = ROOT_DIR / "gvnm" / "output"
SIMILARITY_JSON = ROOT_DIR / "total_formal_all_sentence_adjusted_distance_aggressive_llm.json"
ORIGINAL_XLSX = PROJECT_DIR / "Mora.xlsx"
XML_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


@dataclass
class SentenceRecord:
    version: str
    chapter: int
    sentence: int
    text: str


class DataStore:
    def __init__(self) -> None:
        self.versions: list[str] = []
        self.chapters: list[int] = []
        self.sentences_by_chapter: dict[int, dict[str, list[SentenceRecord]]] = {}
        self.version_communities: dict[str, int] = {}
        self.version_community_groups: list[list[str]] = []
        self.version_ranking: list[dict[str, Any]] = []
        self.version_edges: list[dict[str, Any]] = []
        self.chapter_membership: dict[int, dict[str, int]] = {}
        self.chapter_edges: dict[int, list[dict[str, Any]]] = {}
        self.sentence_similarity_edges: dict[int, list[dict[str, Any]]] = {}
        self.original_text_by_chapter: dict[int, dict[str, dict[int, str]]] = {}
        self.book_affinity: dict[str, dict[str, float | None]] = {}
        self.book_affinity_sources: dict[str, dict[str, str]] = {}
        self.summary: dict[str, Any] = {}
        self._load_all()

    def _load_all(self) -> None:
        self._load_tree(ROOT_DIR / "mora_v4.1_0406.json")
        self._load_original_text(ORIGINAL_XLSX)
        self._load_version_communities(GVNM_DIR / "version_communities.json")
        self._load_version_ranking(GVNM_DIR / "version_pair_similarity_ranking.csv")
        self._load_chapter_membership(GVNM_DIR / "chapter_community_membership.csv")
        self._load_chapter_edges(GVNM_DIR / "chapter_affinity_edges.csv")
        self._load_sentence_similarity_edges(SIMILARITY_JSON)
        self._load_book_affinity(GVNM_DIR / "book_affinity_matrix.csv")
        self._load_book_affinity_sources(GVNM_DIR / "book_affinity_sources.csv")
        self._load_summary(GVNM_DIR / "community_detection_summary.json")

    def _load_tree(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as f:
            root = json.load(f)
        chapters = set()
        for version_node in root.get("children", []):
            if version_node.get("type") != "version":
                continue
            version = str(version_node.get("name") or version_node.get("version"))
            self.versions.append(version)
            for chapter_node in version_node.get("children", []):
                if chapter_node.get("type") != "chapter":
                    continue
                chapter = int(chapter_node["chapter_number"])
                chapters.add(chapter)
                version_bucket = self.sentences_by_chapter.setdefault(chapter, {})
                records = version_bucket.setdefault(version, [])
                for sentence_node in chapter_node.get("children", []):
                    if sentence_node.get("type") != "sentence":
                        continue
                    records.append(
                        SentenceRecord(
                            version=version,
                            chapter=chapter,
                            sentence=int(sentence_node["sentence_number"]),
                            text=str(sentence_node.get("text", "")).replace("#", "□"),
                        )
                    )
                records.sort(key=lambda item: item.sentence)
        self.chapters = sorted(chapters)

    def _xlsx_shared_strings(self, archive: zipfile.ZipFile) -> list[str]:
        try:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        except KeyError:
            return []
        return ["".join(item.itertext()) for item in root.findall("main:si", XML_NS)]

    def _xlsx_sheet_path(self, archive: zipfile.ZipFile, sheet_name: str) -> str:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels.findall("pkgrel:Relationship", XML_NS)
        }
        for sheet in workbook.findall("main:sheets/main:sheet", XML_NS):
            if sheet.attrib.get("name") != sheet_name:
                continue
            rel_id = sheet.attrib.get(f"{{{XML_NS['rel']}}}id")
            target = rel_targets.get(str(rel_id), "")
            if target.startswith("/"):
                return target.lstrip("/")
            return f"xl/{target}" if not target.startswith("xl/") else target
        raise KeyError(f"sheet {sheet_name} not found")

    def _xlsx_cell_text(self, cell: ET.Element, shared_strings: list[str]) -> str:
        cell_type = cell.attrib.get("t")
        if cell_type == "inlineStr":
            return "".join(cell.itertext())
        value = cell.find("main:v", XML_NS)
        if value is None or value.text is None:
            return ""
        if cell_type == "s":
            try:
                return shared_strings[int(value.text)]
            except (ValueError, IndexError):
                return ""
        if cell_type == "b":
            return "TRUE" if value.text == "1" else "FALSE"
        return value.text

    def _xlsx_column_index(self, cell_ref: str) -> int:
        letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
        index = 0
        for letter in letters:
            index = index * 26 + (ord(letter) - ord("A") + 1)
        return index - 1

    def _xlsx_rows(self, path: Path, sheet_name: str) -> list[list[str]]:
        with zipfile.ZipFile(path) as archive:
            shared_strings = self._xlsx_shared_strings(archive)
            sheet_path = self._xlsx_sheet_path(archive, sheet_name)
            root = ET.fromstring(archive.read(sheet_path))
            rows: list[list[str]] = []
            for row in root.findall("main:sheetData/main:row", XML_NS):
                values: dict[int, str] = {}
                for cell in row.findall("main:c", XML_NS):
                    ref = cell.attrib.get("r", "")
                    if not ref:
                        continue
                    values[self._xlsx_column_index(ref)] = self._xlsx_cell_text(
                        cell, shared_strings
                    )
                if not values:
                    rows.append([])
                    continue
                width = max(values) + 1
                rows.append([values.get(index, "") for index in range(width)])
        return rows

    def _is_original_text_value(self, value: str) -> bool:
        text = str(value or "").strip()
        if not text or text == "~":
            return False
        return re.fullmatch(r"\d+", text) is None

    def _load_original_text(self, path: Path) -> None:
        if not path.exists():
            return
        rows = self._xlsx_rows(path, "Sheet4")
        if not rows:
            return
        header = [str(item).strip() for item in rows[0]]
        version_columns = {
            version: header.index(version)
            for version in self.versions
            if version in header
        }
        if "seg" not in header or "ln" not in header:
            return
        chapter_col = header.index("seg")
        sentence_col = header.index("ln")
        raw_lookup: dict[tuple[int, int, str], str] = {}
        for row in rows[1:]:
            try:
                chapter = int(float(row[chapter_col]))
                sentence = int(float(row[sentence_col]))
            except (IndexError, TypeError, ValueError):
                continue
            for version, col_index in version_columns.items():
                raw_value = row[col_index] if col_index < len(row) else ""
                if not self._is_original_text_value(raw_value):
                    continue
                raw_lookup[(chapter, sentence, version)] = str(raw_value).strip()

        for chapter, version_bucket in self.sentences_by_chapter.items():
            for version, records in version_bucket.items():
                for record in records:
                    raw_text = raw_lookup.get((chapter, record.sentence, version))
                    if not raw_text:
                        continue
                    self.original_text_by_chapter.setdefault(chapter, {}).setdefault(
                        version, {}
                    )[record.sentence] = raw_text

    def _load_version_communities(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.version_community_groups = payload.get("communities", [])
        for community_id, versions in enumerate(self.version_community_groups, start=1):
            for version in versions:
                self.version_communities[version] = community_id

    def _load_version_ranking(self, path: Path) -> None:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                item = {
                    "rank": int(row["rank"]),
                    "source": row["version_a"],
                    "target": row["version_b"],
                    "similarity": float(row["similarity"]),
                    "distance": float(row["distance_1_minus_similarity"]),
                }
                self.version_ranking.append(item)
                self.version_edges.append(item)

    def _load_chapter_membership(self, path: Path) -> None:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                chapter = int(row["chapter"])
                self.chapter_membership.setdefault(chapter, {})[row["version"]] = int(
                    row["community_id"]
                )

    def _load_chapter_edges(self, path: Path) -> None:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                chapter = int(row["chapter"])
                self.chapter_edges.setdefault(chapter, []).append(
                    {
                        "source": row["version_a"],
                        "target": row["version_b"],
                        "similarity": float(row["similarity"]),
                        "support": int(row["supporting_sentence_pairs"]),
                    }
                )
        for edges in self.chapter_edges.values():
            edges.sort(key=lambda item: item["similarity"], reverse=True)

    def _sentence_node_id(self, version: str, chapter: int, sentence: int) -> str:
        return f"sent_{version}_{chapter}_{sentence}"

    def _chapter_node_id(self, version: str, chapter: int) -> str:
        return f"chap_{version}_{chapter}"

    def _sentence_exists(self, version: str, chapter: int, sentence: int) -> bool:
        return any(
            record.sentence == sentence
            for record in self.sentences_by_chapter.get(chapter, {}).get(version, [])
        )

    def _load_sentence_similarity_edges(self, path: Path) -> None:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            raw_edges = json.load(f)
        seen: set[tuple[str, str]] = set()
        for edge in raw_edges:
            try:
                chapter = int(edge["chapter_number"])
                sentence = int(edge["sentence_number"])
                source_version = str(edge["original_text_version"])
                target_version = str(edge["modified_text_version"])
                distance = float(edge["normalized_distance"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (
                self._sentence_exists(source_version, chapter, sentence)
                and self._sentence_exists(target_version, chapter, sentence)
            ):
                continue
            source = self._sentence_node_id(source_version, chapter, sentence)
            target = self._sentence_node_id(target_version, chapter, sentence)
            key = tuple(sorted((source, target)))
            if key in seen:
                continue
            seen.add(key)
            self.sentence_similarity_edges.setdefault(chapter, []).append(
                {
                    "source": source,
                    "target": target,
                    "type": "sentence_similarity",
                    "sentence": sentence,
                    "sourceVersion": source_version,
                    "targetVersion": target_version,
                    "distance": distance,
                    "similarity": max(0.0, 1.0 - distance),
                    "originalText": str(edge.get("original_text", "")),
                    "modifiedText": str(edge.get("modified_text", "")),
                }
            )
        for edges in self.sentence_similarity_edges.values():
            edges.sort(key=lambda item: item["distance"])

    def _load_book_affinity(self, path: Path) -> None:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)[1:]
            for row in reader:
                version = row[0]
                values: dict[str, float | None] = {}
                for other, value in zip(header, row[1:]):
                    raw = value.strip()
                    values[other] = None if raw in {"", "NA", "NaN", "null"} else float(raw)
                self.book_affinity[version] = values

    def _load_book_affinity_sources(self, path: Path) -> None:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)[1:]
            for row in reader:
                version = row[0]
                self.book_affinity_sources[version] = {
                    other: value for other, value in zip(header, row[1:])
                }

    def _load_summary(self, path: Path) -> None:
        self.summary = json.loads(path.read_text(encoding="utf-8"))

    def overview(self) -> dict[str, Any]:
        return {
            "versions": self.versions,
            "chapter_count": len(self.chapters),
            "sentence_count": sum(
                len(records)
                for chapter in self.sentences_by_chapter.values()
                for records in chapter.values()
            ),
            "version_communities": self.version_community_groups,
            "top_pairs": self.version_ranking[:10],
            "summary": self.summary,
        }

    def _select_version_graph_edges(
        self,
        threshold: float,
        top_k: int,
    ) -> list[dict[str, Any]]:
        keep: set[tuple[str, str]] = set()
        neighbors: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for edge in self.version_edges:
            pair = tuple(sorted((edge["source"], edge["target"])))
            by_pair[pair] = edge
            if edge["similarity"] >= threshold:
                keep.add(pair)
            neighbors[edge["source"]].append(edge)
            neighbors[edge["target"]].append(edge)

        for version, edges in neighbors.items():
            ranked = sorted(edges, key=lambda item: item["similarity"], reverse=True)
            for edge in ranked[:top_k]:
                keep.add(tuple(sorted((edge["source"], edge["target"]))))

        selected = [{**by_pair[pair], "type": "version_similarity"} for pair in keep]
        selected.sort(key=lambda item: item["rank"])
        return selected

    def version_graph(self, threshold: float = 0.6, top_k: int = 3) -> dict[str, Any]:
        nodes = [
            {
                "id": version,
                "label": version,
                "type": "version",
                "community": self.version_communities.get(version, 0),
                "sentenceCount": sum(
                    len(chapter.get(version, []))
                    for chapter in self.sentences_by_chapter.values()
                ),
            }
            for version in self.versions
        ]
        links = self._select_version_graph_edges(threshold=threshold, top_k=top_k)
        return {
            "nodes": nodes,
            "links": links,
            "edgePolicy": {
                "threshold": threshold,
                "topKPerNode": top_k,
                "edgeCount": len(links),
            },
        }

    def version_ranking_api(self, limit: int = 66) -> dict[str, Any]:
        return {"items": self.version_ranking[:limit]}

    def _normalize_search_text(self, value: str) -> str:
        return re.sub(r"[\s#□]+", "", value).lower()

    def _subsequence_score(self, query: str, text: str) -> float:
        if not query or not text:
            return 0.0
        pos = 0
        matched = 0
        for char in query:
            found = text.find(char, pos)
            if found < 0:
                continue
            matched += 1
            pos = found + 1
        return matched / len(query)

    def _search_score(self, query: str, text: str) -> float:
        if not query:
            return 0.0
        if query in text:
            coverage = len(query) / max(1, len(text))
            return 2.0 + coverage
        return self._subsequence_score(query, text)

    def search_sentences(self, query: str, limit: int = 12) -> dict[str, Any]:
        normalized_query = self._normalize_search_text(query)
        if not normalized_query:
            return {"query": query, "items": []}

        items = []
        for chapter in self.chapters:
            version_bucket = self.sentences_by_chapter.get(chapter, {})
            for version, records in version_bucket.items():
                sentence_count = len(records)
                for record in records:
                    normalized_text = self._normalize_search_text(record.text)
                    score = self._search_score(normalized_query, normalized_text)
                    if score < 0.55:
                        continue
                    items.append(
                        {
                            "chapter": chapter,
                            "chapterDisplay": chapter + 1,
                            "version": version,
                            "sentence": record.sentence,
                            "sentenceDisplay": record.sentence + 1,
                            "sentenceCount": sentence_count,
                            "nodeId": self._sentence_node_id(
                                version, chapter, record.sentence
                            ),
                            "text": record.text,
                            "score": score,
                            "matchType": "contains"
                            if normalized_query in normalized_text
                            else "fuzzy",
                        }
                    )

        items.sort(
            key=lambda item: (
                -item["score"],
                -item["sentenceCount"],
                item["version"],
                item["sentence"],
            )
        )
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank
        return {"query": query, "items": items[:limit], "total": len(items)}

    def chapter_api(self, chapter: int) -> dict[str, Any]:
        version_bucket = self.sentences_by_chapter.get(chapter)
        if version_bucket is None:
            raise KeyError(f"chapter {chapter} not found")
        membership = self.chapter_membership.get(chapter, {})
        edges = self.chapter_edges.get(chapter, [])
        nodes = []
        links = []
        for version in self.versions:
            records = version_bucket.get(version, [])
            nodes.append(
                {
                    "id": self._chapter_node_id(version, chapter),
                    "label": version,
                    "type": "chapter",
                    "version": version,
                    "chapter": chapter,
                    "community": membership.get(version, 0),
                    "sentenceCount": len(records),
                }
            )
            for record in records:
                nodes.append(
                    {
                        "id": self._sentence_node_id(version, chapter, record.sentence),
                        "label": f"{version}:{record.sentence}",
                        "type": "sentence",
                        "version": version,
                        "chapter": chapter,
                        "sentence": record.sentence,
                        "text": record.text,
                        "community": membership.get(version, 0),
                    }
                )
                links.append(
                    {
                        "source": self._chapter_node_id(version, chapter),
                        "target": self._sentence_node_id(
                            version, chapter, record.sentence
                        ),
                        "type": "contains_sentence",
                        "similarity": 0.18,
                    }
                )
        for edge in edges[:40]:
            links.append(
                {
                    **edge,
                    "source": self._chapter_node_id(edge["source"], chapter),
                    "target": self._chapter_node_id(edge["target"], chapter),
                    "sourceVersion": edge["source"],
                    "targetVersion": edge["target"],
                    "type": "chapter_similarity",
                }
            )
        sentence_edges = self.sentence_similarity_edges.get(chapter, [])
        links.extend(sentence_edges[:900])
        sentences = {
            version: [
                {
                    "sentence": record.sentence,
                    "text": record.text,
                }
                for record in version_bucket.get(version, [])
            ]
            for version in self.versions
        }
        return {
            "chapter": chapter,
            "nodes": nodes,
            "links": links,
            "sentences": sentences,
            "communities": membership,
            "topEdges": edges[:20],
            "ranking": [
                {
                    **edge,
                    "source": self._chapter_node_id(edge["source"], chapter),
                    "target": self._chapter_node_id(edge["target"], chapter),
                    "sourceVersion": edge["source"],
                    "targetVersion": edge["target"],
                    "type": "chapter_similarity",
                }
                for edge in edges
            ],
            "sentenceRanking": sentence_edges,
        }

    def _original_rows(self, chapter: int, version: str) -> list[dict[str, Any]]:
        rows = self.original_text_by_chapter.get(chapter, {}).get(version, {})
        return [
            {
                "sentence": sentence,
                "sentenceDisplay": sentence + 1,
                "text": text,
                "nodeId": self._sentence_node_id(version, chapter, sentence),
            }
            for sentence, text in sorted(rows.items())
        ]

    def _nearest_sentence_match(
        self,
        chapter: int,
        version: str,
        sentence: int,
    ) -> dict[str, Any] | None:
        node_id = self._sentence_node_id(version, chapter, sentence)
        candidates = []
        for edge in self.sentence_similarity_edges.get(chapter, []):
            if edge["source"] != node_id and edge["target"] != node_id:
                continue
            other_id = edge["target"] if edge["source"] == node_id else edge["source"]
            other_version = (
                edge["targetVersion"]
                if edge["source"] == node_id
                else edge["sourceVersion"]
            )
            if other_version == version:
                continue
            other_sentence = self._parse_sentence_node_id(other_id)["sentence"]
            if not self._original_rows(chapter, other_version):
                continue
            candidates.append(
                {
                    "version": other_version,
                    "sentence": other_sentence,
                    "nodeId": other_id,
                    "distance": edge["distance"],
                    "similarity": edge["similarity"],
                    "source": "sentence",
                }
            )
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item["distance"], item["version"]))
        return candidates[0]

    def _nearest_chapter_match(self, chapter: int, version: str) -> dict[str, Any] | None:
        candidates = []
        for edge in self.chapter_edges.get(chapter, []):
            if edge["source"] != version and edge["target"] != version:
                continue
            other_version = edge["target"] if edge["source"] == version else edge["source"]
            if not self._original_rows(chapter, other_version):
                continue
            candidates.append(
                {
                    "version": other_version,
                    "sentence": None,
                    "nodeId": self._chapter_node_id(other_version, chapter),
                    "distance": None,
                    "similarity": edge["similarity"],
                    "source": "chapter",
                }
            )
        if not candidates:
            return None
        candidates.sort(key=lambda item: item["similarity"], reverse=True)
        return candidates[0]

    def _parse_sentence_node_id(self, node_id: str) -> dict[str, Any]:
        match = re.fullmatch(r"sent_(.+)_(\d+)_(\d+)", node_id)
        if not match:
            raise KeyError(f"invalid sentence node id: {node_id}")
        return {
            "version": match.group(1),
            "chapter": int(match.group(2)),
            "sentence": int(match.group(3)),
        }

    def parallel_reading_api(
        self,
        chapter: int,
        version: str,
        sentence: int,
    ) -> dict[str, Any]:
        if not self._sentence_exists(version, chapter, sentence):
            raise KeyError(f"sentence not found: {version} chapter {chapter} sentence {sentence}")
        left_rows = self._original_rows(chapter, version)
        if not left_rows:
            raise KeyError(f"original text not found for {version} chapter {chapter}")
        nearest = self._nearest_sentence_match(chapter, version, sentence)
        if nearest is None:
            nearest = self._nearest_chapter_match(chapter, version)
        if nearest is None:
            raise KeyError(f"parallel version not found for {version} chapter {chapter}")
        right_rows = self._original_rows(chapter, nearest["version"])
        if not right_rows:
            raise KeyError(f"original text not found for {nearest['version']} chapter {chapter}")
        return {
            "chapter": chapter,
            "focus": {
                "version": version,
                "sentence": sentence,
                "sentenceDisplay": sentence + 1,
                "nodeId": self._sentence_node_id(version, chapter, sentence),
            },
            "nearest": {
                "version": nearest["version"],
                "sentence": nearest["sentence"],
                "sentenceDisplay": None
                if nearest["sentence"] is None
                else nearest["sentence"] + 1,
                "nodeId": nearest["nodeId"],
                "distance": nearest["distance"],
                "similarity": nearest["similarity"],
                "source": nearest["source"],
            },
            "columns": [
                {
                    "side": "left",
                    "version": version,
                    "rows": left_rows,
                },
                {
                    "side": "right",
                    "version": nearest["version"],
                    "rows": right_rows,
                },
            ],
        }

    def chapters_api(self) -> dict[str, Any]:
        return {
            "chapters": [
                {
                    "chapter": chapter,
                    "versions": len(self.sentences_by_chapter.get(chapter, {})),
                    "communityAvailable": chapter in self.chapter_membership,
                    "edgeCount": len(self.chapter_edges.get(chapter, [])),
                }
                for chapter in self.chapters
            ]
        }

    def book_affinity_api(self) -> dict[str, Any]:
        return {
            "versions": self.versions,
            "matrix": [
                [self.book_affinity.get(a, {}).get(b) for b in self.versions]
                for a in self.versions
            ],
            "sources": [
                [self.book_affinity_sources.get(a, {}).get(b, "unknown") for b in self.versions]
                for a in self.versions
            ],
        }


DATA = DataStore()


class VisualizationHandler(BaseHTTPRequestHandler):
    server_version = "MoraViz/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        try:
            if path.startswith("/api/"):
                self._handle_api(path, query)
            else:
                self._serve_static(path)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": repr(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_api(self, path: str, query: dict[str, list[str]]) -> None:
        if path == "/api/overview":
            self._send_json(DATA.overview())
            return
        if path == "/api/version-graph":
            threshold = float(query.get("threshold", ["0.6"])[0])
            top_k = int(query.get("top_k", query.get("topK", ["3"]))[0])
            self._send_json(DATA.version_graph(threshold=threshold, top_k=top_k))
            return
        if path == "/api/version-ranking":
            limit = int(query.get("limit", ["66"])[0])
            self._send_json(DATA.version_ranking_api(limit=limit))
            return
        if path == "/api/chapters":
            self._send_json(DATA.chapters_api())
            return
        if path == "/api/book-affinity":
            self._send_json(DATA.book_affinity_api())
            return
        if path == "/api/search-sentences":
            query_text = query.get("q", [""])[0]
            limit = int(query.get("limit", ["12"])[0])
            self._send_json(DATA.search_sentences(query_text, limit=limit))
            return
        if path == "/api/parallel-reading":
            chapter = int(query.get("chapter", ["0"])[0])
            version = query.get("version", [""])[0]
            sentence = int(query.get("sentence", ["0"])[0])
            self._send_json(
                DATA.parallel_reading_api(
                    chapter=chapter,
                    version=version,
                    sentence=sentence,
                )
            )
            return
        match = re.fullmatch(r"/api/chapter/(\d+)", path)
        if match:
            self._send_json(DATA.chapter_api(int(match.group(1))))
            return
        self._send_json({"error": f"unknown api path: {path}"}, status=HTTPStatus.NOT_FOUND)

    def _serve_static(self, path: str) -> None:
        if path in {"", "/"}:
            target = STATIC_DIR / "index.html"
        else:
            target = (STATIC_DIR / path.lstrip("/")).resolve()
            if not str(target).startswith(str(STATIC_DIR.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN)
                return
        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[viz] {self.address_string()} - {fmt % args}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Mora visualization web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8066)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), VisualizationHandler)
    print(f"Mora visualization system: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
