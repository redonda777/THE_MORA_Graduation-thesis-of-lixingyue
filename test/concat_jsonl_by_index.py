import argparse
import heapq
import json
import tempfile
from pathlib import Path
from typing import List, Tuple, TextIO


DELIM = "\x1e"  # Unit Separator: unlikely to appear inside JSON text


def parse_index(obj: dict, index_field: str) -> int:
    if index_field not in obj:
        raise KeyError(f"Missing field: {index_field}")
    # Be strict: require an integer-like value.
    idx = obj[index_field]
    if isinstance(idx, bool):
        # bool is subclass of int; treat it as invalid
        raise ValueError(f"Field '{index_field}' must be an int, got bool: {idx!r}")
    return int(idx)


def flush_chunk(
    chunk: List[Tuple[int, int, str]],
    tmp_dir: Path,
    chunk_no: int,
) -> Path:
    # Stable sort for equal index: (index, serial)
    chunk.sort(key=lambda x: (x[0], x[1]))
    out_path = tmp_dir / f"chunk_{chunk_no:05d}.jsonl.tmp"
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for idx, serial, raw_line in chunk:
            # Store metadata for merge without mutating the final JSON.
            f.write(f"{idx}{DELIM}{serial}{DELIM}{raw_line}\n")
    return out_path


def merge_chunks(
    chunk_paths: List[Path],
    output_file: Path,
) -> None:
    if not chunk_paths:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("", encoding="utf-8", newline="\n")
        return

    # Heap items: (index, serial, chunk_i, raw_line)
    heap: List[Tuple[int, int, int, str]] = []
    chunk_files: List[TextIO] = []

    # Stream the first line of each chunk into the heap.
    chunk_files = [p.open("r", encoding="utf-8", newline="\n") for p in chunk_paths]
    try:
        for i, f in enumerate(chunk_files):
            line = f.readline()
            if not line:
                continue
            line = line.rstrip("\n")
            a, b, raw = line.split(DELIM, 2)
            heap.append((int(a), int(b), i, raw))
        heapq.heapify(heap)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8", newline="\n") as out:
            while heap:
                idx, serial, i, raw = heapq.heappop(heap)
                out.write(raw + "\n")
                next_line = chunk_files[i].readline()
                if next_line:
                    next_line = next_line.rstrip("\n")
                    a, b, raw2 = next_line.split(DELIM, 2)
                    heapq.heappush(heap, (int(a), int(b), i, raw2))
    finally:
        for f in chunk_files:
            try:
                f.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate all .jsonl files in a directory and sort by 'index' field."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing .jsonl files")
    parser.add_argument("--output-file", required=True, help="Output combined .jsonl file")
    parser.add_argument("--pattern", default="*.jsonl", help="Glob pattern for jsonl files")
    parser.add_argument("--index-field", default="index", help="Field name used for sorting")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200000,
        help="Number of records per in-memory chunk before writing temp files",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding used for reading/writing (default: utf-8)",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="If set, skip lines that can't be parsed as JSON or lack index field",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    pattern = args.pattern

    jsonl_files = sorted(input_dir.glob(pattern))
    if not jsonl_files:
        raise FileNotFoundError(f"No files matched: {input_dir / pattern}")

    tmp_dir_obj = tempfile.TemporaryDirectory(prefix="jsonl_sort_chunks_")
    tmp_dir = Path(tmp_dir_obj.name)
    chunk_paths: List[Path] = []

    buffer: List[Tuple[int, int, str]] = []
    serial = 0
    chunk_no = 0

    try:
        for file_path in jsonl_files:
            with file_path.open("r", encoding=args.encoding, newline="\n") as f:
                for line_no, line in enumerate(f, start=1):
                    raw_line = line.rstrip("\n")
                    if not raw_line.strip():
                        continue
                    try:
                        obj = json.loads(raw_line)
                        idx = parse_index(obj, args.index_field)
                    except Exception:
                        if args.skip_invalid:
                            continue
                        raise RuntimeError(
                            f"Failed parsing line. file={file_path} line={line_no}"
                        )

                    buffer.append((idx, serial, raw_line))
                    serial += 1

                    if len(buffer) >= args.chunk_size:
                        chunk_paths.append(flush_chunk(buffer, tmp_dir, chunk_no))
                        buffer = []
                        chunk_no += 1

        if buffer:
            chunk_paths.append(flush_chunk(buffer, tmp_dir, chunk_no))

        # Merge temp chunks.
        merge_chunks(chunk_paths, output_file)
    finally:
        # TemporaryDirectory cleanup is best-effort.
        tmp_dir_obj.cleanup()


if __name__ == "__main__":
    main()

