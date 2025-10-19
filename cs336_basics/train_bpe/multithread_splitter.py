import os
from typing import BinaryIO
from concurrent.futures import ThreadPoolExecutor

class MultiThreadSplitter:
    @staticmethod
    def split(file: BinaryIO, special_tokens: list[str]) -> list[tuple[int, int]]:
        """
        读取文件文本内容，根据机器核数确定线程数，
        并行根据特殊token划分文件边界，返回[(start, end), ...]。
        使用这些区间切片后，结果不包含任何 special token。
        """
        raw = file.read()
        data = raw if isinstance(raw, (bytes, bytearray)) else raw.encode("utf-8")

        token_bytes = [t.encode("utf-8") for t in special_tokens]
        file_len = len(data)
        if file_len == 0:
            return []

        n_threads = max(1, os.cpu_count() or 1)
        chunk_size = file_len // n_threads
        overlap = max(len(t) for t in token_bytes) if token_bytes else 1

        def find_boundaries(start, end):
            s = max(0, start - overlap)
            e = min(file_len, end + overlap)
            text_chunk = data[s:e]

            indices = []
            for token in token_bytes:
                pos = text_chunk.find(token)
                while pos != -1:
                    real_start = s + pos
                    real_end = real_start + len(token)
                    indices.append((real_start, real_end))
                    pos = text_chunk.find(token, pos + 1)
            return indices

        futures = []
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            for i in range(n_threads):
                start = i * chunk_size
                end = file_len if i == n_threads - 1 else (i + 1) * chunk_size
                futures.append(executor.submit(find_boundaries, start, end))

        # 收集所有 token 出现位置
        token_spans = []
        for fut in futures:
            token_spans.extend(fut.result())

        token_spans = sorted(set(token_spans))

        # 构造 [(start, end)] 段，不含 special token
        boundaries: list[tuple[int, int]] = []
        last_end = 0
        for (tok_start, tok_end) in token_spans:
            if tok_start > last_end:
                boundaries.append((last_end, tok_start))
            last_end = tok_end
        if last_end < file_len:
            boundaries.append((last_end, file_len))

        return boundaries