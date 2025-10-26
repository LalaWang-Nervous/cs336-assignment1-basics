import os
from typing import BinaryIO
from concurrent.futures import ThreadPoolExecutor

class MultiThreadSplitter:
    @staticmethod
    def split_bytes(data: bytes, special_tokens: list[str]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        token_bytes = [t.encode("utf-8") for t in special_tokens]
        # 按照长度从大到小排序token_bytes
        if token_bytes:
            token_bytes = sorted(token_bytes, key=lambda x: len(x), reverse=True)

        file_len = len(data)
        if file_len == 0:
            return ([], [])

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

        def filter_overlapping_spans_efficient(token_spans):
            """
            更高效的方法过滤重叠区间
            """
            if not token_spans:
                return []
            
            # 按起始位置升序，长度降序排序
            sorted_spans = sorted(token_spans, key=lambda x: (x[0], -(x[1] - x[0])))
            
            filtered_spans = []
            current_max_end = -1
            
            for span in sorted_spans:
                start, end = span
                
                # 如果当前区间完全包含在前一个最大区间内，则跳过
                if end <= current_max_end:
                    continue
                
                # 否则添加当前区间，并更新最大结束位置
                filtered_spans.append(span)
                current_max_end = max(current_max_end, end)
            
            return filtered_spans
        
        token_spans = filter_overlapping_spans_efficient(token_spans)

        # 构造 [(start, end)] 段，不含 special token
        boundaries: list[tuple[int, int]] = []
        last_end = 0
        for (tok_start, tok_end) in token_spans:
            if tok_start > last_end:
                boundaries.append((last_end, tok_start))
            last_end = tok_end
        if last_end < file_len:
            boundaries.append((last_end, file_len))

        return boundaries, token_spans
        
    @staticmethod
    def split_file(file: BinaryIO, special_tokens: list[str]) -> list[tuple[int, int]]:
        raw = file.read()
        data = raw if isinstance(raw, (bytes, bytearray)) else raw.encode("utf-8")
        return MultiThreadSplitter.split_bytes(data, special_tokens)