import os
from typing import BinaryIO
from concurrent.futures import ThreadPoolExecutor
import regex as re
from typing import BinaryIO
from concurrent.futures import ThreadPoolExecutor

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class SingleThreadBPETrainer:
    def __init__(self, chunks : list[bytes]):
        self.chunks = chunks

        # 存储每一个通过PAT分割出来的单词，单词到数量的映射， str -> int
        self.word_cnt = dict[str, int]()
        # 存储每一个通过PAT分割出来的单词，内部对应的token pair, str -> list[tuple[bytes, bytes]]
        self.word_pattern_dict = dict[str, list[tuple[bytes, bytes]]]()
        # 存储每一个通过PAT分割出来的单词，按照pair分割的elements
        self.word_elements_dict = dict[str, list[bytes]]()
        # 存储每一个token pair对应的数量, tuple[bytes, bytes] -> int
        self.token_pair_statistics = dict[tuple[bytes, bytes], int]()

        self.init_statistics()

    """
    根据自己被分配的chunks，进行token pair统计的初始化
    """
    def init_statistics(self) :
        """
        对初始chunks进行token pair频率统计
        返回频率统计结果
        """
        for chunk in self.chunks:
            iterators = re.finditer(PAT, chunk.decode("utf-8"))
            for match in iterators:
                word = match.group()
                cnt = self.word_cnt.get(word, 0)

                if cnt == 0:
                    pattern = list[tuple[bytes, bytes]]()
                    word_bytes = word.encode("utf-8")
                    # 将word_bytes拆分成token pair
                    for i in range(len(word_bytes) - 1):
                        token_pair = (bytes([word_bytes[i]]), bytes([word_bytes[i + 1]]))
                        pattern.append(token_pair)
                    self.word_pattern_dict[word] = pattern

                    elements = list[bytes]()
                    for i in range(len(word_bytes)):
                        elements.append(bytes([word_bytes[i]]))
                    self.word_elements_dict[word] = elements

                self.word_cnt[word] = cnt + 1
        
        for word, cnt in self.word_cnt.items():
            pattern = self.word_pattern_dict[word]
            for token_pair in pattern:
                self.token_pair_statistics[token_pair] = self.token_pair_statistics.get(token_pair, 0) + cnt

    """
    根据要merge的两个token pair，更新新的token pair的频率统计信息
    """
    def update_vocab(self, merged_token_pair: tuple[bytes, bytes]) -> None:
        # 如果这个pair没出现过，直接跳过
        if merged_token_pair not in self.token_pair_statistics:
            return

        # 从全局统计里减去该 pair 的总数
        old_count = self.token_pair_statistics.pop(merged_token_pair, 0)

        # 缓存结果，以便稍后增量修补
        new_pairs_to_add = dict[tuple[bytes, bytes], int]()
        new_pairs_to_remove = dict[tuple[bytes, bytes], int]()

        # 遍历包含该pair的词进行局部更新
        for word, cnt in self.word_cnt.items():
            elements = self.word_elements_dict[word]
            i = 0
            changed = False

            # 提前过滤：不含该pair的词跳过
            while i < len(elements) - 1:
                if (elements[i], elements[i + 1]) == merged_token_pair:
                    changed = True
                    break
                i += 1
            if not changed:
                continue

            # 实际合并
            new_elements = []
            i = 0
            while i < len(elements):
                if i < len(elements) - 1 and (elements[i], elements[i + 1]) == merged_token_pair:
                    merged = elements[i] + elements[i + 1]
                    new_elements.append(merged)

                    # 移除旧的左右邻居 pair 统计
                    if i > 0:
                        left_pair = (elements[i - 1], elements[i])
                        new_pairs_to_remove[left_pair] = new_pairs_to_remove.get(left_pair, 0) + cnt
                        new_pairs_to_add[(elements[i - 1], merged)] = new_pairs_to_add.get((elements[i - 1], merged), 0) + cnt
                    if i + 2 < len(elements):
                        right_pair = (elements[i + 1], elements[i + 2])
                        new_pairs_to_remove[right_pair] = new_pairs_to_remove.get(right_pair, 0) + cnt
                        new_pairs_to_add[(merged, elements[i + 2])] = new_pairs_to_add.get((merged, elements[i + 2]), 0) + cnt

                    i += 2
                else:
                    new_elements.append(elements[i])
                    i += 1

            # 更新内部状态
            self.word_elements_dict[word] = new_elements
            # 重建 pattern（可选，仅在调试或 merge 输出时使用）
            new_patterns = [(new_elements[j], new_elements[j + 1]) for j in range(len(new_elements) - 1)]
            self.word_pattern_dict[word] = new_patterns

        # 增量修改全局统计
        for pair, c in new_pairs_to_remove.items():
            if pair in self.token_pair_statistics:
                self.token_pair_statistics[pair] -= c
                if self.token_pair_statistics[pair] <= 0:
                    self.token_pair_statistics.pop(pair)
        for pair, c in new_pairs_to_add.items():
            self.token_pair_statistics[pair] = self.token_pair_statistics.get(pair, 0) + c


    """
    获取最新的token pair的频率统计信息
    """
    def get_statistics(self) -> dict[tuple[bytes, bytes], int]:
        return self.token_pair_statistics

class MultiThreadBPETrainer:
    @staticmethod
    def train(file: BinaryIO, 
              vocab_size: int,
              special_tokens: list[str],
              boundaries: list[tuple[int, int]]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        
        # 读取全部内容
        file.seek(0)
        raw = file.read()
        data = raw if isinstance(raw, (bytes, bytearray)) else raw.encode("utf-8")

        # 根据 boundaries 拆分成 chunks(bytes类型)
        chunks = []
        for (start, end) in boundaries:
            chunks.append(data[start:end])

        # 计算线程数
        n_threads = max(1, os.cpu_count() or 1)
        # 均分 chunks 给不同线程
        per_thread = max(1, len(chunks) // n_threads)
        grouped_chunks = [
            chunks[i:i + per_thread] for i in range(0, len(chunks), per_thread)
        ]

        # 并行初始化 SingleThreadBPETrainer
        trainers = []
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(SingleThreadBPETrainer, cks) for cks in grouped_chunks]
            for fut in futures:
                trainers.append(fut.result())

        # 汇总 token pair 统计结果
        global_statistics = dict()
        for trainer in trainers:
            for pair, count in trainer.get_statistics().items():
                global_statistics[pair] = global_statistics.get(pair, 0) + count

        # 初始化 vocab
        vocab = {i: bytes([i]) for i in range(256)}  # 初始为所有字节
        merges: list[tuple[bytes, bytes]] = []

        # 添加特殊 token（优先放在前面）
        for token in special_tokens:
            b = token.encode("utf-8")
            if b not in vocab.values():
                vocab[len(vocab)] = b

        # 不断 merge，直到 vocab_size 达到
        while len(vocab) < vocab_size and global_statistics:
            # 找到出现次数最多的 token pair，如果有相同的次数，取字典序最大的那个
            best_pair = max(global_statistics.items(), key=lambda kv: (kv[1], kv[0]))[0]
            merges.append(best_pair)

            # 并行更新每个trainer的统计
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = [executor.submit(trainer.update_vocab, best_pair) for trainer in trainers]
                # 等待所有任务完成
                for fut in futures:
                    fut.result()

            # 重新汇总全局统计
            global_statistics.clear()
            for trainer in trainers:
                for pair, count in trainer.get_statistics().items():
                    global_statistics[pair] = global_statistics.get(pair, 0) + count

            # 给新 token 编号
            merged_token = best_pair[0] + best_pair[1]
            vocab[len(vocab)] = merged_token

        return vocab, merges