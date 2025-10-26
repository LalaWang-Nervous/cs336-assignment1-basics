from collections.abc import Iterable, Iterator
from cs336_basics.train_bpe.bpe import PAT
from cs336_basics.train_bpe.multithread_splitter import MultiThreadSplitter
import regex as re
import json
import heapq

class TokenizerImpl:
    """
    Construct a tokenizer from a given vocabulary, list of merges, 
    and (optionally) a list of special tokens
    """
    def __init__(self, 
                 vocab : dict[int, bytes], 
                 merges : list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.vocab_reverse = dict[bytes, int]()
        for key, value in self.vocab.items():
            self.vocab_reverse[value] = key

        self.merges_priority = dict[bytes, int]()
        # 数字越大优先级越高
        for idx, t in enumerate(self.merges):
            self.merges_priority[t[0] + t[1]] = len(self.merges) - idx
    
    """
    Classmethod that constructs and return a Tokenizer from 
    a serialized vocabulary and list of merges(in the same format that your BPE training code output) 
    and (optionally) a list of specialtokens.
    """
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 读取文件
        vocab = cls._read_vocab_file(vocab_filepath)
        merges = cls._read_merges_file(merges_filepath)
        return cls(vocab, merges, special_tokens)
    
    @staticmethod
    def _read_vocab_file(filepath) -> dict[int, bytes]:
        # 读取词汇表文件的逻辑，是一个json文件
        with open('vocab.json', 'r', encoding='utf-8') as f:
            result = {v: k.encode('utf-8') for k, v in json.load(f).items()}
        
        return result
    
    @staticmethod
    def _read_merges_file(filepath):
        # 读取合并规则文件的逻辑
        with open(filepath, 'r') as f:
            return f.read().splitlines()
        

    def try_merge(self, 
                  words_bytes_list : list[bytes]) -> tuple[bool, list[bytes]]:
        # 先编码高频的组合
        # 构建一个矩阵M[len(words_bytes_list)][len(words_bytes_list)], 
        # M[i][j]表示words_bytes_list中第i到第j个元素组成的子串的编码优先级
        # 如果不存在则为-1，单个元素设置为0
        matrix = [[-1 for j in range(len(words_bytes_list))] for i in range(len(words_bytes_list))]
        for i in range(len(words_bytes_list)):
            matrix[i][i] = 0

        at_least_one_merged = False
        for i in range(len(words_bytes_list)):
            for j in range(i + 1, len(words_bytes_list)):
                if b''.join(words_bytes_list[i : j + 1]) in self.merges_priority:
                    matrix[i][j] = self.merges_priority[b''.join(words_bytes_list[i : j + 1])]
                    at_least_one_merged = True
                else:
                    matrix[i][j] = -1

        if not at_least_one_merged:
            return False, words_bytes_list

        # 遍历matrix，构建一个堆，堆中的每个节点存储优先级和对应的区间
        # 每次我们取出优先级最大的合并
        heap = []
        for i in range(len(words_bytes_list)):
            for j in range(i, len(words_bytes_list)):
                if matrix[i][j] != -1:
                    heapq.heappush(heap, (-matrix[i][j], (i, j + 1)))
        
        ret = list[bytes]()

        top_inerval = heapq.heappop(heap)[1]
        top_inerval_merged = False
        for idx in range(len(words_bytes_list)):
            if idx < top_inerval[0] or idx >= top_inerval[1]:
                ret.append(words_bytes_list[idx])
            elif not top_inerval_merged:
                ret.append(b''.join(words_bytes_list[top_inerval[0]:top_inerval[1]]))
                top_inerval_merged = True

        return self.try_merge(ret)
        

    def _encode_non_special_token_text(self, text : str) -> list[int]:
        ret = list[int]()

        iterators = re.finditer(PAT, text)
        for match in iterators:
            word = match.group()
            if len(word) == 0:
                continue

            word_bytes = word.encode('utf-8')
            init_list = [bytes([b]) for b in word_bytes]
            _, new_list = self.try_merge(init_list)
            ret.extend([self.vocab_reverse[x] for x in new_list])

        return ret

    """
    Encode an input text into a sequence of token IDs.
    """
    def encode(self, text : str) -> list[int]:
        text_bytes = text.encode('utf-8')
        boundaries, token_spans = MultiThreadSplitter.split_bytes(text_bytes, ([] if self.special_tokens == None else self.special_tokens))
        
        return self._encode_by_boundaries_and_special_token_spans(text_bytes, boundaries, token_spans)

    def _encode_by_boundaries_and_special_token_spans(self, 
                                                      text_bytes: bytes, 
                                                      boundaries: list, 
                                                      token_spans: list) -> list[int]:
        ret = list[int]()
        boundaries_idx = 0
        token_spans_idx = 0

        while boundaries_idx < len(boundaries) and token_spans_idx < len(token_spans):
            if boundaries[boundaries_idx][0] < token_spans[token_spans_idx][0]:
                boundary_text = text_bytes[boundaries[boundaries_idx][0]:boundaries[boundaries_idx][1]].decode('utf-8')
                ret.extend(self._encode_non_special_token_text(boundary_text))
                boundaries_idx += 1
            else:
                token_bytes = text_bytes[token_spans[token_spans_idx][0]:token_spans[token_spans_idx][1]]
                ret.append(self.vocab_reverse[token_bytes])
                token_spans_idx += 1
        
        while boundaries_idx < len(boundaries):
            boundary_text = text_bytes[boundaries[boundaries_idx][0]:boundaries[boundaries_idx][1]].decode('utf-8')
            ret.extend(self._encode_non_special_token_text(boundary_text))
            boundaries_idx += 1

        while token_spans_idx < len(token_spans):
            token_bytes = text_bytes[token_spans[token_spans_idx][0]:token_spans[token_spans_idx][1]]
            ret.append(self.vocab_reverse[token_bytes])
            token_spans_idx += 1

        return ret

    """
    Given an iterable of strings (e.g., a Python file handle), 
    return a generator that lazily yields token IDs. 
    This is required for memory-eﬀicient tokenization of large files 
    that we cannot directly load intomemory.
    """
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            # 将新文本块添加到缓冲区
            text_bytes = text_chunk.encode('utf-8')
            boundaries, token_spans = MultiThreadSplitter.split_bytes(
                text_bytes, 
                ([] if self.special_tokens is None else self.special_tokens))
            
            token_ids = self._encode_by_boundaries_and_special_token_spans(text_bytes, boundaries, token_spans)
            for token_id in token_ids:
                yield token_id

    """
    Decode a sequence of token IDs into text.
    """
    def decode(self, ids : list[int]) -> str:
        ret_bytes = bytes() 
        for id in ids:
            ret_bytes += self.vocab[id]
        
        return ret_bytes.decode('utf-8', errors='replace')
