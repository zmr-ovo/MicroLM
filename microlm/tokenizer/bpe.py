import os
from collections import defaultdict, Counter
import regex as re  # type: ignore
import json

"""BPE（Byte Pair Encoding）分词器训练模块。

提供 BPE 词表的完整训练流程：从单字节初始化、预分词、频次统计、
迭代合并到最终词表生成，以及词表文件的保存。
"""

def train_bpe(input_path, vocab_size, special_tokens) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练 BPE 分词器，返回词表和合并规则。

    从 256 个单字节 token 出发，按 GPT-2 正则预分词后统计相邻 token 对频次，
    反复合并最高频 pair 直到达到目标词表大小。特殊符号不会被拆分。
    """

    ##1.初始化词表
    vocab = {b:bytes([b]) for b in range(256)}
    merge_nums = vocab_size - 256 - len(special_tokens)

    ##2.读取数据，并根据特殊字符"拆分
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    if special_tokens:
        special_regex = '|'.join(re.escape(t) for t in special_tokens)
        corpus = re.split(special_regex, text)
        corpus = [p for p in corpus if p]
    else:
        corpus = [text]

    ##3.预分词，统计
    gpt2_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    raw_counts = Counter()
    for corpu in corpus:
        words = gpt2_pat.findall(corpu)
        for word in words:
            raw_counts[tuple(bytes([b]) for b in word.encode('utf-8'))] += 1
    word_list = []
    word_freq = []
    for word, freq in raw_counts.items():
        word_list.append(list(word))
        word_freq.append(freq)
    pair_freq = defaultdict(int)
    pair_id = defaultdict(set)
    for id,word in enumerate(word_list):
        for i in range(len(word)-1):
            pair = (word[i], word[i+1])
            pair_freq[pair] += word_freq[id]
            pair_id[pair].add(id)
    merge = []

    ##4.迭代合并流程
    for _ in range(merge_nums):
        if not pair_freq:
            break
        best_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]
        if pair_freq[best_pair] == 0:
            break
        ids = list(pair_id[best_pair])
        for id in ids:
            word = word_list[id]
            freq = word_freq[id]
            i = 0
            while(i<len(word)-1):
                if(word[i] == best_pair[0] and word[i+1] == best_pair[1]):
                    if i>0:
                        prev_pair = (word[i-1], word[i])
                        pair_freq[prev_pair] -= freq
                        if pair_freq[prev_pair] == 0:
                            del pair_freq[prev_pair]
                    if i<len(word) -2:
                        next_pair = (word[i+1], word[i+2])
                        pair_freq[next_pair] -= freq
                        if pair_freq[next_pair] == 0:
                            del pair_freq[next_pair]
                    word[i] = word[i] + word[i+1]
                    del word[i+1]
                    if i>0:
                        new_prev_pair = (word[i-1], word[i])
                        pair_freq[new_prev_pair] += freq
                        pair_id[new_prev_pair].add(id)
                    if i<len(word)-1:
                        new_next_pair = (word[i], word[i+1])
                        pair_freq[new_next_pair] += freq
                        pair_id[new_next_pair].add(id)
                    ##配对成功时，i不需要移动，因为下一次会检查新的word[i]和word[i+1]
                else:
                    i+=1
        merge.append(best_pair)
        if best_pair in pair_freq: del pair_freq[best_pair]
        if best_pair in pair_id: del pair_id[best_pair]

    ##5.构建最终词表vocab
    for word in merge:
        vocab[len(vocab)] = word[0] + word[1]
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')
    return vocab,merge

def bytes_to_unicode():
    """将 0-255 字节映射为可见 Unicode 字符，用于保存词表文件时保证可读性。

    可打印字符（如字母、数字）映射到自身，控制字符等不可见字节
    映射到高区 Unicode 码点（chr(256+)）。这是 GPT-2 的标准做法。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b in bs:
            continue
        bs.append(b)
        cs.append(n+256)
        n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs,cs))

def save_tokenizer_files(vocab, merges, out_dir):
    """将训练好的词表和合并规则保存到磁盘。

    输出两个文件：vocab.json（词表，token ID → 可见字符映射后的字符串）
    和 merge.txt（合并规则，每行一条，格式为"左token 右token"）。
    """
    os.makedirs(out_dir, exist_ok=True)
    bytes_encoder = bytes_to_unicode()
    json_vocab ={
        k:"".join([bytes_encoder[b] for b in v])
        for k,v in vocab.items()
    }
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding = "utf-8") as f:
        json.dump( json_vocab, f, indent = 4)
    with open(os.path.join(out_dir, "merge.txt"), "w", encoding ="utf-8") as f:
        for p1,p2 in merges:
            r1 = "".join(bytes_encoder[b] for b in p1)
            r2 = "".join(bytes_encoder[b] for b in p2)
            f.write(f"{r1} {r2}\n")
