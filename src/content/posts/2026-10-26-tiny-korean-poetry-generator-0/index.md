---
title: 'Tiny Korean poetry generator #1: data and tokenizer'
pubDate: 2026-10-26
tags: [dev]
---

## Intro
I'm going to make a tiny Korean poetry generator from scratch.

Why from scratch? There are a lot of great LLMs already and they can write poems much better than mine for sure. (maybe? honestly i haven't checked :)) But I don't want to just use them, I want to know what happens inside. So I'm gonna build every part by myself. tokenizer, model, training and inference.

This is the plan for now
1. Why I build it myself, and collecting data
2. Korean tokenizer: byte BPE vs syllable
3. Model architecture: RoPE, RMSNorm, GQA, SwiGLU
4. Pretraining on Wikipedia: loss curve and how the outputs change
5. Fine-tuning on poems: the moment the model starts to write "like a poem"
6. Inference optimization: before and after KV cache

This post covers 1 and 2.

All the code is on [GitHub](https://github.com/junpark4395/tiny-korean-poetry-generator). (some helper functions like `BPEUtils` are only there)

## Data
To pretrain a model, I need a lot of Korean text. I chose Korean Wikipedia. It's free, big and clean enough. I downloaded it from [Hugging Face (wikimedia/wikipedia, 20231101.ko)](https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.ko) as parquet files (3 files in total).
Poems will come later in part 5.

## Byte BPE vs syllable
A model can't read text. It only reads numbers. So I need a tokenizer to turn text into ids.
For Korean, I had two options.

- Syllable: every Korean character (like 가, 나, 한) is one token. Simple. But there are 11,172 possible Hangul syllables, and if a character is not in the vocab (rare characters, hanja, emoji...), the model can't handle it.
- Byte BPE: start from 256 bytes, and keep merging the most frequent pair. Any text can be encoded cuz everything is bytes in the end. And frequent words like `경제` or `이었다` become one token, so sequences get shorter.

I chose byte BPE. The downside is that a Korean character is 3 bytes in UTF-8, so the tokenizer has to learn how to make characters first. (you'll see broken characters later)

My target vocab size is 16,000.

## How BPE works
Before the code, let me show a tiny example. Let's say we only have 3 words: `한국`, `한글`, `한국어`.
In UTF-8, every Korean character is 3 bytes. So they look like this

```Bash
한국   → [237, 149, 156, 234, 181, 173]
한글   → [237, 149, 156, 234, 184, 128]
한국어 → [237, 149, 156, 234, 181, 173, 236, 150, 180]
```

Now find the most frequent pair, merge it into a new id, and repeat.

```Bash
step 0: (237, 149) appears 3 times → 256 = �     (piece of 한)
step 1: (256, 156) appears 3 times → 257 = 한
step 2: (257, 234) appears 3 times → 258 = 한�   (한 + piece of 국/글)
step 3: (258, 181) appears 2 times → 259 = 한�
step 4: (259, 173) appears 2 times → 260 = 한국

한국   → [260]
한글   → [258, 184, 128]
한국어 → [260, 236, 150, 180]
```

That's it. The merged pairs are the "rules", and the tokenizer just applies them in the same order.
You can see the broken characters (�) in the middle. That's the downside I talked about.

## Training BPE
This is my first version.

```Python
import pandas as pd
import pickle
from bpe_utils import BPEUtils

class BPETrainer:
    def __init__(self, filepath):
        self.vocab_size = 16000
        self.df = pd.read_parquet(filepath, columns=["text"])

    def train_bpe(self):
        ids_list = []
        for text in self.df["text"]:
            for token in BPEUtils.pretokenize(text):
                ids_list.append(list(token.encode('utf-8')))

        pair_size = self.vocab_size - 256 - 1
        rules = {}

        for step in range(pair_size):
            if step % 100 == 0:
                print(f"step: {step}, total: {pair_size}")

            count_map = {}
            for ids in ids_list:
                BPEUtils.count_pairs(ids, count_map)

            if not count_map:
                break

            max_pair = None
            max_count = -1
            for key, value in count_map.items():
                if max_count < value:
                    max_pair = key
                    max_count = value

            new_id = 256 + step
            rules[max_pair] = new_id

            for i in range(len(ids_list)):
                ids_list[i] = BPEUtils.merge(ids_list[i], max_pair, new_id)

        return rules


if __name__ == "__main__":
    bpe_trainer = BPETrainer("./dataset/train-00000-of-00003.parquet")
    rules = bpe_trainer.train_bpe()

    with open("./output/bpe_rules_16k.pkl", "wb") as f:
        pickle.dump(rules, f)

```

I tried this code but it's tooooo slow.

```Python
import pandas as pd

df = pd.read_parquet("./dataset/train-00000-of-00003.parquet")
print(len(df))

# ---output---
# 215966
```

As you can see, it has 200k texts in the first file (I have 3 files)
And it spends a lot of time processing the same words again and again.


```Python
ids_count = defaultdict(int)
for text in df["text"][:100]:
    for token in BPEUtils.pretokenize(text):
        ids_count[tuple(token.encode('utf-8'))] += 1

top_10 = sorted(ids_count.items(), key=lambda x: x[1], reverse=True)[:10]
for key, value in top_10:
    print(f"meaning: {bytes(key).decode('utf-8')}, key: {key}, value: {value}")

# meaning:
# , key: (10,), value: 8539
# meaning: ., key: (46,), value: 7026
# meaning: ,, key: (44,), value: 4473
# meaning: ), key: (41,), value: 1868
# meaning: (, key: (40,), value: 1640
# meaning: 년, key: (235, 133, 132), value: 1569
# meaning: 월, key: (236, 155, 148), value: 935
# meaning:  (, key: (32, 40), value: 856
# meaning:  -, key: (32, 45), value: 817
# meaning: 일, key: (236, 157, 188), value: 733
```


So this is my next version. I changed ids_list to ids_count to avoid processing duplicates. (and I only used the first 10,000 texts for now)

```Python
# Before
ids_list = []
for text in self.df["text"]:
    for token in BPEUtils.pretokenize(text):
        ids_list.append(list(token.encode('utf-8')))

count_map = {}
for ids in ids_list:
    BPEUtils.count_pairs(ids, count_map)

for i in range(len(ids_list)):
    ids_list[i] = BPEUtils.merge(ids_list[i], max_pair, new_id)

# After
from collections import defaultdict

ids_count = defaultdict(int)
for text in self.df["text"][:10000]:
    for token in BPEUtils.pretokenize(text):
        ids_count[tuple(token.encode('utf-8'))] += 1

count_map = {}
for ids, weight in ids_count.items():
    BPEUtils.count_pairs(ids, count_map, weight)

new_ids_count = defaultdict(int)
for ids, count in ids_count.items():
    new_ids_count[tuple(BPEUtils.merge(ids, max_pair, new_id))] += count
ids_count = new_ids_count
```

But it's still slow. I can't even see step 100 in the output.
The problem is the merging logic. Every step, it counts all pairs again and tries to merge every word, even though most words don't have max_pair. I only need to update the ids that have max_pair.

So I made an index `pair_words` that remembers which words have each pair. Now I count pairs only once at the beginning, and after each merge I only fix the counts of the words that had max_pair. (subtract the old pairs, add the new pairs)

```Python
# before the loop: count pairs once, and remember which words have each pair
words = [list(ids) for ids in ids_count.keys()]
counts = list(ids_count.values())

count_map = {}
pair_words = defaultdict(set)
for i, ids in enumerate(words):
    BPEUtils.count_pairs(ids, count_map, counts[i])
    for pair in zip(ids, ids[1:]):
        pair_words[pair].add(i)

# inside the loop: only update the words that have max_pair
for i in pair_words.pop(max_pair):
    old_ids = words[i]
    new_ids = BPEUtils.merge(old_ids, max_pair, new_id)

    BPEUtils.count_pairs(old_ids, count_map, -counts[i])
    BPEUtils.count_pairs(new_ids, count_map, counts[i])
    for pair in zip(new_ids, new_ids[1:]):
        pair_words[pair].add(i)

    for pair in zip(old_ids, old_ids[1:]):
        if count_map.get(pair) == 0:
            del count_map[pair]

    words[i] = new_ids
```

Finally I can see the second step in the output. I guess it will take a few hours.
I still need to optimize it

```Python
# Before
for text in self.df["text"][:10000]:
    for token in BPEUtils.pretokenize(text):
        ids_count[tuple(token.encode('utf-8'))] += 1

words = [list(ids) for ids in ids_count.keys()]
counts = list(ids_count.values())

# After
for text in self.df["text"][:10000]:
    for token in BPEUtils.pretokenize(text):
        token_count[token] += 1

words = [list(token.encode('utf-8')) for token in token_count.keys()]
counts = list(token_count.values())
```

`ids_count[tuple(token.encode('utf-8'))] += 1 ` This code was one of the problems. I only need to encode the same word once.
Now it's acceptable. Let's see the result

```Bash
step: 14600, total: 15743
step: 14700, total: 15743
step: 14800, total: 15743
step: 14900, total: 15743
step: 15000, total: 15743
step: 15100, total: 15743
step: 15200, total: 15743
step: 15300, total: 15743
step: 15400, total: 15743
step: 15500, total: 15743
step: 15600, total: 15743
step: 15700, total: 15743
train: 454.98s
```

It took 7 mins. hmm... I know there must be something I can improve. But I don't want to spend more time on making rules.
Let me check inside the output file

```Python
import pickle

with open("output/bpe_rules_16k.pkl", "rb") as f:
    rules = pickle.load(f)

vocab = {i: bytes([i]) for i in range(256)}
for (id1, id2), new_id in rules.items():
    vocab[new_id] = vocab[id1] + vocab[id2]

for key, value in list(rules.items())[1000:1050]:
    print(f"meaning: {vocab[value].decode('utf-8', errors='replace')}, key: {key}, value: {value}")

# meaning:  러, key: (1142, 172), value: 1256
# meaning: 10, key: (49, 48), value: 1257
# meaning:  시대, key: (432, 347), value: 1258
# meaning:  네, key: (785, 164), value: 1259
# meaning: �, key: (236, 189), value: 1260
# meaning: 헌, key: (699, 140), value: 1261
# meaning:  주장, key: (464, 403), value: 1262
# meaning: 존, key: (524, 180), value: 1263
# meaning: 번째, key: (839, 913), value: 1264
# meaning: 봉, key: (235, 1040), value: 1265
# meaning:  이루, key: (304, 658), value: 1266
# meaning:  존재, key: (1032, 562), value: 1267
# meaning: 성을, key: (394, 282), value: 1268
# meaning:  드, key: (440, 156), value: 1269
# meaning:

# , key: (419, 10), value: 1270
# meaning: 우스, key: (421, 323), value: 1271
# meaning:  모든, key: (561, 966), value: 1272
# meaning: 리를, key: (310, 314), value: 1273
# meaning: ol, key: (111, 108), value: 1274
# meaning:  �, key: (258, 168), value: 1275
# meaning:  경제, key: (496, 405), value: 1276
# meaning:  수도, key: (395, 319), value: 1277
# meaning: 런, key: (396, 176), value: 1278
# meaning: 리카, key: (310, 611), value: 1279
# meaning: 범, key: (392, 148), value: 1280
# meaning: 특, key: (409, 185), value: 1281
# meaning:  t, key: (32, 116), value: 1282
# meaning:  인구, key: (468, 391), value: 1283
# meaning:  논, key: (590, 188), value: 1284
# meaning: 이었다, key: (266, 420), value: 1285
# meaning:  G, key: (32, 71), value: 1286
# meaning:  규, key: (356, 156), value: 1287
# meaning:  중심, key: (447, 815), value: 1288
# meaning:  위한, key: (509, 280), value: 1289
# meaning: 박, key: (344, 149), value: 1290
# meaning: 나라, key: (370, 336), value: 1291
# meaning: 17, key: (49, 55), value: 1292
# meaning:  기록, key: (412, 735), value: 1293
# meaning:  순, key: (876, 156), value: 1294
# meaning:  돌, key: (374, 140), value: 1295
# meaning:  취, key: (256, 1107), value: 1296
# meaning:  날, key: (369, 160), value: 1297
# meaning:  이상, key: (304, 430), value: 1298
# meaning:  아니, key: (427, 503), value: 1299
# meaning: 규, key: (473, 156), value: 1300
# meaning:  윤, key: (368, 164), value: 1301
# meaning: 하였고, key: (437, 299), value: 1302
# meaning: 추, key: (558, 148), value: 1303
# meaning: 찰, key: (474, 176), value: 1304
# meaning:  등의, key: (508, 263), value: 1305

```
I printed from 1000 to 1050 cuz a Korean character is 3 bytes, so most of the early rules are just pieces of characters and look broken.
According to this result, it makes sense to me. (you know I'm Korean. trust me!)


Now it's time to make a tokenizer to encode and decode

```Python
import pickle
from bpe_utils import BPEUtils

class Tokenizer:
    def __init__(self, rules):
        self.rules = rules

        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in self.rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]

    def encode(self, data):
        result = []
        for token in BPEUtils.pretokenize(data):
            ids = list(token.encode('utf-8'))

            while len(ids) >= 2:
                pairs = set(zip(ids, ids[1:]))
                pair = min(pairs, key=lambda p: self.rules.get(p, float('inf')))
                if pair not in self.rules:
                    break

                ids = BPEUtils.merge(ids, pair, self.rules[pair])

            result.extend(ids)

        return result


    def decode(self, ids):
        decoded = b"".join(self.id_to_bytes[i] for i in ids)
        return decoded.decode('utf-8', errors='replace')


if __name__ == "__main__":
    rules = None
    with open("output/bpe_rules_16k.pkl", "rb") as f:
        rules = pickle.load(f)

    tokenizer = Tokenizer(rules)
    encoded = tokenizer.encode("미국대한민국")
    decoded = tokenizer.decode(encoded)

    print(encoded)
    print(decoded)

# ---output---
# [3218, 3193]
# 미국대한민국
```

`미국대한민국` means USA and South Korea. Finally I've got the rules and the tokenizer.
Now I can encode and decode the data.

Let's try a real sentence and see where it splits. (it means "The capital of South Korea is Seoul")

```Python
encoded = tokenizer.encode('대한민국의 수도는 서울이다')
print(encoded)
print([tokenizer.decode([i]) for i in encoded])

# ---output---
# [2101, 10830, 986, 445]
# ['대한민국의', ' 수도는', ' 서울', '이다']
```

Interesting. Particles like `의` and `는` were not split. `대한민국의` and ` 수도는` became one token. I guess they show up together so often in Wikipedia. But `서울이다` was split into ` 서울` and `이다`.
4 tokens for 14 characters. Not bad.

Before finishing today's job, let me check the compression rate.

```Python
if __name__ == "__main__":
    rules = None
    with open("output/bpe_rules_16k.pkl", "rb") as f:
        rules = pickle.load(f)

    tokenizer = Tokenizer(rules)

    df = pd.read_parquet("./dataset/train-00001-of-00003.parquet", columns=["text"])
    texts = df["text"].sample(n=300, random_state=0)

    n_chars = n_bytes = n_tokens = 0
    for text in texts:
        n_chars += len(text)
        n_bytes += len(text.encode('utf-8'))
        n_tokens += len(tokenizer.encode(text))

    print(f"bytes/token: {n_bytes / n_tokens:.2f}")
    print(f"chars/token: {n_chars / n_tokens:.2f}")

# ---output---
# bytes/token: 4.18
# chars/token: 1.95
```

Simply put, one token holds about 4 bytes. So it's 4 times shorter than plain UTF-8 bytes.
And compared to syllables (1 char = 1 token), it's about 2 times shorter. (chars/token: 1.95)
So the same text needs only half the tokens. That's why I chose byte BPE!
Umm.. Well.. Good right?


