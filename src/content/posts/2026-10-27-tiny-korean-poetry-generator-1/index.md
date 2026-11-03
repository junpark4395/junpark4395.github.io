---
title: 'Tiny Korean poetry generator #2: preparing the training data'
pubDate: 2026-11-03
tags: [dev]
---

I have the tokenizer I made in the previous post. It's time to tokenize the training data

```python
import pandas as pd
import pickle
import numpy as np
from bpe_utils import BPEUtils

class Tokenizer:
    def __init__(self, rules):
        self.rules = rules
        self.end_of_text = "<|endoftext|>"
        self.end_of_text_id = 256 + len(self.rules)

        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        for (id1, id2), new_id in self.rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]

        self.id_to_bytes[self.end_of_text_id] = self.end_of_text.encode('utf-8')

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

    paths = [
        './dataset/train-00000-of-00003.parquet',
        './dataset/train-00001-of-00003.parquet',
        './dataset/train-00002-of-00003.parquet',
    ]

    f = open('./output/train.bin', "wb")

    for path in paths:
        step = 0
        df = pd.read_parquet(path, columns=['text'])
        size = len(df["text"])
        for text in df["text"]:
            if step % 100 == 0:
                print(f"step: {step}, total: {size}")

            encoded = tokenizer.encode(text) + [tokenizer.end_of_text_id]
            f.write(np.array(encoded, dtype=np.uint16).tobytes())
            step += 1

    f.close()
```
I have three separated parquets (0~2). I will merge only the text column into one file. and don't forget to add the end mark between each row (Cuz the new file doesn't have separator)

Btw, the encode method takes too much time (40-50 mins). Let me improve the performance by caching

```python
# in __init__
self.cache = {}

def encode(self, data):
    result = []
    for token in BPEUtils.pretokenize(data):
        if token in self.cache:
            ids = self.cache[token]
        else:
            ids = list(token.encode('utf-8'))
            while len(ids) >= 2:
                pairs = set(zip(ids, ids[1:]))
                pair = min(pairs, key=lambda p: self.rules.get(p, float('inf')))
                if pair not in self.rules:
                    break

                ids = BPEUtils.merge(ids, pair, self.rules[pair])

            self.cache[token] = ids

        result.extend(ids)

    return result
```
I applied cache for each token. Cuz the merge results are always the same for each token

```Bash
---output---
step: 215000, total: 215965
step: 215100, total: 215965
step: 215200, total: 215965
step: 215300, total: 215965
step: 215400, total: 215965
step: 215500, total: 215965
step: 215600, total: 215965
step: 215700, total: 215965
step: 215800, total: 215965
step: 215900, total: 215965
encode: 366.44s
```

Now it takes about 6 mins.. hmm... not that bad

```Bash
(venv) ➜  output git:(main) ✗ ls -alh
total 1245552
drwxr-xr-x@  4 junpark  staff   128B Oct 27 16:49 .
drwxr-xr-x@ 16 junpark  staff   512B Oct 27 16:54 ..
-rw-r--r--@  1 junpark  staff   164K Oct 26 21:17 bpe_rules_16k.pkl
-rw-r--r--@  1 junpark  staff   600M Oct 27 16:55 train.bin
```

The size of train.bin is about 600M. Does it make sense?
First I tried to guess it with the parquet size (1331MB / 4.18), but parquet is a compressed format and it has other columns too. So it's not accurate.
Let me just count the tokens.

```python
ids = np.fromfile('./output/train.bin', dtype=np.uint16)
print(len(ids))

# ---output---
# 314381417
```

314M tokens. And I used `dtype=np.uint16`, so every token is 2 bytes.
314,381,417 × 2 = 628,762,834 bytes ≈ 600M. Same as the file size. good

(uint16 is enough cuz the max id is 15999 and uint16 can hold up to 65535. It's half the size of int32)

Now the data is ready. Next time, let's train a model with this data.



