---
title: 'A clean way to encode position: RoPE'
pubDate: 2026-10-24
tags: [dev]
math: true
---

## Intro
Attention doesn't care about the location of tokens
But in language, word order matters a lot. It can easily change the meaning

Here's an example
- Dogs love cats
- Cats love dogs

Since attention ignores position, `dogs` and `cats` have the same attention weight for `love`.

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

d = 4
v_dogs = torch.randn(1, d)
v_cats = torch.randn(1, d)
v_love = torch.randn(1, d)


# sentence 1: Dogs love cats
s1 = torch.cat([v_dogs, v_love, v_cats], dim=0)
# sentence 2: Cats love dogs
s2 = torch.cat([v_cats, v_love, v_dogs], dim=0)

W_q = torch.randn(d, d)
W_k = torch.randn(d, d)

def get_love_attention(seq):
    Q = seq @ W_q
    K = seq @ W_k
    scores = (Q @ K.T) / (d ** 0.5)
    probs = F.softmax(scores, dim=-1)
    return probs[1] # index of love

att1 = get_love_attention(s1)
att2 = get_love_attention(s2)

print("Attention weights from 'love' in 'Dogs love cats' [Dogs, love, cats]:")
print(att1.numpy().round(4))

print("\nAttention weights from 'love' in 'Cats love dogs' [Cats, love, dogs]:")
print(att2.numpy().round(4))

```
```Bash
---output---
Attention weights from 'love' in 'Dogs love cats' [Dogs, love, cats]:
[0.1419 0.1716 0.6865]

Attention weights from 'love' in 'Cats love dogs' [Cats, love, dogs]:
[0.6865 0.1716 0.1419]
```

As you can see, love gives 0.1419 to dogs and 0.6865 to cats in both sentences. Only the order changed.

## RoPE
Let's see how does RoPE handle this problem.
The basic idea is rotation. it rotates every 2-dimension pair of q and k according to the predefined angles (for now i'm calculating angles each time)


```Python
import math
import torch
import torch.nn.functional as F

torch.manual_seed(42)

d = 4

v_dogs = torch.randn(d)
v_cats = torch.randn(d)
v_love = torch.randn(d)

s1 = torch.stack([v_dogs, v_love, v_cats])  # Dogs love cats
s2 = torch.stack([v_cats, v_love, v_dogs])  # Cats love dogs

W_q = torch.randn(d, d)
W_k = torch.randn(d, d)

def rotate_2d(x, y, angle):
    new_x = x * math.cos(angle) - y * math.sin(angle)
    new_y = x * math.sin(angle) + y * math.cos(angle)
    return new_x, new_y

def apply_rope(vec, pos):
    out = vec.clone()
    # pair up neighboring dims: (0, 1), (2, 3), ...
    for i in range(d // 2):
        speed = 1.0 / (10000 ** (2 * i / d))  # each pair rotates at its own speed
        angle = pos * speed
        out[2 * i], out[2 * i + 1] = rotate_2d(vec[2 * i], vec[2 * i + 1], angle)
    return out

def get_love_attention_with_rope(seq):
    Q = seq @ W_q
    K = seq @ W_k

    # rotate each token's q and k by its position
    Q_rot = torch.stack([apply_rope(Q[pos], pos) for pos in range(len(seq))])
    K_rot = torch.stack([apply_rope(K[pos], pos) for pos in range(len(seq))])

    scores = (Q_rot @ K_rot.T) / math.sqrt(d)
    probs = F.softmax(scores, dim=-1)
    return probs[1]  # index of love

rope_att1 = get_love_attention_with_rope(s1)
rope_att2 = get_love_attention_with_rope(s2)

print("Attention weights from 'love' in 'Dogs love cats' [Dogs, love, cats]:")
print(rope_att1.numpy().round(4))

print("\nAttention weights from 'love' in 'Cats love dogs' [Cats, love, dogs]:")
print(rope_att2.numpy().round(4))
```
```Bash
---output---
Attention weights from 'love' in 'Dogs love cats' [Dogs, love, cats]:
[0.0607 0.0742 0.8652]

Attention weights from 'love' in 'Cats love dogs' [Cats, love, dogs]:
[0.2    0.4116 0.3884]
```

In this code, each token has a 4-dimension vector(cuz d is 4). In order to rotate them we should make them 2-dimension vectors to use a rotation equation. I just paired them sequentially({0,1}, {2,3}, {4,5}, ... ) but apparently HuggingFace and llama use a different method to pair them. at this moment, I don't know how much it's important.

Anyway
```Python
def apply_rope(vec, pos):
    out = vec.clone()
    for i in range(d // 2):
        speed = 1.0 / (10000 ** (2 * i / d))  # each pair rotates at its own speed
        angle = pos * speed
        out[2 * i], out[2 * i + 1] = rotate_2d(vec[2 * i], vec[2 * i + 1], angle)
    return out
```
each 2-dim vector will be rotated by different angles.
I was wondering what elements angle has
```Python
d = 4
theta = 10000
seq_len = 3
half_ids = torch.arange(0, d // 2)
inv_freq = 1.0 / (theta ** ((2.0 * half_ids) / d))
positions = torch.arange(seq_len)

angles = positions[:, None] * inv_freq[None, :]

print("Angles shape:", angles.shape)
print("Angles values:\n", angles)
```
```Bash
---output---
Angles shape: torch.Size([3, 2])
Angles values:
 tensor([[0.0000, 0.0000],
        [1.0000, 0.0100],
        [2.0000, 0.0200]])
```

3 rows because we have 3 tokens (dogs, love, cats)
2 cols because d is 4 (4 / 2 = 2 to be paired)


## Why did the weights change?

So why did the weights change? I calculated it by hand to see.

First, `love` is at position 1 in both sentences. So the query of `love` gets rotated by the same angles `[1.0, 0.01]` in both cases.

$$
q_{love} = [-0.7775,\ -0.5552,\ -1.0880,\ -0.0112]
$$

The first pair `(-0.7775, -0.5552)` rotates by 1 rad. cos 1 is 0.5403 and sin 1 is 0.8415, so

$$
\begin{aligned}
x' &= -0.7775 \times 0.5403 - (-0.5552) \times 0.8415 = 0.0471 \\
y' &= -0.7775 \times 0.8415 + (-0.5552) \times 0.5403 = -0.9543
\end{aligned}
$$

The second pair rotates by only 0.01 rad, so it almost doesn't move.

$$
q_{love}^{rot} = [0.0471,\ -0.9543,\ -1.0879,\ -0.0220]
$$

So far nothing is different between the two sentences. The difference comes from the keys.

In `Dogs love cats`, dogs is at position 0 so it doesn't rotate at all. cats is at position 2 so it rotates by `[2.0, 0.02]`.
In `Cats love dogs`, it's the opposite.

Then I calculated the scores (q · k / √4)

| | dogs | love | cats |
|---|---|---|---|
| without RoPE | -0.1758 | 0.0145 | 1.4009 |
| RoPE, Dogs love cats | -0.1870 | 0.0145 | 2.4707 |
| RoPE, Cats love dogs | -0.0434 | 0.0145 | -0.7073 |

`love → love` is 0.0145 in every case, even without RoPE. I think it's cuz query and key of `love` are rotated by the same angle, so the rotation cancels out.

But `cats` is totally different. 2.4707 when it comes after `love`, -0.7073 when it comes before `love`. Same word, but different score because of the position.

After softmax
```Bash
Dogs love cats: exp = [0.8294, 1.0146, 11.8310], sum = 13.6751
             → [0.0607, 0.0742, 0.8652]

Cats love dogs: exp = [0.4930, 1.0146, 0.9575], sum = 2.4651
             → [0.2000, 0.4116, 0.3884]
```

The weight of `love → love` changed too (0.0742 → 0.4116) but its score is the same. It's just because the sum of exp changed.

## What I noticed

If the rotation cancels out when the angle is the same, maybe the score only depends on the distance between tokens? So I tried this

```Python
q_love = v_love @ W_q
k_dogs = v_dogs @ W_k

print(apply_rope(q_love, 1) @ apply_rope(k_dogs, 0))  # distance 1
print(apply_rope(q_love, 5) @ apply_rope(k_dogs, 4))  # distance 1
```
```Bash
---output---
tensor(-0.3740)
tensor(-0.3740)
```

Same value. So absolute position doesn't matter, only the distance (query position - key position) matters.

And the first pair (1 rad per step) did almost all the work. For `cats`, the first pair changed from -0.94 to 2.16 but the second pair (0.01 rad per step) only changed from 0.23 to 0.31. I guess the slow pairs are for long distance and 2 tokens is too short for them.

## Simple Question
We're currently using pos and speed to calculate angles. What if every speed is the same?
Positions are still different though, 2-dim vectors for one token have the same angle.
what's the problem?? we know this is an angle. so 0 degree, 360 degree 720 degree are basically same. which means you can't recognize how much they are far from each other correctly in a sentence. this is one of the reasons the speed goes down for each pair. a longer sentence means a higher chance that two different distances have the same angle.
But if every speed is slow, close tokens get almost the same angle (0.01 vs 0.02), so you can't tell them apart either. That's why we need both fast and slow pairs

## Conclusion
So RoPE is a clean way how to apply the location information. It doesn't change V (values in Q, K, V) at all. we also can pre-calculate every angles. so it's pretty efficient

And what I learned is the score only depends on the distance between tokens, not the absolute position. And fast and slow pairs together let us tell both close and far distances apart.

Umm... Bye