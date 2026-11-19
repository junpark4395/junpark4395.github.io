---
title: 'Tiny Korean poetry generator #3: pretrain and fine tuning'
pubDate: 2026-11-19
tags: [dev]
---

In the previous post, I tokenized Korean Wikipedia into `train.bin` (314M tokens).
Now it's time to train a model with this data.

First I'll pretrain a tiny GPT model on Wikipedia so it learns basic Korean. Then I'll fine-tune it on Korean poems.

## Model
Before looking at the training code, let me briefly go over the model (`model.py` is on [GitHub](https://github.com/junpark4395/tiny-korean-poetry-generator)).
It's a basic GPT-style decoder-only transformer. Token embedding + positional embedding, 6 transformer blocks (Pre-LN, causal multi-head attention, GELU FFN), and weight tying between embedding and unembedding.

Here are the model hyperparameters.

```python
vocab_size = 16000
context_len = 256
embed_dim = 384
n_layer = 6
n_head = 6
ff_dim = 4 * embed_dim
```

`vocab_size` is 16,000 cuz that's the tokenizer I made in post #1 (256 bytes + 15,743 merge rules + 1 `<|endoftext|>` token).

`context_len` is 256. Attention gets quadratically slower and heavier as the sequence gets longer, so 256 is a realistic limit on my Mac. And the poems I'll fine-tune on are about 240 tokens on average anyway, so 256 is enough to fit a poem.

`embed_dim = 384`, `n_layer = 6`, `n_head = 6` gives `head_dim = 64` (384 / 6), and `ff_dim` is `4 * embed_dim` (1536), which is the usual transformer setup.

With this config, the total parameter count is `16,896,640` (16.9M).
Why this size? Chinchilla scaling says roughly 20 tokens per parameter. I have 314M tokens, so a 16.9M model (`16.9M × 20 ≈ 338M`) is right around that range. And it's small enough to train on my Mac (`device = 'mps'`).

## Pretraining
Here is `pre_train.py`.

```python
import os
import torch
import numpy as np
import torch.nn.functional as F
from model import GPT

vocab_size=16000
context_len = 256
embed_dim = 384
n_layer = 6
n_head = 6
batch_size = 32
ff_dim = 4 * embed_dim
device = 'mps'
learning_rate = 1e-3
max_iters = 38000
model_save_path = './output/model_pretrain.pt'
check_point_path = './output/checkpoint.pt'


def load_data(ids):
    size = len(ids)
    idx_list = torch.randint(0, size - context_len, (batch_size,)).tolist()

    batch_x = np.stack([ids[idx:idx + context_len] for idx in idx_list])
    batch_y = np.stack([ids[idx + 1:idx + context_len + 1] for idx in idx_list])

    batch_x = torch.from_numpy(batch_x.astype(np.int64)).to(device)
    batch_y = torch.from_numpy(batch_y.astype(np.int64)).to(device)

    return batch_x, batch_y

@torch.no_grad()
def estimate_loss(model, eval_ids):
    eval_iters = 50
    eval_losses = []
    for i in range(eval_iters):
        batch_x, batch_y = load_data(eval_ids)

        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

        eval_losses.append(loss.item())

    return sum(eval_losses) / eval_iters

if __name__ == "__main__":
    ids = np.memmap('./output/train.bin', dtype=np.uint16, mode='r')

    model = GPT(
        vocab_size=vocab_size,
        max_context_len=context_len,
        embed_dim=embed_dim,
        n_head=n_head,
        n_layer=n_layer,
        ff_dim=ff_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    start_step = 0
    if os.path.exists(check_point_path):
        checkpoint = torch.load(check_point_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint['step']
        print(f"resume from step {start_step}")

    losses = []
    log_interval = 100
    eval_interval = 500

    size = int(len(ids) * 0.99)
    train_ids = ids[:size]
    eval_ids = ids[size:]

    for i in range(start_step, max_iters):
        batch_x, batch_y = load_data(train_ids)

        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (i + 1) % log_interval == 0:
            avg_loss = sum(losses[-log_interval:]) / log_interval
            print(f"step {i + 1:>6}/{max_iters} | loss {avg_loss:.4f}")

        if (i + 1) % eval_interval == 0:
            eval_avg_loss = estimate_loss(model, eval_ids)
            print(f"step {i + 1:>6}/{max_iters} | eval loss {eval_avg_loss:.4f}")

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': i+1
            }
            torch.save(checkpoint, check_point_path)

    model.save(model_save_path)
```

Let me explain why I picked the training settings.

`batch_size` is 32, so each step sees `32 × 256 = 8,192` tokens. Going bigger slows down each step and uses more memory on MPS, and going too small makes the gradients noisy, so 32 felt right.

`learning_rate` is `1e-3`, a common starting value for small GPT models with AdamW. I didn't add a scheduler, just kept it constant to keep things simple.

`max_iters` is 38,000. I split the data 99% train / 1% eval (1% of 314M is still ~3.1M tokens, plenty for eval).
`314M × 0.99 / 8,192 ≈ 38,000`. So 38,000 steps is roughly 1 epoch over Wikipedia.

Also, `train.bin` is 600MB so I used `np.memmap` instead of loading the whole file into memory, and randomly sliced chunks of 256 tokens (`batch_y` is just `batch_x` shifted by 1). And since training takes hours on a laptop, I save `checkpoint.pt` every 500 steps so I can resume if it stops.

Let's see the result.

```bash
---output---
number of parameters: 16,896,640 (16.9M)
step    100/38000 | loss 8.0574
step    500/38000 | loss 6.6456
step    500/38000 | eval loss 6.6394
step   1000/38000 | loss 5.8538
step   1000/38000 | eval loss 5.9704
...
step   5000/38000 | loss 4.1248
step   5000/38000 | eval loss 4.2927
step  10000/38000 | loss 3.8112
step  10000/38000 | eval loss 3.9243
step  20000/38000 | loss 3.5744
step  20000/38000 | eval loss 3.7452
step  30000/38000 | loss 3.4687
step  30000/38000 | eval loss 3.6471
...
step  37500/38000 | eval loss 3.6219
step  37600/38000 | loss 3.4282
step  37700/38000 | loss 3.4502
step  37800/38000 | loss 3.3989
step  37900/38000 | loss 3.4072
step  38000/38000 | loss 3.3957
step  38000/38000 | eval loss 3.5959
```

Random guessing across 16,000 tokens gives `ln(16000) ≈ 9.68`.
The loss drops fast at first (8.05 → 4.12 in the first 5,000 steps) and then slowly goes down to `loss 3.3957` and `eval loss 3.5959` at step 38,000.

Train loss and eval loss stay pretty close, so it's not overfitting. The saved model `model_pretrain.pt` is about 64MB. Not bad.

## Fine-tuning
Now the model knows Korean, at least Wikipedia-style Korean. To make it write poems, I need poem data.

I collected 1,460 Korean modern poems from [KPoEM](https://huggingface.co/datasets/AKS-DHLAB/KPoEM) and Korean Wikisource (public domain poets like Kim Sowol, Yoon Dongju, etc.) and tokenized them with the same BPE tokenizer into `poem_train.bin`. (the scripts are on GitHub)

`poem_train.bin` has 354,944 tokens (~693KB). Way smaller than Wikipedia's 314M tokens.

Here is `fine_tune.py`. (`load_data` and `estimate_loss` are the same as `pre_train.py`, so I skipped them here)

```python
learning_rate = 1e-4
max_iters = 750
model_save_path = './output/model_pretrain_fine_tune.pt'

# load_data and estimate_loss are the same as pre_train.py

if __name__ == "__main__":
    ids = np.memmap('./output/poem_train.bin', dtype=np.uint16, mode='r')

    model = GPT.load_from('./output/model_pretrain.pt', device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    losses = []
    log_interval = 10
    eval_interval = 50

    size = int(len(ids) * 0.90)
    train_ids = ids[:size]
    eval_ids = ids[size:]

    for i in range(max_iters):
        batch_x, batch_y = load_data(train_ids)

        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (i + 1) % log_interval == 0:
            avg_loss = sum(losses[-log_interval:]) / log_interval
            print(f"step {i + 1:>6}/{max_iters} | loss {avg_loss:.4f}")

        if (i + 1) % eval_interval == 0:
            eval_avg_loss = estimate_loss(model, eval_ids)
            print(f"step {i + 1:>6}/{max_iters} | eval loss {eval_avg_loss:.4f}")

    model.save(model_save_path)
```

Instead of starting from scratch, I load `model_pretrain.pt` and train with a 10x smaller learning rate (`1e-4`). I don't want to ruin what it already learned from Wikipedia.

For the split, I used 90% / 10% cuz the poem dataset is tiny (1% would be only ~3.5k tokens).
90% of 355k tokens is ~319k tokens, so 1 epoch is only about 39 steps (`319k / 8,192`). That means `max_iters = 750` is around 19 epochs.

Let's check the output.

```bash
---output---
step     10/750 | loss 4.9983
step     50/750 | loss 4.6682
step     50/750 | eval loss 4.3190
step    100/750 | loss 4.5228
step    100/750 | eval loss 4.1600
step    200/750 | loss 4.3102
step    200/750 | eval loss 4.0594
step    350/750 | loss 4.0948
step    350/750 | eval loss 3.9635
step    450/750 | loss 4.0291
step    450/750 | eval loss 3.9352
step    550/750 | loss 3.8806
step    550/750 | eval loss 3.9422
step    650/750 | loss 3.7198
step    650/750 | eval loss 3.9189
step    700/750 | loss 3.6840
step    700/750 | eval loss 3.9459
step    750/750 | loss 3.6661
step    750/750 | eval loss 3.9689
```

At step 10, the loss starts around 4.99. Higher than Wikipedia's 3.4, which makes sense cuz poems look totally different from Wikipedia articles (short lines, lots of line breaks, old poetic words).

Train loss keeps dropping down to `3.6661`, while eval loss hits its lowest point at step 650 (`3.9189`) and starts going back up after that (`3.9459` at 700, `3.9689` at 750). It's starting to overfit, which is expected with only 355k tokens. So stopping around 750 steps makes sense.

## Generating poems
Let's see what it actually writes.
I ran `generator.py` (on GitHub) with `prompt = '여름'` ("summer"), `temperature = 0.8`, and `max_new_tokens = 250`.

```bash
---output---
--- sample 1 ---
여름하고 길에 있는 것이니?

이제 모를 때는 어둡게도 번뜩거려도
나는 근심에 언제나 이 일을 하던가?
그들이 더러는 소리도 같이 잘 가만히도다

이 세상엔 혼아지 나려도 나라요 이 세상엔 인생의 모든 것입니다
그러나 나의 생각은 나의 생각에서 따랐으랴
그래서 나의 생각은 나의 생각이 나의 생각입니다
그리하면 나의 생각은 나의 생각에 있으랴

--- sample 2 ---
여름불 속에 가볍게 날려버린
모든것은 빗겨지고 눈물만 누워 발돋움하여

아아 아침의 산과 봄
봄밤의 달빛이여
별 위에 고요히 빛나니
꽃밤에 새여 울던
비물처럼 심어서
수많은 수수께끼에 쥐어
우리는 그때에 슬퍼앉아
아동이가 다시 돌아가는
마지막에도 고요히 빛나니
비가여!

그날 하루도 밤마다 잊어버리고
꽃밤이여! 오늘도 보드라운
애닯은 두 눈만 앗아을 덮어주네.

--- sample 3 ---
여름을 순철한 마음이 오로지며서
반쯤은 취한 자여운가 하룻날지라도 그립은 아지라서

함께서는 꿈인 동무여!
진달래 꿈을 열어보아라, 우리 모련아!

이 마음만은 그대여!
슬픈 왕관을 사랑하랴!
나는 천사에 기우고갔노라, 그대여!
나는 사랑한다 하여도 원수없다, 너는 혼을림없이
시실어서는 떠도는 그대를 얼마나 힘없이 남겨본다.

--- sample 4 ---
여름의 자리우에
≪여기야!≫ 하차한 아침,
(이에도 사람은, 그대는 그대의 이대로
이 다니는 노래는 그대의 노래라서,
포푸토를 여기는 나는 이 싸움의 시원한 파리다,
트레지아!≫ 그대는 그대의 노래라고
온실! 그대의 앉던 어느 날,
당신이 그대의 눈을 실은 없음을,
「벗어붙은 보살입처럼 살아서 있어라,
조그만 내 깨어진 눈물이 아니라며,
잠은 뫼 위에 떠오르는 우리 몸이 아니라!」
「벗 어째서 내가 나의 가슴에서 나의 몸에
떨어지는 밤, 나의 가슴속에,
우리 눈에, 우리의 가슴은 그대의 입술로
바다 속 깊이 흐르는 기쁨의 마음의 옛적에 깊이 묻힌 체,
그것은 불쌍한 웃음을 우러뜨리고
굼벗은 얼굴 아래,
우리 눈에, 우리의 가슴은 그대의 가슴과 같지 않으매라.

≪어째서 나의 가슴이 나의 가슴속에…

--- sample 5 ---
여름이 높은 물방울에장
바늘은 ‘삐’에결가하는 봄바람에
천사 봄바람을 들으며
성금질-새벽을 깨우는 아름다운
은비-듸아바련 나는 껌에오.
양치기 울고는 오면서
철호 우에 있어 울어 가며
공중에 향내를 헤치며
허망과 찾도록
그만 그 길을 잃고
선댕이 비둘기를 끊고
장미꽃 같은 꽃에 걸린
이낒은 활활에
아미는 사람아!

(미녀)
오오!

아이, 늘아,
봄이여,
어쩌다, 차마……

아, 아내는 다시 한 번이고
기쁨에 드리는 누이.

바람이여,
그만 내 가슴에
꽃다지, 꽃다지
아내, 단조하는 누이.

바람이여,
길게도, 한울한가.

바람이여,
봄이여,
사라지여!
아, 나도 내
```

Umm... if you actually read it, it doesn't really make sense.
There are made-up words (`혼아지`, `비물`, `순철한`), weird grammar, and repetition loops like `나의 생각은 나의 생각이 나의 생각입니다` ("my thought is my thought is my thought").

Still, at a glance, it really looks like a Korean poem.
Short lines, stanza breaks, poetic endings (`~이여!`, `~도다`, `~노라`, `~으랴`), lines like `봄밤의 달빛이여 / 별 위에 고요히 빛나니` ("moonlight of a spring night / quietly shining above the stars"), and even old-school punctuation (`≪ ≫`, `「 」`) from the early 1900s poets in the dataset. Not bad for a 16.9M model trained from scratch on a laptop.

Why is the quality only at this level?

First, the model is just too small (16.9M params). It can learn local word patterns, basic grammar, and the shape of a poem, but it doesn't have enough capacity to keep a coherent meaning from start to finish.

Second, the poem dataset is tiny (1,460 poems, 355k tokens). With that little data, it overfits easily and gets stuck repeating phrases. And since the poems are mostly from the early 1900s, it also copies old expressions and punctuation.

On top of that, pretraining was only 1 epoch on Wikipedia with a short context length (256), constant learning rate, and no regularization. And `generator.py` only uses plain temperature sampling (`0.8`) with no top-k, top-p, or repetition penalty, so weird tokens and loops slip through easily.

## Stopping here
I know there are a lot of things I could improve.
Learning rate warmup and cosine decay, gradient clipping, dropout, mixed precision, saving the best eval checkpoint instead of the last step, better sampling, or upgrading the model itself (RoPE, RMSNorm, SwiGLU, GQA...).

All of those would make the loss lower and the poems better for sure.
But my goal isn't to build a great poetry model. My goal is to build an inference engine in C++ and run this model on it. It already speaks Korean and writes something that looks like a poem, so that's enough for me. I'm not going to spend more time improving the model or training.

Now I have a working model (`model_pretrain_fine_tune.pt`). Next time, let's start building an inference engine in C++ for it.
