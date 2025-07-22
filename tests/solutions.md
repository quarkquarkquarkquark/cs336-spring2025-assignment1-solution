# transformer_accounting

(a)

 Embedding layer and lm_head params:

$$
2vocab\_size\times d_{model}=2\times 50257\times 1600=160,822,400
$$

Transformer block params:

2 rms_norm layers:  $2d\_model=3200$

1 mha layer: $4d_{model}^2=10,240,000$

1 ffn layer: $3d_{ff} \times d_{model}=3\times 6400 \times 1600=30,720,000$

total: $num\_layers\times 40,963,200=1,966,233,600$

all model params:

$$
160822400+1966233600+1600=2127057600
$$

memory:

2127057600*4=8,508,230,400 bytes=8.5GB

(b)

Embedding layer: no matrix mutiplication

Transformer block:

    2 rms_norm: no matrix mutiplication

    1 mha:

$$
3*2*d_{model}*context\_length*d_{model} \\+num\_heads*2*context\_length*d_{head}*context\_length\\ +num\_heads*2*context\_length*context\_length*d_{head}\\ +2*context\_length*d_{model}*d_{model}\\ = 27,682,406,400
$$

    1 ffn:$6\ context\_length\times d_{model}\times d_{ff}= 62,914,560,000$

lm_head: $2 context\_length \times d_{model}\times vocab\_size=102,926,336$

total: 4,513,336,524,800=4.5TFLOPs

(c) ffn parts require most FLOPs.

(d)

|         | GPT-2 small | GPT-2 medium | GPT-2 large |
| ------- | ----------- | ------------ | ----------- |
| mha     | 0.097TFLOPs | 0.309TFLOPs  | 0.676TFLOPs |
| ffn     | 0.362TFLOPs | 0.966TFLOPs  | 1.812TFLOPs |
| lm_head | 0.079TFLOPs | 0.105TFLOPs  | 0.132TFLOPs |
| total   | 0.538TFLOPs | 1.381TFLOPs  | 2.620TFLOPs |

attention parts take more while lm_head parts take less propotion of total flops.

(e) According to the expression, the multi-head attention (MHA) layers contain a quadratic term in context length, while all other components depend only linearly on it. Consequently, when the context length is multiplied by 16, the FLOPs for the feed-forward (FFN) and language-model-head (LM-head) layers increase by a factor of 16, whereas the MHA FLOPs grow by a factor of 74.2.
