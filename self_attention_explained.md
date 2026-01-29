# Self-Attention: A Simple Explanation

Self-attention is the core mechanism that powers Transformers (BERT, GPT, T5, etc.). It was introduced in "Attention is All You Need" (2017) and revolutionized deep learning by eliminating the need for recurrence.

## The Problem with Bahdanau/Luong

Both Bahdanau and Luong attention are **cross-attention** mechanisms:
- Queries come from the **decoder**
- Keys and Values come from the **encoder**
- Used for encoder-decoder tasks (translation, summarization)

**Limitations:**
1. Still relies on sequential RNN/LSTM processing
2. Can't parallelize within a sequence
3. Long-range dependencies still difficult (information flows through hidden states)
4. Decoder can attend to encoder, but encoder can't attend to itself

**What if we want each position in a sequence to attend to all other positions in the SAME sequence?**

## The Self-Attention Solution

**Self-attention** lets every position in a sequence attend to every other position in that **same sequence**.

```
Input: "The cat sat on the mat"

When processing "cat":
- Attend to "The" (determiner relationship)
- Attend to "cat" (self-reference)
- Attend to "sat" (subject-verb relationship)
- Attend to "mat" (indirect object relationship)
```

**Key insight:** Q, K, V all come from the **same source** (hence "self" attention).

## The Architecture Evolution

### Bahdanau/Luong (Cross-Attention):
```
Encoder sequence → Keys, Values
Decoder sequence → Queries

Queries attend to Keys/Values
```

### Self-Attention:
```
Input sequence → Queries, Keys, Values (all from same source)

Each position attends to all positions in the same sequence
```

## The Math: Query, Key, Value (QKV)

This is the heart of self-attention.

### Step 1: Create Q, K, V from Input
```
Input: X = [x_1, x_2, ..., x_n]  shape: [seq_len, d_model]

Q = X @ W_Q    # [seq_len, d_k]  - "What am I looking for?"
K = X @ W_K    # [seq_len, d_k]  - "What do I contain?"
V = X @ W_V    # [seq_len, d_v]  - "What do I provide?"
```

**W_Q, W_K, W_V** are learned weight matrices:
- W_Q: [d_model, d_k]
- W_K: [d_model, d_k]
- W_V: [d_model, d_v]

Typically, d_k = d_v = d_model // num_heads

### Step 2: Compute Attention Scores
```
scores = Q @ K^T    # [seq_len, seq_len]
```

Each entry scores[i, j] measures compatibility between position i's query and position j's key.

**Example:**
```
Sentence: "The cat sat"

         Q_the  Q_cat  Q_sat
K_the  [  .      .      .   ]
K_cat  [  .      .      .   ]  ← scores matrix
K_sat  [  .      .      .   ]

scores[1, 0] = similarity between "cat" query and "The" key
```

### Step 3: Scale the Scores
```
scores = scores / sqrt(d_k)
```

**Why scale?**
- When d_k is large, dot products grow in magnitude
- Large values push softmax into regions with tiny gradients (saturation)
- Scaling keeps values in a reasonable range

### Step 4: Apply Softmax
```
attention_weights = softmax(scores, dim=-1)    # [seq_len, seq_len]
```

Each row is a probability distribution summing to 1.

**Example:**
```
           The   cat   sat
The     [ 0.7   0.2   0.1 ]
cat     [ 0.3   0.5   0.2 ]  ← "cat" attends 50% to itself
sat     [ 0.1   0.4   0.5 ]
```

### Step 5: Weighted Sum of Values
```
output = attention_weights @ V    # [seq_len, d_v]
```

**Complete Formula:**
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

## Detailed Example

Input: "The cat sat"

Embeddings (simplified, d_model=4):
```
X = [
  [0.1, 0.2, 0.3, 0.4],  ← "The"
  [0.5, 0.6, 0.7, 0.8],  ← "cat"
  [0.9, 1.0, 1.1, 1.2]   ← "sat"
]
```

Learned weights (simplified):
```
W_Q = random matrix [4, 4]
W_K = random matrix [4, 4]
W_V = random matrix [4, 4]
```

**Forward pass:**
```
1. Q = X @ W_Q → [3, 4]
2. K = X @ W_K → [3, 4]
3. V = X @ W_V → [3, 4]

4. scores = Q @ K^T → [3, 3]
   [[20.5, 45.3, 70.1],
    [45.3, 102.5, 159.7],
    [70.1, 159.7, 249.3]]

5. scores = scores / sqrt(4) = scores / 2
   [[10.25, 22.65, 35.05],
    [22.65, 51.25, 79.85],
    [35.05, 79.85, 124.65]]

6. attention_weights = softmax(scores)
   [[0.0, 0.0, 1.0],      ← "The" attends to "sat"
    [0.0, 0.0, 1.0],      ← "cat" attends to "sat"
    [0.0, 0.0, 1.0]]      ← "sat" attends to "sat"

7. output = attention_weights @ V
   Each position gets a weighted sum of all value vectors
```

## Why "Self" Attention?

| Aspect | Cross-Attention (Bahdanau/Luong) | Self-Attention |
|--------|----------------------------------|----------------|
| **Q source** | Decoder hidden states | Same sequence |
| **K source** | Encoder hidden states | Same sequence |
| **V source** | Encoder hidden states | Same sequence |
| **Purpose** | Decoder attends to encoder | Sequence attends to itself |
| **Use case** | Translation, summarization | Contextual encoding |

## Multi-Head Attention (Extending Self-Attention)

Instead of one attention, use **multiple heads** in parallel.

**Single head:** One set of W_Q, W_K, W_V
**Multi-head:** h sets of W_Q^i, W_K^i, W_V^i

```
head_1 = Attention(Q @ W_Q^1, K @ W_K^1, V @ W_V^1)
head_2 = Attention(Q @ W_Q^2, K @ W_K^2, V @ W_V^2)
...
head_h = Attention(Q @ W_Q^h, K @ W_K^h, V @ W_V^h)

MultiHead = Concat(head_1, ..., head_h) @ W_O
```

**Why multiple heads?**
- Different heads learn different types of relationships:
  - Head 1: Syntactic dependencies (subject-verb)
  - Head 2: Semantic similarity (synonyms)
  - Head 3: Positional relationships (nearby words)
  - Head 4: Coreference (pronouns to nouns)

**Dimension splitting:**
If d_model = 512 and num_heads = 8:
- Each head has d_k = d_v = 512/8 = 64
- Cheaper than 8 full-size heads!
- Total computation ≈ single-head attention

## Masked Self-Attention (For Language Modeling)

In GPT-style models, we can't let a word see future words during training.

**Solution:** Mask out future positions with -inf before softmax.

```
scores = Q @ K^T / sqrt(d_k)

mask = [
  [0,   -inf, -inf],    ← "The" can only see "The"
  [0,    0,   -inf],    ← "cat" can see "The", "cat"
  [0,    0,    0   ]    ← "sat" can see all
]

scores = scores + mask

attention_weights = softmax(scores)
# Future positions will have 0 attention weight after softmax
```

**Result:** Causal/autoregressive attention where each position only attends to past positions.

## Self-Attention vs RNN/LSTM

| Aspect | RNN/LSTM | Self-Attention |
|--------|----------|----------------|
| **Parallelization** | Sequential (one at a time) | Fully parallel |
| **Path length** | O(n) hops | O(1) direct connection |
| **Long-range deps** | Degrades over distance | Same strength regardless |
| **Complexity** | O(n) time, O(1) memory | O(n²) time and memory |
| **Position info** | Inherent (sequential) | Must add (positional encoding) |

**Trade-off:** Self-attention is O(n²) in sequence length, expensive for very long sequences.

## Positional Encoding (Critical!)

Self-attention is **permutation invariant** - it doesn't know word order!

```
"The cat sat" = "sat cat The" (to self-attention, without positional info)
```

**Solution:** Add positional encodings to input embeddings.

```
# Original (Vaswani et al., 2017)
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

# Modern (learned)
positional_embedding = nn.Embedding(max_seq_len, d_model)
```

Final input:
```
X_with_pos = X_embeddings + Positional_Encodings
```

## Complete Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT SEQUENCE                          │
│            "The cat sat on the mat"                          │
│         [x_1, x_2, x_3, x_4, x_5, x_6]                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  EMBED + POSITIONAL ENCODING                 │
│         X_emb = Embedding(X) + PositionalEncoding(X)        │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
         ┌──────────┐  ┌──────────┐  ┌──────────┐
         │  X @ W_Q │  │  X @ W_K │  │  X @ W_V │
         └──────────┘  └──────────┘  └──────────┘
                ▼             ▼             ▼
              Query         Key          Value
           [seq, d_k]   [seq, d_k]   [seq, d_v]
                │             │             │
                └──────┬──────┘             │
                       ▼                    │
            ┌─────────────────────┐         │
            │  scores = Q @ K^T   │         │
            │  scores /= sqrt(d_k)│         │
            └─────────────────────┘         │
                       │                    │
                       ▼                    │
            ┌─────────────────────┐         │
            │  attention_weights  │         │
            │  = softmax(scores)  │         │
            └─────────────────────┘         │
                       │                    │
                       └─────────┬──────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │  output = weights @ V    │
                    │     [seq_len, d_v]       │
                    └──────────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   CONTEXTUAL ENCODING    │
                    │  Each word now has info  │
                    │   from all other words   │
                    └──────────────────────────┘
```

## Implementation Components

### 1. Scaled Dot-Product Attention
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [batch, seq_len, d_k]
    K: [batch, seq_len, d_k]
    V: [batch, seq_len, d_v]
    mask: [batch, seq_len, seq_len] or [seq_len, seq_len]
    """
    d_k = Q.size(-1)

    # Compute scores: [batch, seq_len, seq_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask (optional)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Weighted sum
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

### 2. Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Output linear layer
        self.W_O = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """Split into multiple heads"""
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()

        # Reshape to [batch, seq_len, num_heads, d_k]
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose to [batch, num_heads, seq_len, d_k]
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """Combine heads back"""
        # x: [batch, num_heads, seq_len, d_k]
        batch_size, num_heads, seq_len, d_k = x.size()

        # Transpose to [batch, seq_len, num_heads, d_k]
        x = x.transpose(1, 2)

        # Reshape to [batch, seq_len, d_model]
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Linear projections
        Q = self.W_Q(Q)  # [batch, seq_len, d_model]
        K = self.W_K(K)
        V = self.W_V(V)

        # Split into multiple heads
        Q = self.split_heads(Q)  # [batch, num_heads, seq_len, d_k]
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Scaled dot-product attention
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Expand mask for heads: [batch, 1, seq_len, seq_len]
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # Combine heads
        output = self.combine_heads(output)  # [batch, seq_len, d_model]

        # Final linear projection
        output = self.W_O(output)

        return output, attention_weights
```

### 3. Positional Encoding
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
```

### 4. Complete Self-Attention Layer
```python
class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention: Q, K, V all from same source
        attn_output, attn_weights = self.attention(x, x, x, mask)

        # Residual connection + LayerNorm
        x = self.norm(x + self.dropout(attn_output))

        return x, attn_weights
```

## Key Shapes to Remember

| Tensor | Shape | Description |
|--------|-------|-------------|
| `X` | [batch, seq_len, d_model] | Input embeddings |
| `Q` | [batch, seq_len, d_k] | Queries |
| `K` | [batch, seq_len, d_k] | Keys |
| `V` | [batch, seq_len, d_v] | Values |
| `scores` | [batch, seq_len, seq_len] | Attention scores |
| `attention_weights` | [batch, seq_len, seq_len] | After softmax |
| `output` | [batch, seq_len, d_v] | Weighted sum of values |
| `mask` | [batch, seq_len, seq_len] | Optional mask |

For multi-head:
- After split_heads: [batch, num_heads, seq_len, d_k]
- After combine_heads: [batch, seq_len, d_model]

## Visualization Example

Attention weights for "The animal didn't cross the street because it was too tired"

```
               The  animal  didn't  cross  the  street  because  it  was  too  tired
The          [ 0.5   0.1     0.1     0.05   0.1   0.05    0.05   0.05  0.0   0.0   0.0 ]
animal       [ 0.1   0.6     0.1     0.05   0.05  0.05    0.0    0.05  0.0   0.0   0.0 ]
didn't       [ 0.05  0.2     0.5     0.15   0.05  0.05    0.0    0.0   0.0   0.0   0.0 ]
cross        [ 0.05  0.3     0.2     0.3    0.05  0.1     0.0    0.0   0.0   0.0   0.0 ]
the          [ 0.05  0.05    0.05    0.05   0.6   0.2     0.0    0.0   0.0   0.0   0.0 ]
street       [ 0.0   0.05    0.05    0.1    0.2   0.6     0.0    0.0   0.0   0.0   0.0 ]
because      [ 0.1   0.1     0.3     0.1    0.05  0.05    0.2    0.05  0.05  0.0   0.0 ]
it           [ 0.05  0.7     0.05    0.05   0.05  0.05    0.0    0.05  0.0   0.0   0.0 ]  ← attends to "animal"!
was          [ 0.0   0.05    0.1     0.0    0.0   0.0     0.05   0.3   0.5   0.0   0.0 ]
too          [ 0.0   0.0     0.0     0.0    0.0   0.0     0.0    0.0   0.3   0.5   0.2 ]
tired        [ 0.0   0.1     0.1     0.0    0.0   0.0     0.0    0.2   0.1   0.1   0.4 ]
```

Notice how "it" attends strongly to "animal" (0.7), resolving the coreference!

## Transformer Block (Self-Attention + FFN)

Self-attention is usually combined with a feed-forward network:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-head self-attention
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual + norm
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # FFN + residual + norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x
```

## Advantages of Self-Attention

1. **Parallelization** - All positions computed simultaneously
2. **Long-range dependencies** - Direct O(1) path between any two positions
3. **Interpretability** - Attention weights show relationships
4. **Flexibility** - Same mechanism for encoding and decoding
5. **No vanishing gradients** - Direct gradient flow

## Limitations

1. **Quadratic complexity** - O(n²) in sequence length
   - Problem for very long sequences (10k+ tokens)
   - Solutions: Sparse attention, Linformer, Reformer

2. **No inherent positional info** - Must add positional encodings

3. **Memory intensive** - Stores full attention matrix

4. **Requires more data** - No inductive bias (unlike CNNs/RNNs)

## Self-Attention vs Bahdanau/Luong

| Aspect | Bahdanau/Luong | Self-Attention |
|--------|----------------|----------------|
| **Type** | Cross-attention | Self-attention |
| **Q from** | Decoder | Same sequence |
| **K, V from** | Encoder | Same sequence |
| **Sequential?** | Yes (RNN-based) | No (fully parallel) |
| **Complexity** | O(n) per position | O(n²) total |
| **Use case** | Encoder-decoder | Contextual encoding |
| **Position info** | Inherent (RNN) | Must add explicitly |

## Common Pitfalls

1. **Forgetting positional encoding** - Self-attention is permutation invariant!
2. **Wrong dimensions for multi-head** - d_model must be divisible by num_heads
3. **Not scaling scores** - Leads to vanishing gradients in softmax
4. **Incorrect mask shape** - Must broadcast correctly for multi-head
5. **Forgetting residual connections** - Critical for training deep models

## Implementation Checklist

- [ ] Input embeddings + positional encoding
- [ ] Linear projections for Q, K, V
- [ ] Split into multiple heads (for multi-head)
- [ ] Compute scores: Q @ K^T
- [ ] Scale by sqrt(d_k)
- [ ] Apply mask (if needed)
- [ ] Softmax to get attention weights
- [ ] Weighted sum of values
- [ ] Combine heads
- [ ] Output projection
- [ ] Residual connection + layer norm

## Use Cases

**BERT (Encoder-only):**
- Bidirectional self-attention (no masking)
- Sentence classification, NER, QA

**GPT (Decoder-only):**
- Masked self-attention (causal)
- Text generation, completion

**T5/BART (Encoder-Decoder):**
- Encoder: Self-attention (bidirectional)
- Decoder: Masked self-attention + cross-attention to encoder
- Translation, summarization

## Next Steps

1. Implement scaled dot-product attention from scratch
2. Build multi-head attention
3. Add positional encoding
4. Combine into a complete Transformer block
5. Test on a simple task (sentiment analysis, language modeling)
6. Visualize attention weights
7. Compare with RNN-based models

## References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017 (The Transformer paper)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Devlin et al., 2018
- [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) - Radford et al., 2018 (GPT)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar (excellent visual guide)
