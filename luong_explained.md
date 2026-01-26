# Luong Attention: A Simple Explanation

Luong attention was introduced in "Effective Approaches to Attention-based Neural Machine Translation" (2015), just a few months after Bahdanau. It offers simpler, more efficient alternatives to Bahdanau's attention mechanism.

## Bahdanau vs Luong: Key Differences

| Aspect | Bahdanau (2015) | Luong (2015) |
|--------|----------------|--------------|
| **Timing** | Uses **previous** decoder state | Uses **current** decoder state |
| **Scoring** | Additive (concat + tanh) | Dot product, General, or Concat |
| **Complexity** | More parameters (3 weight matrices) | Simpler (0-2 weight matrices) |
| **Input to decoder** | Concatenates embedding + context | Concatenates after LSTM |
| **Variants** | One approach | Global and Local attention |

## The Problem Luong Addresses

While Bahdanau attention works well, it has some inefficiencies:
- Uses previous decoder state (s_t-1) instead of current (s_t)
- Requires multiple weight matrices (W_h, W_s, v)
- More complex computation

**Luong's solution:** Simplify the scoring function and use the current decoder state for better alignment.

## The Architecture

### Bahdanau Flow:
```
previous_hidden → attention → context → [concat with embedding] → LSTM → output
```

### Luong Flow:
```
embedding → LSTM → current_hidden → attention → context → [concat] → output
```

**Key difference:** Luong computes attention AFTER the LSTM step, not before.

## The Math: Three Scoring Functions

Luong proposes three ways to compute alignment scores:

### 1. Dot Product (Simplest)
```
score(s_t, h_i) = s_t^T · h_i
```
- No parameters to learn
- Requires encoder and decoder to have same hidden size
- Fastest computation

### 2. General (Most Common)
```
score(s_t, h_i) = s_t^T · W_a · h_i
```
- One learnable weight matrix W_a
- Can handle different hidden sizes
- Good balance of simplicity and flexibility

### 3. Concat (Similar to Bahdanau)
```
score(s_t, h_i) = v^T · tanh(W · [s_t; h_i])
```
- Similar to Bahdanau but uses current state
- More parameters, more expressive

### Common Steps (All Methods)
```
1. Compute scores for all encoder positions
2. attention_weights = softmax(scores)
3. context = sum(attention_weights * encoder_outputs)
4. Combined = tanh(W_c · [s_t; context])
5. output = W_out · Combined
```

## Global vs Local Attention

Luong introduces two attention variants:

### Global Attention (Default)
- Attends to ALL encoder positions
- Same as Bahdanau's approach
- Better for short-medium sequences

### Local Attention
- Attends to a **WINDOW** of encoder positions, not all
- **D** = window radius (hyperparameter, typically 5-10)
- Window: `[p_t - D, p_t + D]` → attend to 2D+1 positions only

**Two variants for choosing p_t (aligned position):**

1. **Monotonic (local-m)**: `p_t = t`
   - Simple: decoder position = encoder position
   - Good for similar word order (English ↔ French)

2. **Predictive (local-p)**: `p_t = S * sigmoid(v_p^T · tanh(W_p · h_t))`
   - Model predicts where to focus
   - Better for different word orders (English ↔ German)

**Example:**
```
Source: "The cat sat on the mat quickly" (length=7)
         0   1   2  3  4   5     6

D = 2, p_t = 4
Window: [4-2, 4+2] = [2, 6]
Attend to: positions 2, 3, 4, 5, 6 only (5 positions instead of 7)
```

**Benefits:**
- Computation: O(2D+1) instead of O(src_len)
- Much faster for long sequences (documents, paragraphs)
- Still captures relevant context

We'll focus on **Global Attention with General scoring** (most popular).

## Detailed Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                      ENCODER (LSTM)                      │
│   Input: "I love cats"                                   │
│   Output: h1, h2, h3, ..., h_n (all hidden states)      │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                        DECODER                           │
│                                                          │
│  Step 1: <SOS> → LSTM → s_1                             │
│          ↓                                               │
│  Step 2: Attention(s_1, encoder_outputs) → context_1    │
│          ↓                                               │
│  Step 3: Combined = tanh(W_c · [s_1; context_1])        │
│          ↓                                               │
│  Step 4: Output = W_out · Combined → "J'"               │
│                                                          │
│  Repeat for next token...                               │
└─────────────────────────────────────────────────────────┘
```

## Step-by-Step Process

### Encoder Phase
```python
# Encode entire source sentence
embedded = embedding(src)  # [src_len, batch, emb_dim]
encoder_outputs, (hidden, cell) = lstm(embedded)
# encoder_outputs: [src_len, batch, hidden_dim]
```

### Decoder Phase (Each Step)
```python
# 1. Embed current token
embedded = embedding(input)  # [1, batch, emb_dim]

# 2. LSTM step (THIS IS DIFFERENT FROM BAHDANAU)
output, (hidden, cell) = lstm(embedded, (hidden, cell))
# output: [1, batch, hidden_dim]

# 3. Compute attention using CURRENT hidden state
scores = encoder_outputs @ W_a @ hidden.T  # General scoring
attention_weights = softmax(scores)
context = sum(attention_weights * encoder_outputs)

# 4. Combine output + context
combined = tanh(W_c @ [output; context])

# 5. Predict next token
prediction = W_out @ combined
```

## Implementation Components

### 1. Encoder (Same as Bahdanau)
```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell
```

### 2. Luong Attention
```python
class LuongAttention(nn.Module):
    def __init__(self, hidden_dim, method='general'):
        super().__init__()
        self.method = method

        if method == 'general':
            self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif method == 'concat':
            self.W = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [1, batch, hidden]
        # encoder_outputs: [src_len, batch, hidden]

        if self.method == 'dot':
            # Dot product: h^T · s
            scores = torch.sum(encoder_outputs * decoder_hidden, dim=2)

        elif self.method == 'general':
            # General: h^T · W_a · s
            energy = self.W_a(decoder_hidden)  # [1, batch, hidden]
            scores = torch.sum(encoder_outputs * energy, dim=2)

        elif self.method == 'concat':
            # Concat: v^T · tanh(W · [h; s])
            src_len = encoder_outputs.shape[0]
            decoder_hidden = decoder_hidden.repeat(src_len, 1, 1)
            energy = torch.tanh(self.W(torch.cat([encoder_outputs, decoder_hidden], dim=2)))
            scores = self.v(energy).squeeze(2)

        # Normalize scores: [src_len, batch]
        attention_weights = F.softmax(scores, dim=0)

        # Context vector: [batch, hidden]
        context = (attention_weights.unsqueeze(2) * encoder_outputs).sum(dim=0)

        return context, attention_weights
```

### 3. Luong Decoder
```python
class LuongDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout, method='general'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)  # NO context input here!
        self.attention = LuongAttention(hidden_dim, method)

        # Combine layer: [hidden + context] → hidden
        self.W_c = nn.Linear(hidden_dim * 2, hidden_dim)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch]

        # 1. Embed
        input = input.unsqueeze(0)  # [1, batch]
        embedded = self.dropout(self.embedding(input))  # [1, batch, emb_dim]

        # 2. LSTM (NO context concatenation here, unlike Bahdanau!)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: [1, batch, hidden_dim]

        # 3. Attention using CURRENT hidden state
        context, attention_weights = self.attention(hidden, encoder_outputs)
        # context: [batch, hidden_dim]

        # 4. Combine output + context
        combined_input = torch.cat([output.squeeze(0), context], dim=1)
        combined = torch.tanh(self.W_c(combined_input))  # [batch, hidden_dim]

        # 5. Predict
        prediction = self.fc_out(combined)

        return prediction, hidden, cell, attention_weights
```

## Key Shapes to Remember

| Tensor | Shape | Description |
|--------|-------|-------------|
| `encoder_outputs` | [src_len, batch, hidden] | All encoder hidden states |
| `decoder_hidden` | [1, batch, hidden] | Current decoder state (AFTER LSTM) |
| `attention_weights` | [src_len, batch] | Alignment distribution |
| `context` | [batch, hidden] | Weighted encoder summary |
| `combined` | [batch, hidden] | Output + context combined |

## Why Luong Often Works Better

1. **Uses current state** - More accurate alignment since it uses s_t instead of s_t-1
2. **Simpler** - Fewer parameters (especially with dot/general scoring)
3. **Faster** - Less computation, especially with dot product
4. **Flexible** - Multiple scoring options to choose from
5. **Local variant** - Can handle very long sequences efficiently

## Bahdanau vs Luong: Side-by-Side

### Bahdanau Decoder Step:
```
1. Attention(previous_hidden, encoder_outputs) → context
2. LSTM([embedding; context]) → output
3. Predict(output, context) → next_token
```

### Luong Decoder Step:
```
1. LSTM(embedding) → output
2. Attention(output, encoder_outputs) → context
3. Combine(output, context) → combined
4. Predict(combined) → next_token
```

## When to Use Which?

**Use Bahdanau when:**
- You want the attention to guide what the LSTM sees
- You have limited data (more parameters = more capacity)

**Use Luong when:**
- You want faster training
- You prefer simpler architectures
- You need to handle very long sequences (use local attention)
- You want to experiment with different scoring functions

**In practice:** Both work well! Luong is more popular in modern implementations due to simplicity.

## Visualization

Attention weights for "I love cats" → "J'aime les chats":

```
              Source: "I    love   cats"
Target:             ┌──────────────────┐
  "J'"              │ 0.9   0.05  0.05 │  ← focuses on "I"
  "aime"            │ 0.05  0.9   0.05 │  ← focuses on "love"
  "les"             │ 0.05  0.05  0.9  │  ← focuses on "cats"
  "chats"           │ 0.05  0.1   0.85 │  ← focuses on "cats"
                    └──────────────────┘
```

## Implementation Checklist

- [ ] Encoder returns ALL hidden states
- [ ] Decoder does LSTM FIRST, then attention
- [ ] Attention uses CURRENT hidden state (not previous)
- [ ] Context is combined with LSTM output (not input)
- [ ] Choose scoring method: dot/general/concat
- [ ] Output layer uses combined vector

## Common Pitfalls

1. **Using previous hidden state** - Luong uses current! (after LSTM)
2. **Concatenating context to LSTM input** - That's Bahdanau! Luong concatenates after.
3. **Wrong hidden size for dot product** - Encoder and decoder must match for dot scoring
4. **Forgetting the combine layer** - Don't just concatenate, use W_c and tanh

## Next Steps

1. Implement Luong attention from scratch
2. Compare performance with Bahdanau on same dataset
3. Try different scoring functions (dot, general, concat)
4. Experiment with local attention for long sequences
5. Visualize attention weights to verify alignment

## References

- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) - Luong et al., 2015
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Bahdanau et al., 2015
- This implementation will use PyTorch and the Multi30k dataset for English-German translation
