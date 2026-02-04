# Attention is All You Need: The Transformer Architecture

This document explains the complete Transformer architecture from the groundbreaking 2017 paper "Attention is All You Need" by Vaswani et al.

## What Problem Does It Solve?

**Before Transformers:**
- RNN/LSTM: Sequential processing, slow, vanishing gradients
- Bahdanau/Luong: Better, but still relies on RNNs
- CNNs: Parallel but struggle with long-range dependencies

**Transformer Solution:**
- **NO recurrence** - fully parallel processing
- **Self-attention** - direct connections between all positions
- **Stacked layers** - deep architectures that actually train well
- **Fast & effective** - revolutionized NLP

## The Big Picture

The Transformer is an **encoder-decoder** architecture:

```
                    ENCODER-DECODER ARCHITECTURE

INPUT (Source)                                    OUTPUT (Target)
"I love cats"                                     "<SOS> Ich liebe Katzen"
     ↓                                                      ↓
┌─────────────────────┐                          ┌─────────────────────┐
│   INPUT EMBEDDING   │                          │  OUTPUT EMBEDDING   │
│  + POS ENCODING     │                          │  + POS ENCODING     │
└─────────────────────┘                          └─────────────────────┘
     ↓                                                      ↓
┌─────────────────────┐                          ┌─────────────────────┐
│   ENCODER STACK     │                          │   DECODER STACK     │
│   (N=6 layers)      │                          │   (N=6 layers)      │
│                     │                          │                     │
│  ┌───────────────┐  │                          │  ┌───────────────┐  │
│  │ Self-Attention│  │                          │  │Masked Self-   │  │
│  │   +           │  │                          │  │  Attention    │  │
│  │ Feed-Forward  │  │──────────────────────────┼─▶│      +        │  │
│  └───────────────┘  │    Encoder Outputs       │  │Cross-Attention│  │
│        × 6          │    (Keys & Values)       │  │      +        │  │
└─────────────────────┘                          │  │ Feed-Forward  │  │
                                                 │  └───────────────┘  │
                                                 │        × 6          │
                                                 └─────────────────────┘
                                                           ↓
                                                 ┌─────────────────────┐
                                                 │   LINEAR + SOFTMAX  │
                                                 └─────────────────────┘
                                                           ↓
                                                    "Ich liebe Katzen"
```

## Core Components

### 1. Multi-Head Self-Attention

**Why multiple heads?**
Different heads learn different relationships:
- Head 1: Syntactic structure (subject-verb-object)
- Head 2: Semantic similarity
- Head 3: Positional relationships
- Head 4: Long-range dependencies
- etc.

**Formula:**
```
MultiHead(Q, K, V) = Concat(head₁, ..., head_h) @ W_O

where head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)
```

**Key insight:** Split d_model into h heads:
- d_model = 512, h = 8 → each head has d_k = d_v = 64
- Total computation ≈ single-head with d_k = 512
- But learns h different representations!

### 2. Positional Encoding

**Why It's Needed:**
Self-attention is permutation-invariant (treats input as a set, not sequence). Without positional encoding, "I love cats" and "cats love I" would be identical to the model.

**Sinusoidal encoding (original paper):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` = position in sequence (0, 1, 2, ...)
- `i` = dimension index
- Even dimensions (0, 2, 4...) use **sine**
- Odd dimensions (1, 3, 5...) use **cosine**

**Concrete Example:**

Let's say `d_model = 4` and we have 3 tokens: `["I", "love", "cats"]` → positions `[0, 1, 2]`

For position 0 ("I"):
```
PE(0, 0) = sin(0 / 10000^(0/4)) = sin(0) = 0.000
PE(0, 1) = cos(0 / 10000^(0/4)) = cos(0) = 1.000
PE(0, 2) = sin(0 / 10000^(2/4)) = sin(0) = 0.000
PE(0, 3) = cos(0 / 10000^(2/4)) = cos(0) = 1.000
Vector: [0.000, 1.000, 0.000, 1.000]
```

For position 1 ("love"):
```
PE(1, 0) = sin(1 / 10000^(0/4)) = sin(1) = 0.841
PE(1, 1) = cos(1 / 10000^(0/4)) = cos(1) = 0.540
PE(1, 2) = sin(1 / 10000^(2/4)) = sin(0.01) = 0.010
PE(1, 3) = cos(1 / 10000^(2/4)) = cos(0.01) = 1.000
Vector: [0.841, 0.540, 0.010, 1.000]
```

For position 2 ("cats"):
```
PE(2, 0) = sin(2) = 0.909
PE(2, 1) = cos(2) = -0.416
PE(2, 2) = sin(0.02) = 0.020
PE(2, 3) = cos(0.02) = 0.999
Vector: [0.909, -0.416, 0.020, 0.999]
```

**Final Step:**
```
Final Input = Word Embedding + Positional Encoding
```

**Why sinusoidal?**
- Allows extrapolation to longer sequences
- Relative positions: PE(pos+k) can be represented as linear function of PE(pos)
- No learned parameters - generalizes to unseen sequence lengths
- Different frequencies create unique signatures per position

**Alternative (modern):**
```python
# Learned positional embeddings
pos_embedding = nn.Embedding(max_seq_len, d_model)
```

### 3. Feed-Forward Networks (FFN)

Applied to each position independently and identically:

```
FFN(x) = max(0, x @ W₁ + b₁) @ W₂ + b₂

or: FFN(x) = ReLU(x @ W₁ + b₁) @ W₂ + b₂
```

**Dimensions:**
- Input: d_model (e.g., 512)
- Hidden: d_ff = 4 × d_model (e.g., 2048)
- Output: d_model (e.g., 512)

**Why needed?**
- Self-attention is linear (after softmax)
- FFN adds non-linearity and transformation
- "Position-wise" = same network applied to each position

### 4. Layer Normalization & Residual Connections

**Every sub-layer has:**
```
output = LayerNorm(x + Sublayer(x))
```

**Why residual connections?**
- Enable training very deep networks (6+ layers)
- Direct gradient flow
- Each layer learns a refinement, not a complete transformation

**Why LayerNorm (not BatchNorm)?**
- Works better with variable-length sequences
- Normalizes across features (not batch)
- More stable for NLP

## The Encoder

**Single Encoder Layer:**
```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x
```

**Flow through one encoder layer:**
```
Input X [batch, seq_len, d_model]
    ↓
┌─────────────────────────────────────┐
│  Multi-Head Self-Attention          │
│  Q, K, V all from X                 │
└─────────────────────────────────────┘
    ↓
  Add & Norm (residual connection)
    ↓
┌─────────────────────────────────────┐
│  Feed-Forward Network               │
│  FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂    │
└─────────────────────────────────────┘
    ↓
  Add & Norm (residual connection)
    ↓
Output [batch, seq_len, d_model]
```

**Full Encoder:**
Stack 6 encoder layers:
```python
self.layers = nn.ModuleList([
    EncoderLayer(d_model, num_heads, d_ff, dropout)
    for _ in range(6)
])
```

## The Decoder

**Single Decoder Layer:**
```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        # Masked self-attention (look at past only)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (attend to encoder)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # Cross-attention to encoder
        cross_attn_output = self.cross_attn(
            x,                    # Q from decoder
            encoder_output,       # K from encoder
            encoder_output,       # V from encoder
            src_mask
        )
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))

        return x
```

**Flow through one decoder layer:**
```
Target X [batch, tgt_len, d_model]
    ↓
┌─────────────────────────────────────┐
│  Masked Multi-Head Self-Attention   │
│  (prevents looking at future)       │
│  Q, K, V all from X                 │
└─────────────────────────────────────┘
    ↓
  Add & Norm
    ↓
┌─────────────────────────────────────┐
│  Multi-Head Cross-Attention         │
│  Q from decoder                     │ ← Encoder output
│  K, V from encoder output           │
└─────────────────────────────────────┘
    ↓
  Add & Norm
    ↓
┌─────────────────────────────────────┐
│  Feed-Forward Network               │
└─────────────────────────────────────┘
    ↓
  Add & Norm
    ↓
Output [batch, tgt_len, d_model]
```

**Key Difference from Encoder:**
- **Masked self-attention**: Position i can only attend to positions ≤ i
- **Cross-attention**: Additional attention layer to encoder outputs
- **Three** sub-layers instead of two

## Masking

### 1. Padding Mask (Source & Target)
Prevents attention to `<PAD>` tokens:

```python
# Create padding mask
# src: [batch, src_len]
src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
# Result: [batch, 1, 1, src_len]

# Example: src = ["I", "love", "cats", "<PAD>"]
# mask = [1, 1, 1, 0]
```

### 2. Causal Mask (Target only)
Prevents looking at future positions during training:

```python
def create_causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0  # Lower triangular

# Example for seq_len=4:
# [[1, 0, 0, 0],     Position 0 can see: [0]
#  [1, 1, 0, 0],     Position 1 can see: [0, 1]
#  [1, 1, 1, 0],     Position 2 can see: [0, 1, 2]
#  [1, 1, 1, 1]]     Position 3 can see: [0, 1, 2, 3]
```

### 3. Combined Target Mask
```python
tgt_mask = padding_mask & causal_mask
```

## Complete Architecture

```python
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_len=5000,
        dropout=0.1
    ):
        super().__init__()

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def encode(self, src, src_mask=None):
        # Embed + position
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # Embed + position
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source
        encoder_output = self.encode(src, src_mask)

        # Decode target
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        output = self.fc_out(decoder_output)

        return output
```

## Training Process

### 1. Teacher Forcing
During training, use ground truth target tokens as input to decoder:

```python
# src: "I love cats"
# tgt: "<SOS> Ich liebe Katzen <EOS>"

# Input to decoder:  "<SOS> Ich liebe Katzen"
# Expected output:   "Ich liebe Katzen <EOS>"

# Forward pass
output = model(src, tgt_input, src_mask, tgt_mask)

# Compute loss
loss = criterion(output.view(-1, vocab_size), tgt_output.view(-1))
```

### 2. Loss Function
Cross-entropy loss, ignoring padding:

```python
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
```

### 3. Optimizer
Adam with learning rate scheduling:

```python
# Warmup + decay schedule (from paper)
def get_lr(step, d_model, warmup_steps=4000):
    step = max(1, step)
    return d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1,  # Will be multiplied by schedule
    betas=(0.9, 0.98),
    eps=1e-9
)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda step: get_lr(step, d_model, warmup_steps)
)
```

### 4. Label Smoothing (Optional)
Instead of hard targets, use smoothed distribution:

```python
# Instead of [0, 0, 1, 0, 0] for token 2
# Use:      [0.02, 0.02, 0.92, 0.02, 0.02]

class LabelSmoothing(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        # output: [batch * seq_len, vocab_size]
        # target: [batch * seq_len]

        # Create smoothed distribution
        true_dist = torch.zeros_like(output)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.ignore_index] = 0

        # Mask padding positions
        mask = (target != self.ignore_index)

        return self.criterion(output[mask], true_dist[mask])
```

## Inference (Decoding)

### Greedy Decoding
Take highest probability token at each step:

```python
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    encoder_output = model.encode(src, src_mask)

    # Start with <SOS>
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src)

    for i in range(max_len - 1):
        # Create target mask
        tgt_mask = create_causal_mask(ys.size(1)).type_as(src)

        # Decode
        out = model.decode(ys, encoder_output, src_mask, tgt_mask)

        # Get next token
        prob = model.fc_out(out[:, -1])
        next_word = torch.argmax(prob, dim=-1)

        # Append to output
        ys = torch.cat([ys, next_word.unsqueeze(0)], dim=1)

        # Stop if <EOS>
        if next_word == eos_idx:
            break

    return ys
```

### Beam Search (Better)
Keep top k hypotheses at each step:

```python
def beam_search_decode(model, src, src_mask, max_len, start_symbol,
                       eos_symbol, beam_width=5):
    encoder_output = model.encode(src, src_mask)

    # Initialize beam with start symbol
    # Each beam entry: (sequence, score)
    beams = [(torch.ones(1, 1).fill_(start_symbol).type_as(src), 0.0)]
    completed = []

    for i in range(max_len - 1):
        candidates = []

        for seq, score in beams:
            # Skip if already ended
            if seq[0, -1].item() == eos_symbol:
                completed.append((seq, score))
                continue

            # Decode
            tgt_mask = create_causal_mask(seq.size(1)).type_as(src)
            out = model.decode(seq, encoder_output, src_mask, tgt_mask)

            # Get top k tokens
            logits = model.fc_out(out[:, -1])
            log_probs = F.log_softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(log_probs, beam_width)

            # Extend sequences
            for prob, idx in zip(topk_probs[0], topk_indices[0]):
                new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                new_score = score + prob.item()
                candidates.append((new_seq, new_score))

        # Keep top beam_width
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Stop if all beams ended
        if not beams:
            break

    # Return best completed sequence
    all_beams = completed + beams
    best = max(all_beams, key=lambda x: x[1])
    return best[0]
```

## Paper Hyperparameters

### Base Model:
- d_model = 512
- d_ff = 2048
- num_heads = 8
- num_layers = 6 (both encoder and decoder)
- dropout = 0.1
- max_seq_len = 5000

### Big Model:
- d_model = 1024
- d_ff = 4096
- num_heads = 16
- num_layers = 6
- dropout = 0.3

### Training:
- Optimizer: Adam (β₁=0.9, β₂=0.98, ε=10⁻⁹)
- Learning rate: warmup for 4000 steps, then decay
- Batch size: ~25,000 source tokens per batch
- Label smoothing: εₗₛ = 0.1
- Hardware: 8 P100 GPUs
- Training time: 12 hours (base), 3.5 days (big)

## Key Innovations

1. **No Recurrence**: Fully parallelizable across sequence
2. **Self-Attention**: O(1) path between any two positions
3. **Multi-Head**: Learn multiple representation subspaces
4. **Positional Encoding**: Explicit position information
5. **Residual + LayerNorm**: Enable deep networks
6. **Scaled Dot-Product**: Stable gradients for large d_k

## Complexity Analysis

| Operation | Complexity | Sequential Ops | Max Path Length |
|-----------|------------|----------------|-----------------|
| Self-Attention | O(n² · d) | O(1) | O(1) |
| Recurrent | O(n · d²) | O(n) | O(n) |
| Convolutional | O(k · n · d²) | O(1) | O(log_k(n)) |

Where:
- n = sequence length
- d = representation dimension
- k = kernel size

**Self-attention wins on:**
- Parallel processing (O(1) sequential)
- Long-range dependencies (O(1) path)

**Self-attention loses on:**
- Very long sequences (O(n²) memory)

## Variants & Extensions

### 1. Encoder-Only (BERT)
- Bidirectional self-attention
- Masked language modeling
- Next sentence prediction
- Use: Classification, NER, QA

### 2. Decoder-Only (GPT)
- Causal self-attention only
- Autoregressive generation
- Use: Text generation, completion

### 3. Encoder-Decoder (T5, BART)
- Full Transformer
- Use: Translation, summarization

### 4. Efficient Transformers
- **Sparse attention**: Don't attend to all positions
- **Linformer**: Project to lower dimension
- **Reformer**: LSH attention
- **Longformer**: Sliding window + global
- **BigBird**: Sparse, random, global attention

## Common Issues & Solutions

### 1. Exploding/Vanishing Gradients
**Solution**:
- Residual connections
- Layer normalization
- Gradient clipping

### 2. High Memory Usage
**Solution**:
- Gradient checkpointing
- Mixed precision training (FP16)
- Smaller batch sizes
- Gradient accumulation

### 3. Slow Training
**Solution**:
- Learning rate warmup
- Larger batch sizes (if memory allows)
- Multi-GPU training

### 4. Overfitting
**Solution**:
- Dropout (attention, residual, embedding)
- Label smoothing
- More training data
- Data augmentation

## Implementation Tips

1. **Scale embeddings**: Multiply by √d_model before adding positional encoding
2. **Mask shapes**: Ensure broadcasting works correctly for multi-head
3. **Attention scores**: Scale by √d_k to prevent saturation
4. **Initialization**: Xavier/Glorot initialization for linear layers
5. **Warmup**: Essential for stable training
6. **Batch organization**: Pack similar lengths for efficiency

## Comparison Summary

| Feature | RNN/LSTM | Bahdanau/Luong | Transformer |
|---------|----------|----------------|-------------|
| **Recurrence** | Yes | Yes (enc/dec) | No |
| **Attention** | No | Cross-attention | Self + Cross |
| **Parallelizable** | No | Encoder only | Fully |
| **Long-range** | Weak | Better | Best |
| **Training** | Slow | Medium | Fast |
| **Memory** | O(n) | O(n) | O(n²) |
| **Position info** | Implicit | Implicit | Explicit |

## Key Takeaways

1. **Transformer = Self-Attention + Positional Encoding + Residual + LayerNorm**
2. **Encoder**: Stack of (Self-Attention + FFN)
3. **Decoder**: Stack of (Masked Self-Attention + Cross-Attention + FFN)
4. **Training**: Teacher forcing with causal masking
5. **Inference**: Autoregressive with greedy/beam search
6. **Innovation**: Replace recurrence with attention - faster and better!

## Next Steps

1. Implement each component from scratch
2. Build complete Transformer for translation
3. Train on Multi30k dataset
4. Visualize attention patterns
5. Compare with Bahdanau/Luong
6. Explore BERT/GPT architectures
7. Try efficient Transformer variants

## References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Harvard NLP
- [BERT](https://arxiv.org/abs/1810.04805) - Devlin et al., 2018
- [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) - Radford et al., 2018
