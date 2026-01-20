# Bahdanau Attention: A Simple Explanation

Bahdanau attention was introduced in "Neural Machine Translation by Jointly Learning to Align and Translate" (2015). It solved a critical bottleneck in sequence-to-sequence models.

## The Problem with Basic Seq2Seq

In vanilla seq2seq, the encoder compresses the entire input sentence into a single fixed-size vector (the final hidden state). The decoder then generates the output using only this vector.

```
"The cat sat on the mat" → [single vector] → "Le chat s'est assis sur le tapis"
```

**The issue:** Long sentences lose information. The single vector becomes a bottleneck.

## The Attention Solution

Instead of one vector, let the decoder **look at all encoder hidden states** and decide which ones are relevant at each step.

When translating "cat" to "chat", the model should focus on the encoder state for "cat", not "mat".

## The Math

At each decoder step t:

```
1. Score each encoder state:
   score(s_t, h_i) = v^T * tanh(W_h * h_i + W_s * s_t)

2. Convert scores to weights:
   attention_weights = softmax(scores)

3. Compute context vector:
   context = sum(attention_weights * encoder_outputs)
```

Where:
- `s_t` = current decoder hidden state
- `h_i` = encoder hidden state at position i
- `W_h`, `W_s`, `v` = learnable parameters

## The Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         ENCODER                              │
│  "I love cats" → [h1] → [h2] → [h3] → all states saved      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        ATTENTION                             │
│  decoder_hidden + encoder_outputs → context vector           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         DECODER                              │
│  [<SOS>] + context → "J'" → "aime" → "les" → "chats"        │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Overview

### Encoder

Returns ALL hidden states, not just the last one.

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell  # outputs = ALL states
```

### Attention

Computes which encoder states to focus on.

```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # Score each encoder position
        energy = torch.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_hidden))
        scores = self.v(energy).squeeze(-1)

        # Softmax → weights
        attention_weights = F.softmax(scores, dim=0)

        # Weighted sum → context
        context = (attention_weights.unsqueeze(-1) * encoder_outputs).sum(dim=0)

        return context, attention_weights
```

### Decoder

Uses attention at every step.

```python
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = BahdanauAttention(hidden_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.embedding(input.unsqueeze(0))

        # Get context from attention
        context, attention_weights = self.attention(hidden, encoder_outputs)

        # Combine embedding + context
        lstm_input = torch.cat([embedded, context.unsqueeze(0)], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # Predict next word
        prediction = self.fc_out(torch.cat([output.squeeze(0), context], dim=1))

        return prediction, hidden, cell, attention_weights
```

## Key Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| `encoder_outputs` | [src_len, batch, hidden] | All encoder states |
| `decoder_hidden` | [1, batch, hidden] | Current decoder state |
| `attention_weights` | [src_len, batch] | Focus distribution |
| `context` | [batch, hidden] | Weighted summary |

## Why It Works

1. **No information bottleneck** - Decoder can access any part of the input
2. **Soft alignment** - Learns which source words map to which target words
3. **Interpretable** - Attention weights show what the model focuses on

## Visualization

Attention weights can be plotted as a heatmap:

```
              Source: "I   love  cats"
Target:            ┌─────────────────┐
  "J'"             │ 0.8  0.1   0.1  │  ← focuses on "I"
  "aime"           │ 0.1  0.8   0.1  │  ← focuses on "love"
  "les"            │ 0.1  0.1   0.8  │  ← focuses on "cats"
  "chats"          │ 0.1  0.2   0.7  │  ← focuses on "cats"
                   └─────────────────┘
```

## Tips for Implementation

1. **Debug with shapes** - Print tensor shapes at each step
2. **Build incrementally** - Test encoder, then attention, then decoder
3. **Start small** - Use a subset of data for faster iteration
4. **Visualize attention** - Helps verify the model is learning sensible alignments

## References

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Bahdanau et al., 2015
- This implementation uses PyTorch and the Multi30k dataset for English-German translation
