512 Dimensions = 512 Values

  Exactly right. The hidden state is a vector (a row) with 512 numbers:

  hidden state = [0.23, -0.87, 0.45, 0.12, ..., -0.34, 0.56]
                   ↑                                      ↑
                 value 1                              value 512

  Shape: [512] or more precisely [1, 1, 512] = [num_layers, batch, hidden_dim]

  These 512 numbers are the "compressed meaning" of the entire input sentence. The LSTM learned what information to store in each of these positions.

  ---
  How the Decoder Works (Step by Step)

  Think of the decoder as a word-by-word generator. It's like autocomplete on your phone - given what came before, predict the next word.

  The Decoder's Job

  Input:  Context vector (from encoder) + previous word
  Output: Next word

  Let's Trace Through: Translating "i love you" → "je t aime"

  Setup: Encoder has processed "i love you" and produced:
  - hidden: [512 values] - the context vector
  - cell: [512 values] - LSTM memory

  ---
  STEP 1: Generate first French word

  Input to decoder:
    - previous word = <SOS> (start token, index 1)
    - hidden = [context from encoder]
    - cell = [context from encoder]

  Inside decoder:
    1. Embed <SOS>: index 1 → [256 values]

    2. LSTM processes it:
       LSTM(embedding, hidden, cell) → new_hidden, new_cell

    3. Linear layer predicts next word:
       new_hidden [512] → Linear → logits [vocab_size]

       logits = [-2.1, -5.0, -4.2, 0.3, 8.5, -1.2, ...]
                 <PAD> <SOS> <EOS>  j    je   tu
                                         ↑
                                   highest = "je"

  Output: "je", updated hidden, updated cell

  ---
  STEP 2: Generate second French word

  Input to decoder:
    - previous word = "je" (what we just predicted)
    - hidden = [updated from step 1]
    - cell = [updated from step 1]

  Inside decoder:
    1. Embed "je": index 4 → [256 values]

    2. LSTM processes it:
       LSTM(embedding, hidden, cell) → new_hidden, new_cell

    3. Linear layer predicts:
       logits = [-3.1, -6.0, -5.2, 0.1, -2.5, 7.8, ...]
                 <PAD> <SOS> <EOS>  j    je    t
                                               ↑
                                      highest = "t"

  Output: "t", updated hidden, updated cell

  ---
  STEP 3: Generate third French word

  Input to decoder:
    - previous word = "t"
    - hidden = [updated from step 2]
    - cell = [updated from step 2]

  Process → predicts "aime"

  ---
  STEP 4: Generate end token

  Input to decoder:
    - previous word = "aime"
    - hidden = [updated from step 3]
    - cell = [updated from step 3]

  Process → predicts <EOS>

  <EOS> means STOP! Translation complete.

  ---
  Visual Summary

  Encoder output: hidden₀ (contains meaning of "i love you")
                     │
                     ▼
  Step 1:  <SOS> + hidden₀ ──DECODER──► "je"  + hidden₁
                                           │
                                           ▼
  Step 2:  "je" + hidden₁ ──DECODER──► "t"   + hidden₂
                                           │
                                           ▼
  Step 3:  "t"  + hidden₂ ──DECODER──► "aime" + hidden₃
                                           │
                                           ▼
  Step 4:  "aime" + hidden₃ ──DECODER──► <EOS>  ← STOP!

  Final output: "je t aime"

  ---
  Why Does This Work?

  The hidden state carries information forward:

  - hidden₀: "I need to translate something about love and 'you'"
  - hidden₁: "I said 'je', now I need the object"
  - hidden₂: "I said 'je t', now I need the verb"
  - hidden₃: "Sentence is complete, time to stop"

  The LSTM learns to:
  1. Store relevant info from the context in its hidden state
  2. Update that info based on what words it has generated
  3. Use that info to predict sensible next words

  ---
  The Linear Layer (fc_out) Explained

  self.fc_out = nn.Linear(hidden_dim, output_size)  # 512 → vocab_size

  This converts the hidden state into a score for EVERY word in the vocabulary:

  hidden state [512 values]
          │
          ▼ Linear layer (matrix multiplication)
          │
  logits [vocab_size values] = score for each word
          │
          ▼ argmax (pick highest)
          │
  predicted word index

  Example with vocab_size=10:
  logits = [-2.1, -5.0, -4.2, 0.3, 8.5, -1.2, 0.8, -3.3, 2.1, -0.5]
            PAD   SOS   EOS   j    je   tu   il   elle  t    aime

  argmax → index 4 → "je"

  ---
  Key Insight: Decoder is Autoregressive

  Autoregressive means: each prediction depends on previous predictions.

  Can't predict word 3 without knowing word 2
  Can't predict word 2 without knowing word 1
  Must go step by step!

  This is why the decoder processes ONE word at a time, unlike the encoder which can process all words at once.