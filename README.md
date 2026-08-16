## Language Model Architecture Implementation

### History of Language Model
- RNN
- Seq-to-Seq (LSTM)
- Transformer (Attention Mechanism)
- BERT(RoBERTa, ALBERT, DeBERTa)
- GPT (GPT-2,3, InstructGPT, ChatGPT, GPT4,5)
- Other (Meta → Llama, DeepSeek → DeepSeek-R1)


### Reference
- Transformer (https://arxiv.org/abs/1706.03762)
- MoE (https://arxiv.org/abs/1701.06538)
- BERT (Pre-training of Deep Bidirectional Transformers for Language Understanding)
- InstructGPT (Training Language Models to Follow Instructions with Human Feedback)

## Introduction
In this project, I practiced the implementation of commonly used language models from scratch. 


## Results

### 🟢 Encoder-only model

> **Test accuracy:** `19,370 / 20,000`<br>
> ![Accuracy: 96.85%](https://img.shields.io/badge/Accuracy-96.85%25-brightgreen?style=for-the-badge)

#### Training loss

![Encoder-only model training loss](./plots/training_loss.png)

---

### 🟣 Decoder-only model

#### Training loss

![Decoder-only model training loss](./plots/Decoder_loss.png)

#### Test results

![Decoder-only model test results](./plots/Decoder_test.png)

#### TinyStories experiment

| Configuration | Value |
|:--|--:|
| **Dataset** | TinyStories |
| **Rows** | 2,141,709 |
| **Dataset size** | 7.62 GB |
| **Training epochs** | 5 |
| **Total parameters** | 102,983 |
| **Trainable parameters** | 102,983 |

##### Training loss

![TinyStories decoder training loss over 5 epochs](./plots/Decoder_loss_tinystory_5_epoch.png)

##### Perplexity

![TinyStories decoder perplexity over 5 epochs](./plots/Decoder_perplexity_tinystory_5_epoch.png)
