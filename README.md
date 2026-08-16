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


## Result
Encoder-only model: Accuracy: 19370 / 20000 = 0.968500 \
![Alt Text](./plots/training_loss.png) \
Decoder-only model\
![Alt Text](./plots/Decoder_loss.png)
![Alt Text](./plots/Decoder_test.png) \


***Dataset: TinyStory, Number of rows: 2,141,709, 7.62 GB***\
***Total Patameter: 102983, Trainable Parameter: 102983***
![Alt Text](./plots/Decoder_loss_tinystory_5_epoch.png)
![Alt Text](./plots/Decoder_perplexity_tinystory_5_epoch.png)
