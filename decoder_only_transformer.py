# decoder_only_transformer.py
import math
import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *

"""
GPT
Causal (Left-to-right context)
Predict the next token
Fluent text generation
"""

# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        """
        embeddings = vocab_size * d_model
        positional_encodings = d_model * num_positions
        
        you can initialize >= 1 TransformerLayer here
        
        another FFNN/Sequential type layer (linear/relu/softmax/linear)
        -> linear -> relu -> linear(this approach should work well) -> log_softmax
        
        !! the last linear layer should have an output size of num_classes
        
        raise Exception("Implement me")
        """
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)
        self.transform_layer = TransformerLayer(d_model, d_internal)
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.V = nn.Linear(d_internal, num_classes)

    def forward(self, input):
        """
        :param indices: list of input indices, changed to input tensor as which used in decode function
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)

        num of positions = 20
        num of classes = 3
        embeddings -> positional_encoding -> TransformerLayer -> (TransformerLayer - optional) -> ... -> FFNN/Sequential
        -> return

        output layer shape after FFN (20, 3)
        """
        # raise Exception("Implement me")
        embedded_input = self.word_embedding(input)
        encoding = self.positional_encoding.forward(embedded_input)
        attention_list = []
        transform, attention_map = self.transform_layer.forward(encoding)
        attention_list.append(attention_map)
        # transform2, attention_map2 = self.transform_layer.forward(encoding)
        # attention_list.append(attention_map2)
        return torch.nn.functional.log_softmax(self.V(transform), dim=-1), attention_list


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        """
        attention has shape n * d_model
        Q and K of dimension dk, and V of dimension dv
        
        To facilitate these residual connections, all sub-layers in the model, as well as the embedding
        layers, produce outputs of dimension dmodel = 512.(larger model and dataset)
        
        in this project, keep the dmodel small would prevent overfitting risk
        """
        self.g = nn.ReLU()
        self.d_model = d_model
        self.d_internal = d_internal
        self.d_k = d_model
        self.W = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_internal)
        self.key = nn.Linear(d_internal, d_internal)
        self.value = nn.Linear(d_internal, d_model)

    def attention(self, query, key, value):
        """
        Step 1: multiply the query and the key together -> d_model * d_model
        Step 2: divide the result by sqrt(d.internal), smoothing, might not be necessary
        Step 3: apply a softmax to the result (torch.nn.functional.softmax)
        Step 4: multiply the result by the value (torch.mutmal(softmax_result, v) or softmax_result @ v) (torch.bmm?)
        """
        q_kt = torch.matmul(query, key.transpose(1, 0))
        causal_mask = torch.triu(
            torch.ones_like(q_kt, dtype=torch.bool), diagonal=1
        )
        q_kt = q_kt.masked_fill(causal_mask, float("-inf"))
        softmax_result = torch.nn.functional.softmax(q_kt / math.sqrt(self.d_k), dim=-1)
        return torch.matmul(softmax_result, value)

    def FFNN(self, attention):
        # Linear -> ReLu -> Linear
        return self.W(self.g(self.V(attention)))

    def forward(self, input_vecs):
        """
        query = self.query(input_vecs)
        key = self.key(input_vecs)
        value = self.value(input_vecs)

        attention = self.attention(query, key, value)
        return self.FFNN(attention)

        map = torch.matmul(query, key.transpose(1, 0)) - might have to divide
        we CAN softmax on the map - look at nn.functional.softmax

        then we have to convert the map to the dimensionality of d_model - attention_output
        residual connections - adding the attention_output to the input_vecs
        we can residual output 2 (maybe the map)
        """
        query = self.query(input_vecs)
        key = self.key(input_vecs)
        value = self.value(input_vecs)

        attention = self.attention(query, key, value)
        scores = torch.matmul(query, key.transpose(1, 0))
        causal_mask = torch.triu(
            torch.ones_like(scores, dtype=torch.bool), diagonal=1
        )
        attention_map = torch.nn.functional.softmax(
            scores.masked_fill(causal_mask, float("-inf")) / math.sqrt(self.d_k), dim=-1
        )
        return self.FFNN(attention), attention_map


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


def evaluate_language_model(model, examples, loss_fcn):
    model.eval()
    total_loss = 0.0
    num_tokens = 0

    with torch.no_grad():
        for ex in examples:
            input_tokens = ex.input_tensor[:-1]
            target_tokens = ex.input_tensor[1:]
            log_probs, _ = model(input_tokens)
            total_loss += loss_fcn(log_probs, target_tokens).item() * target_tokens.numel()
            num_tokens += target_tokens.numel()

    mean_loss = total_loss / num_tokens if num_tokens else 0.0
    perplexity = math.exp(mean_loss) if num_tokens else float("inf")
    return mean_loss, perplexity


# This is a skeleton for train_decider: you can implement this however you want
def train_decoder(args, train, dev):
    # raise Exception("Not fully implemented yet")

    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:
    vocab_size = 27
    num_positions = 20
    d_model = 20
    d_internal = 20
    num_classes = vocab_size
    num_layers = 1
    lr = 1e-3

    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr)

    num_epochs = 50
    training_losses = []
    dev_losses = []
    dev_perplexities = []
    loss_fcn = nn.CrossEntropyLoss()
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        for ex_idx in ex_idxs:
            ex = train[ex_idx]
            input_tokens = ex.input_tensor[:-1]
            target_tokens = ex.input_tensor[1:]
            prob, _ = model(input_tokens)
            loss = loss_fcn(prob, target_tokens)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        training_losses.append(loss_this_epoch / len(train))
        dev_loss, dev_perplexity = evaluate_language_model(model, dev, loss_fcn)
        dev_losses.append(dev_loss)
        dev_perplexities.append(dev_perplexity)
        print("Epoch %i: train loss = %f, dev loss = %f, dev perplexity = %f" %
              (t + 1, training_losses[-1], dev_loss, dev_perplexity))
        model.train()
    plt.plot(range(1, num_epochs + 1), training_losses)
    plt.plot(range(1, num_epochs + 1), dev_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss by Epoch")
    plt.legend(["Training", "Evaluation"])
    plt.savefig("Decoder_loss.png")
    plt.show()

    plt.plot(range(1, num_epochs + 1), dev_perplexities)
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Test Perplexity by Epoch")
    plt.savefig("Decoder_test.png")
    plt.show()
    model.eval()
    return model

def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
