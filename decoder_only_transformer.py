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
        position_encoding = self.positional_encoding(embedded_input)
        embedding = embedded_input + position_encoding 
        attention_list = []
        transform, attention_map = self.transform_layer(embedding)
        attention_list.append(attention_map)
        # transform2, attention_map2 = self.transform_layer.forward(transform)
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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def attention(self, query, key, value):
        """
        Step 1: multiply the query and the key together -> d_model * d_model
        Step 2: divide the result by sqrt(d.internal), smoothing, might not be necessary
        Step 3: apply a softmax to the result (torch.nn.functional.softmax)
        Step 4: multiply the result by the value (torch.mutmal(softmax_result, v) or softmax_result @ v) (torch.bmm?)
        """
        q_kt = torch.matmul(query, key.transpose(-2, -1))
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
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_internal)
        
        causal_mask = torch.triu(
            torch.ones_like(scores, dtype=torch.bool), diagonal=1
        )
        attention_map = torch.nn.functional.softmax(
            scores.masked_fill(causal_mask, float("-inf")), dim=-1
        )
        attention_output = attention_map @ value

        hidden = self.norm1(input_vecs + attention_output)
        output = self.norm2(hidden + self.FFNN(hidden))

        return output, attention_map


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
        seq_len = x.size(1)
        input_size = x.shape[-2]
        # indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        indices_to_embed = torch.arange(
            seq_len,
            device=x.device,
            dtype=torch.long,
        )
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_decider: you can implement this however you want
def train_decoder(args, vocab_size, num_positions, train, dev):
    # raise Exception("Not fully implemented yet")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # For Mac M1/M2/M3 chips
    else:
        device = torch.device("cpu")

    print(f"Device Type: {device.type}")

    if device.type == "mps":
        if torch.backends.mps.is_available():
            print("Status:      MPS is available and active.")
            print("Hardware:    Apple Silicon GPU (Unified Memory)")
            
            # Check current memory allocated by PyTorch on MPS
            allocated_mem = torch.mps.current_allocated_memory() / 1024**2
            print(f"VRAM Used:   {allocated_mem:.2f} MB (allocated by PyTorch)")
        else:
            print("Status:      MPS device targeted, but MPS is NOT available on this system.")
    elif device.type == "cuda":
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            print(f"Name:        {props.name}")
            print(f"Total VRAM:  {props.total_memory / 1024**3:.2f} GB")
        else:
            print("CUDA is not available.")

    # The following code DOES NOT WORK but can be a starting point for your implementation
    # Some suggested snippets to use:
    vocab_size = vocab_size
    num_positions = num_positions
    d_model = 20
    d_internal = 20
    num_classes = vocab_size
    num_layers = 1
    lr = 1e-3

    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers).to(device)
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr)

    num_epochs = 20
    training_losses = []
    dev_losses = []
    dev_perplexities = []
    loss_fcn = nn.CrossEntropyLoss(ignore_index=-100) # ignore <PAD>
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        print("Epoch %i starts " % (t + 1))
        for batch_idx, (input_tokens, target_tokens, _) in enumerate(train):
            if batch_idx % 10000 == 0:
                print(batch_idx)
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)
            prob, _ = model(input_tokens)
            loss = loss_fcn(prob.reshape(-1, prob.size(-1)), target_tokens.reshape(-1))
            model.zero_grad()
            # mode.zero_grad() clears gradients for all model parameters
            # optimizer.zero_grad() clears gradients only for parameters managed by that optimizer.
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        training_losses.append(loss_this_epoch / len(train))
        dev_loss, dev_perplexity = evaluate_language_model(model, dev, loss_fcn)
        dev_losses.append(dev_loss)
        dev_perplexities.append(dev_perplexity)
        print("Epoch %i: train loss = %f, dev loss = %f, dev perplexity = %f" %
              (t + 1, training_losses[-1], dev_loss, dev_perplexity))
        if t % 5 == 0:
            checkpoint = {
                'epoch': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': training_losses[-1]
            }
            torch.save(checkpoint, 'models/decoder/checkpoint_' + t +'.pth')

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


def evaluate_language_model(model, examples, loss_fcn):
    was_training = model.training
    model.eval()
    total_loss = 0.0
    num_tokens = 0

    # nn.Module does not define a `.device` attribute. Infer the device from
    # the model state and verify that parameters and buffers were moved
    # together.
    model_tensors = list(model.parameters()) + list(model.buffers())
    if not model_tensors:
        raise ValueError("Cannot infer the device of a model with no parameters or buffers")
    model_devices = {tensor.device for tensor in model_tensors}
    if len(model_devices) != 1:
        raise RuntimeError(f"Model tensors are on multiple devices: {model_devices}")
    device = next(iter(model_devices))

    # This matters when the loss contains tensors, such as class weights.
    loss_fcn = loss_fcn.to(device)
    print("Start evaluation ...")
    with torch.no_grad():
        for batch_idx, (input_tokens, target_tokens, _) in enumerate(examples):
            print(batch_idx)
            input_tokens = input_tokens.to(device)
            target_tokens = target_tokens.to(device)
            log_probs, _ = model(input_tokens)
            flat_log_probs = log_probs.reshape(-1, log_probs.size(-1))
            flat_targets = target_tokens.reshape(-1)
            batch_tokens = flat_targets.ne(-100).sum().item()
            if batch_tokens == 0:
                continue
            total_loss += loss_fcn(flat_log_probs, flat_targets).item() * batch_tokens
            num_tokens += batch_tokens

    mean_loss = total_loss / num_tokens if num_tokens else 0.0
    perplexity = math.exp(mean_loss) if num_tokens else float("inf")
    model.train(was_training)
    return mean_loss, perplexity
