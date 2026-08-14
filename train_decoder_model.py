# train_decoder_model.py

import argparse
import json
import time
from functools import partial
import torch.nn as nn
from utils import *
from decoder_only_transformer import *
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import re
from torch.nn.utils.rnn import pad_sequence

####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################


NON_ALPHANUMERIC = re.compile(r"[^a-z0-9 ]+")
class DecoderDataset(Dataset):
    def __init__(self, rows, char_to_idx, max_length):
        self.rows = rows
        self.char_to_idx = char_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        text = self.rows[index]["text"][:self.max_length]
        text = NON_ALPHANUMERIC.sub("", str(text).lower())
        tokens = torch.tensor(
            [self.char_to_idx[c] for c in text],
            dtype=torch.long,
        )
        return tokens[:-1], tokens[1:]



def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='lm.py')
    parser.add_argument('--task', type=str, default='BEFORE', help='task to run (BEFORE or BEFOREAFTER)')
    parser.add_argument('--train', type=str, default='data/lettercounting-train.txt', help='path to train examples')
    parser.add_argument('--dev', type=str, default='data/lettercounting-dev.txt', help='path to dev examples')
    parser.add_argument('--output_bundle_path', type=str, default='classifier-output.json', help='path to write the results json to (you should not need to modify)')
    parser.add_argument('--max-length', type=int, default=5462, help='maximum number of characters per example')
    parser.add_argument('--batch-size', type=int, default=64, help='training and evaluation batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader worker processes')
    args = parser.parse_args()
    return args

def collate_decoder_batch(batch, pad_idx):
    inputs, targets = zip(*batch)

    inputs = pad_sequence(
        inputs,
        batch_first=True,
        padding_value=pad_idx,
    )

    targets = pad_sequence(
        targets,
        batch_first=True,
        padding_value=-100,
    )

    attention_mask = inputs.ne(pad_idx)
    return inputs, targets, attention_mask

def read_examples(file):
    """
    :param file:
    :return: A list of the lines in the file, each exactly 20 characters long
    """
    all_lines = []
    for line in open(file):
        all_lines.append(line[:-1]) # eat the \n
    print("%i lines read in" % len(all_lines))
    return all_lines


def get_letter_count_output(input: str, count_only_previous: bool=True) -> np.array:
    """
    :param input: The string
    :param count_only_previous: True if we should only count previous occurrences, False for all occurrences
    :return: the output for the letter-counting task as a numpy array of 0s, 1s, and 2s
    """
    output = np.zeros(len(input))
    for i in range(0, len(input)):
        if count_only_previous:
            output[i] = min(2, len([c for c in input[0:i] if c == input[i]]))
        else:
            output[i] = min(2, len([c for c in input if c == input[i]]) - 1)  # count all *other* instances of input[i]
    return output

if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)

    # Constructs the vocabulary: lowercase letters a to z and space
    number = [chr(ord('0') + i) for i in range(0, 10)]
    vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' '] + number
    vocab_index = Indexer()
    for char in vocab:
        vocab_index.add_and_get_index(char)
    print(repr(vocab_index))

    count_only_previous = True if args.task == "BEFORE" else False

    # construct the dataloader
    data_set = load_dataset("roneneldan/TinyStories")
    train_set = data_set["train"]
    validate_set = data_set["validation"]

    char_to_idx = {
        char: vocab_index.index_of(char)
        for char in vocab
    }
    char_to_idx["<PAD>"] = len(char_to_idx)

    pad_idx = char_to_idx["<PAD>"]
    train_dataset = DecoderDataset(train_set, char_to_idx, args.max_length)
    validate_dataset = DecoderDataset(validate_set, char_to_idx, args.max_length)

    vocab_size = len(char_to_idx)
    num_positions = args.max_length - 1
    collate_fn = partial(collate_decoder_batch, pad_idx=pad_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )

    dev_loader = DataLoader(
        validate_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    print("Start training ...")
    model = train_decoder(args, vocab_size, num_positions, train_loader, dev_loader)
    dev_loss, dev_perplexity = evaluate_language_model(
        model, dev_loader, nn.NLLLoss()
    )
    print("Final dev loss: %f" % dev_loss)
    print("Final dev perplexity: %f" % dev_perplexity)
