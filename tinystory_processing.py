"""Clean and publish a Hugging Face text dataset."""

import argparse
import re
from functools import partial

from datasets import Dataset, DatasetDict, load_dataset


NON_ALPHANUMERIC = re.compile(r"[^a-z0-9 ]+")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize a Hugging Face dataset, then upload it."
    )
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--config", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--output-repo",
        required=True,
        help="Destination repository ID, for example username/tinystories-clean.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Fixed character length. Defaults to the longest cleaned example.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of worker processes used by Dataset.map/filter.",
    )
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def clean_batch(batch, text_column):
    cleaned_texts = []
    lengths = []
    for value in batch[text_column]:
        text = NON_ALPHANUMERIC.sub("", str(value).lower())
        cleaned_texts.append(text)
        lengths.append(len(text))
    return {text_column: cleaned_texts, "original_length": lengths}


def has_text(example):
    return example["original_length"] > 0


def truncate_batch(batch, text_column, max_length):
    return {
        text_column: [text[:max_length] for text in batch[text_column]],
        "sequence_length": [
            min(len(text), max_length) for text in batch[text_column]
        ],
    }


def get_max_length(dataset):
    return max(
        max(split["original_length"], default=0)
        for split in dataset.values()
    )

def process_dataset(args):
    if args.max_length is not None and args.max_length <= 0:
        raise ValueError("--max-length must be greater than zero")

    dataset = load_dataset(
        args.dataset,
        name=args.config,
        revision=args.revision,
    )
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})

    missing_splits = [
        name
        for name, split in dataset.items()
        if args.text_column not in split.column_names
    ]
    if missing_splits:
        raise ValueError(
            "Text column %r is missing from splits: %s"
            % (args.text_column, ", ".join(missing_splits))
        )

    dataset = dataset.map(
        partial(clean_batch, text_column=args.text_column),
        batched=True,
        num_proc=args.num_proc,
        desc="Lowercasing and removing non-alphanumeric characters",
    )
    dataset = dataset.filter(
        has_text,
        num_proc=args.num_proc,
        desc="Removing empty examples",
    )

    max_length = args.max_length or get_max_length(dataset)
    if max_length == 0:
        raise ValueError("The dataset contains no alphanumeric text")

    # if args.max_length is not None:
    #     dataset = dataset.map(
    #         partial(
    #             truncate_batch,
    #             text_column=args.text_column,
    #             max_length=max_length,
    #         ),
    #         batched=True,
    #         num_proc=args.num_proc,
    #         desc="Truncating text to %i characters" % max_length,
    #     )

    # print(dataset)
    # print("Processed sequence length: %i" % max_length)
    # dataset.push_to_hub(
    #     args.output_repo,
    #     private=args.private,
    #     commit_message="Add lowercase alphanumeric dataset",
    #     num_proc=1,
    # )


if __name__ == "__main__":
    process_dataset(parse_args())
