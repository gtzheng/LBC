import spacy
from collections import Counter
from spacy.tokenizer import Tokenizer
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd

from img_captioning import VITGPT2_CAPTIONING
import argparse
from config import *


def to_singular(nlp: spacy.language.Language, text: str):
    """Convert words in text to their singular form

    Args:
        nlp (spacy.language.Language): a spacy language model
        text (str): input text

    Returns:
        str: the text with words converted to their singular form
    """
    doc = nlp(text)
    if len(doc) == 1:
        return doc[0].lemma_
    else:
        return doc[:-1].text + doc[-2].whitespace_ + doc[-1].lemma_


def get_adjectives(doc: spacy.tokens.Doc):
    """Get adjectives from a spacy doc

    Args:
        doc (spacy.tokens.Doc): a spacy doc

    Returns:
        list: a list of adjectives
    """
    adj_set = set()
    for chunk in doc.noun_chunks:
        adj = []
        split = False
        noun = ""
        for tok in chunk:
            if tok.pos_ == "ADJ":
                adj.append(f"{tok.text}:adj")

        for a in adj:
            adj_set.add(a)

    return list(adj_set)


def get_nouns(nlp: spacy.language.Language, doc: spacy.tokens.Doc):
    """Get nouns from a spacy doc

    Args:
        nlp (spacy.language.Language): a spacy language model
        doc (spacy.tokens.Doc): a spacy doc

    Returns:
        list: a list of nouns
    """
    nouns = []
    noun_set = set()
    for tok in doc:
        if tok.dep_ == "compound":
            comp_str = doc[tok.i : tok.head.i + 1]
            comp_str = to_singular(nlp, comp_str.text)
            for n in comp_str.split(" "):
                noun_set.add(f"{n}:noun")
            nouns.append(f"{comp_str}:noun")
    for tok in doc:
        if tok.pos_ == "NOUN":
            text = tok.text
            if tok.tag_ in {"NNS", "NNPS"}:
                text = tok.lemma_
            if text not in noun_set:
                nouns.append(f"{text}:noun")
    return nouns


def extract_attributes(nlp: spacy.language.Language, texts: list[str]):
    """Extract attributes (nouns and adjectives) from a list of texts

    Args:
        nlp (spacy.language.Language): a spacy language model
        texts (list[str]): a list of texts

    Returns:
        list: a list of attributes
    """
    docs = nlp.pipe(texts)
    attributes = []
    for doc in docs:
        adjs = get_adjectives(doc)
        nouns = get_nouns(nlp, doc)
        for a in adjs:
            attributes.append(a)
        for n in nouns:
            attributes.append(n)
    return attributes


def get_attribute_embeddings(path: str, threshold: int = 10):
    """Generate embeddings from the extracted concepts stored in a file specified by path

    Args:
        path (str): path to the file that stores the extracted concepts
        threshold (int, optional): the threshold for the number of occurrences of a concept. Defaults to 10.
    """
    caption_model = path.split("/")[-1].split("_")[0]
    count = 0
    with open(path, "rb") as f:
        attribute_arr = pickle.load(f)
    attribute_counts = {}
    for attributes in tqdm(attribute_arr, desc="count attributes"):
        for c in attributes[2]:
            if c in attribute_counts:
                attribute_counts[c] += 1
            else:
                attribute_counts[c] = 1
    attribute_counts = [(k, v) for k, v in attribute_counts.items()]
    attribute_counts = sorted(attribute_counts, key=lambda x: -x[1])
    attributes = np.array([t[0] for t in attribute_counts])
    counts = np.array([t[1] for t in attribute_counts])
    vocab = attributes[counts > threshold]

    vocab_size = len(vocab)
    print(
        f"vocab size is {vocab_size}|({len(attributes)}) ({vocab_size/len(attributes):.2f})"
    )
    save_path = "/".join(path.split("/")[0:-1])
    save_path = os.path.join(
        save_path, f"{caption_model}_vocab_thre{threshold}_{vocab_size}.pickle"
    )
    with open(save_path, "wb") as outfile:
        pickle.dump(vocab, outfile)
    attribute2idx = {v: i for i, v in enumerate(vocab)}

    attribute_embeds = np.zeros((len(attribute_arr), vocab_size))
    for idx, attributes in tqdm(enumerate(attribute_arr), desc="Generate embeddings"):
        for c in attributes[2]:
            if c in attribute2idx:
                attribute_embeds[idx, attribute2idx[c]] = 1
    save_path = "/".join(path.split("/")[0:-1])
    save_path = os.path.join(
        save_path,
        f"{caption_model}_img_embeddings_thre{threshold}_vocab{vocab_size}.pickle",
    )
    with open(save_path, "wb") as outfile:
        pickle.dump(attribute_embeds, outfile)


def get_attributes(caption_path: str, splits: int = 0, split_idx: int = 0):
    """Extract concepts from captions stored in a file specified by caption_path

    Args:
        caption_path (str): path to the file that stores the captions
        splits (int, optional): number of splits for parallel processing. Defaults to 0 (process all data in one go).
        split_idx (int, optional): the index of the split. Defaults to 0.

    Returns:
        str: path to the file that stores the extracted concepts
    """
    save_path = "/".join(caption_path.split("/")[0:-1])
    caption_model = caption_path.split("/")[-1].split("_")[0]
    save_path = os.path.join(
        save_path, f"{caption_model}_extracted_attributes_{split_idx}_{splits}.pickle"
    )
    if os.path.exists(save_path):
        return save_path
    nlp = spacy.load("en_core_web_trf")
    words_list = []
    with open(caption_path, "r") as f:
        for i, line in enumerate(f):
            eles = line.split(",")
            file_name = eles[0].strip()
            label = eles[-1].strip()
            caption = ", ".join(eles[1:-1])
            words_list.append((i, file_name, caption))

    num = len(words_list)
    if splits > 0:
        num_per_split = num // splits
        start_idx = num_per_split * split_idx
        if split_idx == splits - 1:  # the last part
            end_idx = num
        else:
            end_idx = num_per_split * (split_idx + 1)
        print(
            f"[split_idx: {split_idx}] total: {num}, num_splits: {splits} num_per_split: {num_per_split}, range: {start_idx}-{end_idx}"
        )
    else:
        start_idx = 0
        end_idx = num
    sel_words_list = words_list[start_idx:end_idx]
    attributes_arr = []

    for eles in tqdm(sel_words_list):
        attributes = extract_attributes(nlp, [eles[2]])
        attributes = list(set(attributes))
        attributes_arr.append((eles[0], eles[1], attributes))

    with open(save_path, "wb") as outfile:
        pickle.dump(attributes_arr, outfile)
    return save_path


def get_data_folder(dataset: str):
    """Get the path to the image folder and the metadata file for a dataset

    Args:
        dataset (str): dataset name

    Returns:
        str: path to the image folder
        str: path to the metadata file
    """
    if dataset == "waterbirds":
        img_path = WATERBIRDS_DATA_FOLDER
    elif dataset == "celeba":
        img_path = CELEBA_DATA_FOLDER
    elif dataset == "nico":
        img_path = NICO_DATA_FOLDER
    elif dataset == "imagenet-9":
        img_path = IMAGENET9_DATA_FOLDER

    csv_path = os.path.join(img_path, "metadata.csv")
    return img_path, csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="name of a dataset")
    parser.add_argument(
        "--model", type=str, default="vit-gpt2", help="image2text model"
    )
    args = parser.parse_args()

    if args.model == "vit-gpt2":
        caption_model = VITGPT2_CAPTIONING()
    else:
        raise ValueError(f"Captioning model {args.model} not supported")

    data_folder, csv_path = get_data_folder(args.dataset)
    print(f"Process {args.dataset}")
    if args.dataset == "waterbirds":
        path_pos = 1
        label_pos = 2
    elif args.dataset == "celeba":
        path_pos = 2
        label_pos = 3
    elif args.dataset == "nico":
        path_pos = 1
        label_pos = 2
    elif args.dataset == "imagenet-9":
        path_pos = 1
        label_pos = 2
    caption_path = caption_model.get_img_captions(
        data_folder, csv_path, path_pos, label_pos
    )

    attribute_path = get_attributes(caption_path)
    get_attribute_embeddings(attribute_path, threshold=10)
