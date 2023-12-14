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

from img_captioning import VITGPT2_CAPTIONING, BLIP_CAPTIONING
import argparse
      
def to_singular(nlp, text):
    doc = nlp(text)
    if len(doc) == 1:
        return doc[0].lemma_
    else:
        return doc[:-1].text + doc[-2].whitespace_ + doc[-1].lemma_


def get_adj_pairs(doc):
    adj_set = set()
    for chunk in doc.noun_chunks:
        adj = []
        split = False
        noun = ""
        for tok in chunk:
            if tok.pos_ == "ADJ":
                adj.append(f"{tok.text}:adj")
            # if tok.pos_ == "CCONJ":
            #     split = True
        for a in adj:
            adj_set.add(a)
        # if len(adj) > 0:
        #     if split:
        #         for a in adj:
        #             adj_set.add(a)  # e.g., black and white to [black, white]
        #     else:
        #         adj = " ".join(adj)  # combine adjectives, such as old fashioned
        #         adj_set.add(adj)
    return list(adj_set)


def get_nouns(nlp, doc):
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


def extract_concepts(nlp, texts):
    docs = nlp.pipe(texts)
    concepts = []
    for doc in docs:
        adjs = get_adj_pairs(doc)
        nouns = get_nouns(nlp, doc)
        for a in adjs:
            concepts.append(a)
        for n in nouns:
            concepts.append(n)
    return concepts


def get_concept_embeddings(path, threshold=10):
    caption_model = path.split("/")[-1].split('_')[0]
    count = 0
    with open(path, "rb") as f:
        concept_arr = pickle.load(f)
    concept_counts = {}
    for concepts in tqdm(concept_arr, desc="count concepts"):
        for c in concepts[2]:
            if c in concept_counts:
                concept_counts[c] += 1
            else:
                concept_counts[c] = 1
    concept_counts = [(k,v) for k,v in concept_counts.items()]
    concept_counts = sorted(concept_counts, key=lambda x: -x[1])
    concepts = np.array([t[0] for t in concept_counts])
    counts = np.array([t[1] for t in concept_counts])
    vocab = concepts[counts>threshold]
    
    vocab_size = len(vocab)
    print(f"vocab size is {vocab_size}|({len(concepts)}) ({vocab_size/len(concepts):.2f})")
    save_path = "/".join(path.split("/")[0:-1])
    save_path = os.path.join(save_path, f"{caption_model}_vocab_thre{threshold}_{vocab_size}.pickle")
    with open(save_path, "wb") as outfile:
        pickle.dump(vocab, outfile)
    concept2idx = {v:i for i,v in enumerate(vocab)}
    
    concept_embeds = np.zeros((len(concept_arr), vocab_size))
    for idx,concepts in tqdm(enumerate(concept_arr), desc="Generate embeddings"):
        for c in concepts[2]:
            if c in concept2idx:
                concept_embeds[idx, concept2idx[c]] = 1
    save_path = "/".join(path.split("/")[0:-1])
    save_path = os.path.join(save_path, f"{caption_model}_img_embeddings_thre{threshold}_vocab{vocab_size}.pickle")
    with open(save_path, "wb") as outfile:
        pickle.dump(concept_embeds, outfile)
 
             
def get_concepts(caption_path, splits=0, split_idx=0):
    save_path = "/".join(caption_path.split("/")[0:-1])
    caption_model = caption_path.split("/")[-1].split('_')[0]
    save_path = os.path.join(save_path, f"{caption_model}_extracted_concepts_{split_idx}_{splits}.pickle")
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
        if split_idx == splits - 1: # the last part
            end_idx = num
        else:
            end_idx = num_per_split * (split_idx+1)
        print(f"[split_idx: {split_idx}] total: {num}, num_splits: {splits} num_per_split: {num_per_split}, range: {start_idx}-{end_idx}")
    else:
        start_idx = 0
        end_idx = num
    sel_words_list = words_list[start_idx:end_idx]
    concepts_arr = []
    
    for eles in tqdm(sel_words_list):
        concepts = extract_concepts(nlp, [eles[2]])
        concepts = list(set(concepts))
        concepts_arr.append((eles[0],eles[1],concepts))

    
    with open(save_path, "wb") as outfile:
        pickle.dump(concepts_arr, outfile)
    return save_path

def get_data_folder(dataset):
    if dataset == "waterbirds":
        csv_path = "/bigtemp/gz5hp/dataset_hub/waterbird_complete95_forest2water2/metadata.csv"
        img_path = "/bigtemp/gz5hp/dataset_hub/waterbird_complete95_forest2water2"
    elif dataset ==  "celeba":
        csv_path = "/bigtemp/gz5hp/dataset_hub/celebfaces/img_align_celeba/metadata.csv"
        img_path = "/bigtemp/gz5hp/dataset_hub/celebfaces/img_align_celeba"
    elif dataset == "metashift":
        csv_path = "/bigtemp/gz5hp/dataset_hub/metashifts/MetaDatasetCatDog/metadata.csv"
        img_path = "/bigtemp/gz5hp/dataset_hub/metashifts/MetaDatasetCatDog"
    return img_path, csv_path
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="name of a dataset")
    parser.add_argument("--model", type=str, default="vit-gpt2", help="image2text model")
    args = parser.parse_args()
    
    if args.model == "vit-gpt2":
        caption_model = VITGPT2_CAPTIONING()
    elif args.model == "blip":
        caption_model = BLIP_CAPTIONING()
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
    caption_path = caption_model.get_img_captions(data_folder, csv_path, path_pos, label_pos)
   
    concept_path = get_concepts(caption_path)
    get_concept_embeddings(concept_path)
    
    