# ------------------------------------------------------------------------------------
# Adopted from RQ-Transformer
# https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/txtimg_datasets/tokenizers/utils.py
# ------------------------------------------------------------------------------------

import os
from functools import lru_cache


@lru_cache()
def default_bpe():
    # used in the original CLIP implementation
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bert_vocab():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "bert-base-uncased-vocab.txt")


@lru_cache()
def gpt2_vocab():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "vocab.json")


@lru_cache()
def gpt2_merges():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "merges.txt")


@lru_cache()
def huggingface_bpe_16k_vocab():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "bpe-16k-vocab.json")


@lru_cache()
def huggingface_bpe_16k_merges():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "bpe-16k-merges.txt")


@lru_cache()
def huggingface_bpe_30k_vocab():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "bpe-30k-vocab.json")


@lru_cache()
def huggingface_bpe_30k_merges():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pretrained",
                        "bpe-30k-merges.txt")
