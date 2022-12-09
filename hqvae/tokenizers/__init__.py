# ------------------------------------------------------------------------------------
# Adopted from RQ-Transformer
# https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/txtimg_datasets/tokenizers/__init__.py
# ------------------------------------------------------------------------------------

from functools import partial

from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer, CharBPETokenizer
from .simple_tokenizer import SimpleTokenizer
from .utils import bert_vocab, gpt2_vocab, gpt2_merges
from .utils import huggingface_bpe_16k_vocab, huggingface_bpe_16k_merges
from .utils import huggingface_bpe_30k_vocab, huggingface_bpe_30k_merges


TOKENIZERS = {
    'simple': SimpleTokenizer,
    'bert_huggingface': partial(BertWordPieceTokenizer, vocab=bert_vocab()),
    'gpt2_huggingface': partial(ByteLevelBPETokenizer.from_file,
                                vocab_filename=gpt2_vocab(),
                                merges_filename=gpt2_merges()),
    'bpe16k_huggingface': partial(CharBPETokenizer.from_file,
                                  vocab_filename=huggingface_bpe_16k_vocab(),
                                  merges_filename=huggingface_bpe_16k_merges(),
                                  unk_token="[UNK]"),
    'bpe30k_huggingface': partial(CharBPETokenizer.from_file,
                                  vocab_filename=huggingface_bpe_30k_vocab(),
                                  merges_filename=huggingface_bpe_30k_merges(),
                                  unk_token="[UNK]")
}


def create_tokenizer(tok_name, *args, **kwargs):
    if tok_name == 'simple' or tok_name == 'bert_huggingface':
        filtered_keys = [key for key in kwargs.keys() if key != 'dropout']
        filtered_dict = {key: kwargs[key] for key in filtered_keys}
        return TOKENIZERS[tok_name](*args, **filtered_dict)
    else:
        return TOKENIZERS[tok_name](*args, **kwargs)
