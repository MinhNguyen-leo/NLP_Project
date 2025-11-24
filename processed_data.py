import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
import spacy

en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
MAX_VOCAB_SIZE = 10000
MIN_FREQ = 2

def load_parallel(en_path, fr_path):
    with open(en_path, encoding="utf8") as f:
        en_lines = [l.strip() for l in f if l.strip()]
    with open(fr_path, encoding="utf8") as f:
        fr_lines = [l.strip() for l in f if l.strip()]
    assert len(en_lines) == len(fr_lines), f"Mismatch lines: {len(en_lines)} != {len(fr_lines)}"
    return list(zip(en_lines, fr_lines))

def yield_tokens(pairs, language):
    tokenizer = en_tokenizer if language == 'en' else fr_tokenizer
    for en_sentence, fr_sentence in pairs:
        sentence = en_sentence if language == 'en' else fr_sentence
        yield tokenizer(sentence)

def build_vocab(pairs, lang="en", max_tokens=MAX_VOCAB_SIZE, min_freq=MIN_FREQ):
    vocab = build_vocab_from_iterator(yield_tokens(pairs, lang),
                                      max_tokens=max_tokens,
                                      min_freq=min_freq,
                                      specials=SPECIAL_TOKENS,
                                      special_first=True)
    vocab.set_default_index(vocab[UNK_TOKEN])
    return vocab

def tokens_to_ids(tokens, vocab, special_map=None):
    ids = []
    for t in tokens:
        ids.append(vocab[t])
    return ids

def encode_sentence_en(text, vocab):
    toks = en_tokenizer(text)
    ids = tokens_to_ids(toks, vocab)
    return torch.tensor(ids, dtype=torch.long)

def encode_sentence_fr(text, vocab):
    toks = fr_tokenizer(text)
    ids = [vocab[SOS_TOKEN]]
    ids += tokens_to_ids(toks, vocab)
    ids += [vocab[EOS_TOKEN]]
    return torch.tensor(ids, dtype=torch.long)  

def make_collate_fn(vocab_en, vocab_fr):
    PAD_ID_EN = vocab_en[PAD_TOKEN]
    PAD_ID_FR = vocab_fr[PAD_TOKEN]

    def collate_fn(batch):
        src_list, tgt_list = zip(*batch)

        # sort by src length desc
        sorted_pairs = sorted(zip(src_list, tgt_list), 
                              key=lambda x: -x[0].size(0))
        src_list, tgt_list = zip(*sorted_pairs)

        src_padded = pad_sequence(src_list, batch_first=True, padding_value=PAD_ID_EN)
        tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=PAD_ID_FR)

        src_lengths = torch.tensor([s.size(0) for s in src_list], dtype=torch.long)
        tgt_lengths = torch.tensor([t.size(0) for t in tgt_list], dtype=torch.long)

        return src_padded, tgt_padded, src_lengths, tgt_lengths

    return collate_fn
class ParallelDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, fr = self.pairs[idx]
        src_ids = encode_sentence_en(en, self.src_vocab)
        tgt_ids = encode_sentence_fr(fr, self.tgt_vocab)
        return src_ids, tgt_ids
    

def save_vocab(vocab, path):
    if hasattr(vocab, "get_stoi"):      # torchtext
        stoi = vocab.get_stoi()
        itos = vocab.get_itos()
    else:                               # vocab tự tạo
        stoi = vocab.stoi
        itos = vocab.itos

    torch.save({
        "stoi": stoi,
        "itos": itos
    }, path)

def load_vocab(path):
    data = torch.load(path)

    stoi = data["stoi"]
    itos = data["itos"]

    # tạo object VocabLike giống fallback
    class VocabLike:
        def __init__(self, stoi, itos):
            self.stoi = stoi
            self.itos = itos
        def __getitem__(self, token):
            return self.stoi.get(token, self.stoi.get("<unk>"))
        def __len__(self):
            return len(self.itos)

    return VocabLike(stoi, itos)

def save_dataset_pytorch(ds, path):
    """
    ds = list of (src_tensor, tgt_tensor)
    """
    torch.save(ds, path)


def load_dataset_pytorch(path):
    """
    return list of (src_tensor, tgt_tensor)
    """
    return torch.load(path)

def create_loader(ds, batch_size, shuffle, collate_fn):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)