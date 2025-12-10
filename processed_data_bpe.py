import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

def bpe_load_parallel(en_path, fr_path):
    with open(en_path, encoding="utf8") as f:
        en_lines = [l.strip() for l in f if l.strip()]
    with open(fr_path, encoding="utf8") as f:
        fr_lines = [l.strip() for l in f if l.strip()]
    assert len(en_lines) == len(fr_lines), f"Mismatch lines: {len(en_lines)} != {len(fr_lines)}"
    return list(zip(en_lines, fr_lines))

def encode_sentence(text, sp):
    pieces = sp.EncodeAsPieces(text.lower())
    ids = [sp.PieceToId(SOS_TOKEN)] + [sp.PieceToId(p) for p in pieces] + [sp.PieceToId(EOS_TOKEN)]
    return torch.tensor(ids)

class ParallelBpeDataset(Dataset):
    def __init__(self, pairs, sp_en, sp_fr):
        self.pairs = pairs
        self.sp_en = sp_en
        self.sp_fr = sp_fr

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, fr = self.pairs[idx]
        src_ids = encode_sentence(en, self.sp_en)
        tgt_ids = encode_sentence(fr, self.sp_fr)
        return src_ids, tgt_ids

def bpe_collate_fn(batch, sp_en, sp_fr):
    src_list, tgt_list = zip(*batch)

    PAD_ID_EN = sp_en.PieceToId(PAD_TOKEN)
    PAD_ID_FR = sp_fr.PieceToId(PAD_TOKEN)

    src_padded = pad_sequence(src_list, batch_first=True, padding_value=PAD_ID_EN)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=PAD_ID_FR)

    src_lens = torch.tensor([len(s) for s in src_list])
    tgt_lens = torch.tensor([len(t) for t in tgt_list])

    return src_padded, tgt_padded, src_lens, tgt_lens

def bpe_create_loader(ds, batch, shuffle, sp_en, sp_fr):
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        collate_fn=lambda x: bpe_collate_fn(x, sp_en, sp_fr)
    )

def bpe_save_dataset(ds, path):
    torch.save(ds, path)


def bpe_load_dataset(path):
    return torch.load(path)