import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
    
        # Fetch token ids from the source tokenizer (or target, they should be the same)
        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: any) -> any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # for [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # for [SOS]

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            # Check config.py seq_len if this error occurs
            raise ValueError(f"Sentence is too long for seq_len of {self.seq_len}")

        # Get the scalar <PAD> token ID
        pad_id = self.pad_token.item()

        # Helper to create a padding tensor of length n
        def pads(n):
            if n <= 0:
                return torch.tensor([], dtype=torch.int64)
            return torch.full((n,), pad_id, dtype=torch.int64)

        # Build encoder input: [SOS] + tokens + [EOS] + pads
        encoder_input = torch.cat(
            [
                self.sos_token,  # (1,)
                torch.tensor(enc_input_tokens, dtype=torch.int64),  # (L,)
                self.eos_token,  # (1,)
                pads(enc_num_padding_tokens)  # (pad_count,)
            ]
        )

        # Build decoder input: [SOS] + tokens + pads
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                pads(dec_num_padding_tokens)
            ]
        )

        # Label (target for loss): tokens + [EOS] + pads
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                pads(dec_num_padding_tokens)
            ]
        )

        # Sanity checks
        assert encoder_input.size(0) == self.seq_len, f"encoder_input size {encoder_input.size(0)} != seq_len {self.seq_len}"
        assert decoder_input.size(0) == self.seq_len, f"decoder_input size {decoder_input.size(0)} != seq_len {self.seq_len}"
        assert label.size(0) == self.seq_len, f"label size {label.size(0)} != seq_len {self.seq_len}"

        # Build masks using scalar pad id
        encoder_mask = (encoder_input != pad_id).unsqueeze(0).unsqueeze(0).int()  # (1,1,Seq)
        # Combine padding mask and causal mask for the decoder
        decoder_mask = (decoder_input != pad_id).unsqueeze(0).int() & causal_mask(decoder_input.size(0)) # (1, Seq_Len) & (1, Seq_Len, Seq_Len)

        return {
            "encoder_input": encoder_input,  # (Seq_Len)
            "decoder_input": decoder_input,  # (Seq_Len)
            "encoder_mask": encoder_mask,    # (1, 1, Seq_Len)
            "decoder_mask": decoder_mask,    # (1, Seq_Len, Seq_Len)
            "label": label,                  # (Seq_Len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    

def causal_mask(size):
    # Creates a mask of shape (1, size, size)
    mask = torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask == 0 # Returns True for positions that are allowed (on and below diagonal)