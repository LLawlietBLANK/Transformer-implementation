import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader , random_split
from torch.utils.tensorboard import SummaryWriter


from config import get_weights_file_path

from dataset import BilingualDataset , causal_mask
from model import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel # Kept for reference
from tokenizers.trainers import WordLevelTrainer # Kept for reference
from tokenizers.pre_tokenizers import Whitespace

# FIX 1: Uncommented the WordPiece imports
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer


from tqdm import tqdm

import warnings 

from pathlib import Path

from config import get_config


def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"][lang]

# This is your WordPiece tokenizer function, it's correct.
def build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not tokenizer_path.exists():
        # Use WordPiece (more robust than WordLevel for translation / OOV)
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordPieceTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )

        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Safety check: ensure required special tokens exist in vocab; if not, raise helpful error
        vocab = tokenizer.get_vocab()
        missing = [t for t in ("[UNK]","[PAD]","[SOS]","[EOS]") if t not in vocab]
        if missing:
            raise ValueError(f"Loaded tokenizer for {lang} is missing special tokens: {missing}. "
                             "Delete the tokenizer file and re-run, or re-train tokenizer.")
    return tokenizer



def get_ds(config):
    ds_raw = load_dataset("opus_books" , f'{config["lang_src"]}-{config["lang_tgt"]}', split = 'train')

    #Build Tokenizer
    tokenizer_src = build_tokenizer(config , ds_raw , config['lang_src'])
    tokenizer_tgt = build_tokenizer(config , ds_raw , config['lang_tgt'])

    #keeping 90% for training and 10% for validation
    train_ds_size = int(0.9* len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(ds_raw , [train_ds_size , val_ds_size])

    train_ds = BilingualDataset(train_ds_raw , tokenizer_src , tokenizer_tgt , config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw , tokenizer_src , tokenizer_tgt , config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        # FIX 2: Used tokenizer_tgt for the target language
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max Length of the source sentence: {max_len_src}')
    print(f'Max Length of the target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds , batch_size= config['batch_size'], shuffle= True)
    val_dataloader = DataLoader(val_ds , batch_size=1 ,shuffle= True) # batch_size=1 for validation

    return train_dataloader , val_dataloader , tokenizer_src , tokenizer_tgt


def get_model(config , vocab_src_len , vocab_tgt_len) :
    model = build_transformer(vocab_src_len , vocab_tgt_len , config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents = True , exist_ok =  True)

    train_dataloader , val_dataloader , tokenizer_src , tokenizer_tgt = get_ds(config)
    
    # get vocab sizes (ints)
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()

    # create model then move it to device
    model = get_model(config, src_vocab_size, tgt_vocab_size)
    model.to(device)


    #tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters() , lr = config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading Model : {model_filename}") 
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict']) # Added model load
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # FIX 3: Loss function must ignore the PAD token from the TARGET tokenizer
    loss_fn = nn.CrossEntropyLoss(ignore_index= tokenizer_tgt.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch , config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader , desc = f"Processing epoch {epoch:02d}")

        for batch in batch_iterator :
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input , encoder_mask)
            decoder_output = model.decode(encoder_output , encoder_mask, decoder_input , decoder_mask)
            proj_output = model.project(decoder_output) # (Batch, Seq_Len, Tgt_Vocab_Size)

            label = batch["label"].to(device) # (Batch, Seq_Len)

            # (Batch * Seq_Len, Tgt_Vocab_Size) vs (Batch * Seq_Len)
            loss = loss_fn(proj_output.view(-1 , tokenizer_tgt.get_vocab_size()),label.view(-1))

            batch_iterator.set_postfix({f"loss" : f"{loss.item():6.3f}"})

            writer.add_scalar('train_loss' , loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        
        # Run validation at the end of each epoch (recommended)
        # run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)


        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config , f'{epoch:02d}')
        torch.save({
            'epoch' : epoch ,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step' : global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings('ignore') 
    config = get_config()
    train_model(config)