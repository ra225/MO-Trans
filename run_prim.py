import math
from dataclasses import dataclass

import numpy as np
import sacrebleu
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchtext.datasets import Multi30k
from tqdm import tqdm

#rom torchvision import models
#from torchviz import make_dot

seed = 13
torch.manual_seed(seed)
np.random.seed(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# %%

SRC = "de"
TRG = "en"
PROBLEM_NAME = "Multi30k"

# %% md

# Get German and English tokenizers from SentencePiece

# %%

'''
train_iter = Multi30k(split='train', language_pair=(SRC, TRG))
f_de = open("Multi30k_de_text.txt", "w")
f_en = open("Multi30k_en_text.txt", "w")
for pair in train_iter:
    f_de.write(pair[0] + '\n')
    f_en.write(pair[1] + '\n')
f_de.close()
f_en.close()
'''

# %%

en_vocab_size = 8200
de_vocab_size = 10000
vocab_sizes = {"en": en_vocab_size, "de": de_vocab_size}

# %%

# train sentencepiece models to get tokenizers
spm.SentencePieceTrainer.train \
    (
        '--input='+PROBLEM_NAME+'_train.'+TRG+' --model_prefix='+PROBLEM_NAME+'_'+TRG+f' --user_defined_symbols=<pad> --vocab_size={locals()[str(TRG+"_vocab_size")]}')
spm.SentencePieceTrainer.train \
    (
        '--input='+PROBLEM_NAME+'_train.'+SRC+' --model_prefix='+PROBLEM_NAME+'_'+SRC+f' --user_defined_symbols=<pad> --vocab_size={locals()[str(SRC+"_vocab_size")]}')

# make SentencePieceProcessor instances and load the model files
de_sp = spm.SentencePieceProcessor()
de_sp.load(PROBLEM_NAME+'_de.model')
#de_sp.load('Multi30k_de.model')
en_sp = spm.SentencePieceProcessor()
en_sp.load(PROBLEM_NAME+'_en.model')
#en_sp.load('Multi30k_en.model')

tokenizers = {"en": en_sp.encode_as_ids, "de": de_sp.encode_as_ids}
detokenizers = {"en": en_sp.decode_ids, "de": de_sp.decode_ids}

# encode: text => id
print(en_sp.encode_as_pieces('This is a test'))
print(en_sp.encode_as_ids('This is a test'))

# decode: id => text
print(en_sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']))
print(en_sp.decode_ids([302, 258, 10, 4, 2395]))

# %%

print([en_sp.id_to_piece(id) for id in range(20)])
print([de_sp.id_to_piece(id) for id in range(20)])

# %%

# indexes of special symbols
UNK, BOS, EOS, PAD = 0, 1, 2, 3

# %% md

# Data processing

# %%
'''
train_iter = Multi30k(split='train', language_pair=(SRC, TRG))
valid_iter = Multi30k(split='valid', language_pair=(SRC, TRG))
test_iter = Multi30k(split='test', language_pair=(SRC, TRG))
'''
train_iter=[]
with open(PROBLEM_NAME+'_train.'+SRC,'r') as f:
    for line in f:
        train_iter.append([line,""])
num=0
with open(PROBLEM_NAME+'_train.'+TRG,'r') as f:
    for line in f:
        train_iter[num][1] = line
        train_iter[num] = tuple(train_iter[num])
        num+=1


valid_iter=[]
with open(PROBLEM_NAME+'_val.'+SRC,'r') as f:
    for line in f:
        valid_iter.append([line,""])
num=0
with open(PROBLEM_NAME+'_val.'+TRG,'r') as f:
    for line in f:
        valid_iter[num][1] = line
        valid_iter[num] = tuple(valid_iter[num])
        num+=1

test_iter=[]
with open(PROBLEM_NAME+'_test.'+SRC,'r') as f:
    for line in f:
        test_iter.append([line,""])
num=0
with open(PROBLEM_NAME+'_test.'+TRG,'r') as f:
    for line in f:
        test_iter[num][1] = line
        test_iter[num] = tuple(test_iter[num])
        num+=1

train_set = [(x.rstrip('\n'), y.rstrip('\n')) for x, y in train_iter if x != '']
valid_set = [(x.rstrip('\n'), y.rstrip('\n')) for x, y in valid_iter if x != '']
test_set = [(x.rstrip('\n'), y.rstrip('\n')) for x, y in test_iter if x != '']
print(len(train_set), len(valid_set), len(test_set))
for i in range(10):
    print(train_set[i])

# %%

max_seq_len = 50


def tokenize_dataset(dataset):
    'tokenize a dataset and add [BOS] and [EOS] to the beginning and end of the sentences'
    return [(torch.tensor([BOS] + tokenizers[SRC](src_text)[0:max_seq_len - 2] + [EOS]),
             torch.tensor([BOS] + tokenizers[TRG](trg_text)[0:max_seq_len - 2] + [EOS]))
            for src_text, trg_text in dataset]


train_tokenized = tokenize_dataset(train_set)
valid_tokenized = tokenize_dataset(valid_set)
test_tokenized = tokenize_dataset(test_set)


# %%

class TranslationDataset(Dataset):
    'create a dataset for torch.utils.data.DataLoader() '

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pad_sequence(batch):
    'collate function for padding sentences such that all \
    the sentences in the batch have the same length'
    src_seqs = [src for src, trg in batch]
    trg_seqs = [trg for src, trg in batch]
    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs,
                                                 batch_first=True, padding_value=PAD)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_seqs,
                                                 batch_first=True, padding_value=PAD)
    return src_padded, trg_padded


# %%

batch_size = 128


class Dataloaders:
    'Dataloaders contains train_loader, test_loader and valid_loader for training and evaluation '

    def __init__(self):
        self.train_dataset = TranslationDataset(train_tokenized)
        self.valid_dataset = TranslationDataset(valid_tokenized)
        self.test_dataset = TranslationDataset(test_tokenized)

        # each batch returned by dataloader will be padded such that all the texts in
        # that batch have the same length as the longest text in that batch
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,
                                                        shuffle=True, collate_fn=pad_sequence)

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size,
                                                       shuffle=True, collate_fn=pad_sequence)

        self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=batch_size,
                                                        shuffle=True, collate_fn=pad_sequence)


# %% md

# Transformer Model

# %%

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_embed, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert d_embed % h == 0  # check the h number
        self.d_k = d_embed // h
        self.d_embed = d_embed
        self.h = h
        self.WQ = nn.Linear(d_embed, d_embed)
        self.WK = nn.Linear(d_embed, d_embed)
        self.WV = nn.Linear(d_embed, d_embed)
        self.linear = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_key, x_value, mask=None):
        nbatch = x_query.size(0)  # get batch size
        # 1) Linear projections to get the multi-head query, key and value tensors
        # x_query, x_key, x_value dimension: nbatch * seq_len * d_embed
        # LHS query, key, value dimensions: nbatch * h * seq_len * d_k
        query = self.WQ(x_query).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
        key = self.WK(x_key).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
        value = self.WV(x_value).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
        # 2) Attention
        # scores has dimensions: nbatch * h * seq_len * seq_len
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 3) Mask out padding tokens and future tokens
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        # p_atten dimensions: nbatch * h * seq_len * seq_len
        p_atten = torch.nn.functional.softmax(scores, dim=-1)
        p_atten = self.dropout(p_atten)
        # x dimensions: nbatch * h * seq_len * d_k
        x = torch.matmul(p_atten, value)
        # x now has dimensions:nbtach * seq_len * d_embed
        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.d_embed)
        return self.linear(x)  # final linear layer


class ResidualConnection(nn.Module):
    '''residual connection: x + dropout(sublayer(layernorm(x))) '''

    def __init__(self, dim, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, sublayer):
        return x + self.drop(sublayer(self.norm(x)))


# I simply let the model learn the positional embeddings in this notebook, since this
# almost produces identital results as using sin/cosin functions embeddings, as claimed
# in the original transformer paper. Note also that in the original paper, they multiplied
# the token embeddings by a factor of sqrt(d_embed), which I do not do here.

class Encoder(nn.Module):
    '''Encoder = token embedding + positional embedding -> a stack of N EncoderBlock -> layer norm'''

    def __init__(self, config):
        super().__init__()
        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.encoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_embed))
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.N_encoder)])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_embed)

    def forward(self, input, mask=None):
        x = self.tok_embed(input)
        x_pos = self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x + x_pos)
        output_arr=[]
        for layer in self.encoder_blocks:
            x = layer(x, mask)
            output_arr.append(self.norm(x))
        #return output_arr
        return self.norm(x)


class EncoderBlock(nn.Module):
    '''EncoderBlock: self-attention -> position-wise fully connected feed-forward layer'''

    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.atten = MultiHeadedAttention(config.h, config.d_embed, config.dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residual1 = ResidualConnection(config.d_embed, config.dropout)
        self.residual2 = ResidualConnection(config.d_embed, config.dropout)

    def forward(self, x, mask=None):
        # self-attention
        x = self.residual1(x, lambda x: self.atten(x, x, x, mask=mask))
       # x = self.residual2(x, self.feed_forward)
       # x = self.residual2(x, self.feed_forward)
        # position-wise fully connected feed-forward layer
     #   x =  self.residual2(x, lambda x: self.atten2(x, x, x, mask=mask))
        return self.residual2(x, self.feed_forward)


class Decoder(nn.Module):
    '''Decoder = token embedding + positional embedding -> a stack of N DecoderBlock -> fully-connected layer'''

    def __init__(self, config):
        super().__init__()
        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.decoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_embed))
        self.dropout = nn.Dropout(config.dropout)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.N_decoder)])
        self.norm = nn.LayerNorm(config.d_embed)
        self.linear = nn.Linear(config.d_embed, config.decoder_vocab_size)

    def future_mask(self, seq_len):
        '''mask out tokens at future positions'''
        mask = (torch.triu(torch.ones(seq_len, seq_len, requires_grad=False), diagonal=1) != 0).to(DEVICE)
        return mask.view(1, 1, seq_len, seq_len)

    def forward(self, memory, src_mask, trg, trg_pad_mask):
        seq_len = trg.size(1)
        trg_mask = torch.logical_or(trg_pad_mask, self.future_mask(seq_len))
        x = self.tok_embed(trg) + self.pos_embed[:, :trg.size(1), :]
        x = self.dropout(x)
        i = 0
        for layer in self.decoder_blocks:
            x = layer(memory, src_mask, x, trg_mask)
            i+=1
        x = self.norm(x)
        logits = self.linear(x)
        return logits


class DecoderBlock(nn.Module):
    ''' EncoderBlock: self-attention -> position-wise feed-forward (fully connected) layer'''

    def __init__(self, config):
        super().__init__()
        self.nd = 3
        self.atten1 = MultiHeadedAttention(config.h, config.d_embed)
        self.atten2 = MultiHeadedAttention(config.h, config.d_embed)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residuals = nn.ModuleList([ResidualConnection(config.d_embed, config.dropout)
                                        for i in range(3)])

    def forward(self, memory, src_mask, decoder_layer_input, trg_mask):
        x = memory
        y = decoder_layer_input
        y = self.residuals[0](y, lambda y: self.atten1(y, y, y, mask=trg_mask))
        # keys and values are from the encoder output
        y = self.residuals[1](y, lambda y: self.atten2(y, x, x, mask=src_mask))
        return self.residuals[2](y, self.feed_forward)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_mask, trg, trg_pad_mask):
        return self.decoder(self.encoder(src, src_mask), src_mask, trg, trg_pad_mask)


# %%

@dataclass
class ModelConfig:
    encoder_vocab_size: int
    decoder_vocab_size: int
    d_embed: int
    # d_ff is the dimension of the fully-connected  feed-forward layer
    d_ff: int
    # h is the number of attention head
    h: int
    N_encoder: int
    N_decoder: int
    max_seq_len: int
    dropout: float


def make_model(config):
    model = Transformer(Encoder(config), Decoder(config)).to(DEVICE)

    # initialize model parameters
    # it seems that this initialization is very important!
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# %% md

# Training and evaluation helper functions

# %%

def make_batch_input(x, y):
    src = x.to(DEVICE)
    trg_in = y[:, :-1].to(DEVICE)
    trg_out = y[:, 1:].contiguous().view(-1).to(DEVICE)
    src_pad_mask = (src == PAD).view(src.size(0), 1, 1, src.size(-1))
    trg_pad_mask = (trg_in == PAD).view(trg_in.size(0), 1, 1, trg_in.size(-1))
    return src, trg_in, trg_out, src_pad_mask, trg_pad_mask


# %%

from numpy.lib.utils import lookfor


def train_epoch(model, dataloaders):
    model.train()
    grad_norm_clip = 1.0
    losses, acc, count = [], 0, 0
    num_batches = len(dataloaders.train_loader)
    pbar = tqdm(enumerate(dataloaders.train_loader), total=num_batches)
    for idx, (x, y) in pbar:
        optimizer.zero_grad()
        src, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch_input(x, y)
        pred = model(src, src_pad_mask, trg_in, trg_pad_mask).to(DEVICE)
        pred = pred.view(-1, pred.size(-1))
        loss = loss_fn(pred, trg_out).to(DEVICE)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        # report progress
        if idx > 0 and idx % 50 == 0:
            pbar.set_description(f'train loss={loss.item():.3f}, lr={scheduler.get_last_lr()[0]:.5f}')
    return np.mean(losses)


def train(model, dataloaders, epochs):
    global early_stop_count
    global train_loss
    global valid_loss
    best_valid_loss = float('inf')
    train_size = len(dataloaders.train_loader) * batch_size
    for ep in range(epochs):
        train_loss = train_epoch(model, dataloaders)
        valid_loss = validate(model, dataloaders.valid_loader)

       # print(f'ep: {ep}: train_loss={train_loss:.5f}, valid_loss={valid_loss:.5f}')
        print(f'ep: {ep}:')
        print_performance()

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        else:
            if scheduler.last_epoch > 2 * warmup_steps:
                early_stop_count -= 1
               # if early_stop_count <= 0:
                #    return train_loss, valid_loss
    return train_loss, valid_loss


def validate(model, dataloder):
    'compute the validation loss'
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloder):
            src, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch_input(x, y)
            pred = model(src, src_pad_mask, trg_in, trg_pad_mask).to(DEVICE)
            pred = pred.view(-1, pred.size(-1))
            losses.append(loss_fn(pred, trg_out).item())
    return np.mean(losses)


# %%

def translate(model, x):
    'translate source sentences into the target language, without looking at the answer'
    with torch.no_grad():
        dB = x.size(0)
        y = torch.tensor([[BOS] * dB]).view(dB, 1).to(DEVICE)
        x_pad_mask = (x == PAD).view(x.size(0), 1, 1, x.size(-1)).to(DEVICE)
        memory = model.encoder(x, x_pad_mask)
        for i in range(max_seq_len):
            y_pad_mask = (y == PAD).view(y.size(0), 1, 1, y.size(-1)).to(DEVICE)
            logits = model.decoder(memory, x_pad_mask, y, y_pad_mask)
            last_output = logits.argmax(-1)[:, -1]
            last_output = last_output.view(dB, 1)
            y = torch.cat((y, last_output), 1).to(DEVICE)
    return y


# %%

def remove_pad(sent):
    '''truncate the sentence if BOS is in it,
     otherwise simply remove the padding tokens at the end'''
    if sent.count(EOS) > 0:
        sent = sent[0:sent.index(EOS) + 1]
    while sent and sent[-1] == PAD:
        sent = sent[:-1]
    return sent


def decode_sentence(detokenizer, sentence_ids):
    'convert a tokenized sentence (a list of numbers) to a literal string'
    if not isinstance(sentence_ids, list):
        sentence_ids = sentence_ids.tolist()
    sentence_ids = remove_pad(sentence_ids)
    return detokenizer(sentence_ids).replace("<bos>", "") \
        .replace("<eos>", "").strip().replace(" .", ".")


def evaluate(model, dataloader, num_batch=None):
    'evaluate the model, and compute the BLEU score'
    model.eval()
    refs, cans, bleus = [], [], []
    with torch.no_grad():
        for idx, (x, y) in enumerate(dataloader):
            src, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch_input(x, y)
            translation = translate(model, src)
            trg_out = trg_out.view(x.size(0), -1)
            refs = refs + [decode_sentence(detokenizers[TRG], trg_out[i]) for i in range(len(src))]
            cans = cans + [decode_sentence(detokenizers[TRG], translation[i]) for i in range(len(src))]
            if num_batch and idx >= num_batch:
                break
        print(min([len(x) for x in refs]))
        bleus.append(sacrebleu.corpus_bleu(cans, [refs]).score)
        # print some examples
        for i in range(3):
            print(f'src:  {decode_sentence(detokenizers[SRC], src[i])}')
            print(f'trg:  {decode_sentence(detokenizers[TRG], trg_out[i])}')
            print(f'pred: {decode_sentence(detokenizers[TRG], translation[i])}')
        return np.mean(bleus)

def print_performance():
    print("train set examples:")
    train_bleu = evaluate(model, data_loaders.train_loader, 20)
    print("validation set examples:")
    valid_bleu = evaluate(model, data_loaders.valid_loader)
    print("test set examples:")
    test_bleu = evaluate(model, data_loaders.test_loader)
    test_loss = validate(model, data_loaders.test_loader)
    print(f'train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, test_loss: {test_loss:.4f}')
    print(f'train_PPL: {math.exp(train_loss):.4f}, valid_PPL: {math.exp(valid_loss):.4f}, test_PPL: {math.exp(test_loss):.4f}')
    print(f'test_bleu: {test_bleu:.4f}, valid_bleu: {valid_bleu:.4f} train_bleu: {train_bleu:.4f}')


# %% md

# Training

# %%%

config = ModelConfig(encoder_vocab_size=vocab_sizes[SRC],
                     decoder_vocab_size=vocab_sizes[TRG],
                     d_embed=512,
                     d_ff=512,
                     h=8,
                     N_encoder=6,
                     N_decoder=6,
                     max_seq_len=max_seq_len,
                     dropout=0.1
                     )

data_loaders = Dataloaders()
train_size = len(data_loaders.train_loader) * batch_size
model = make_model(config)

#net_plot = make_dot(model)
#net_plot.view()

model_size = sum([p.numel() for p in model.parameters()])
print(f'model_size: {model_size}, train_set_size: {train_size}')
warmup_steps = 3 * len(data_loaders.train_loader)
# lr first increases in the warmup steps, and then descreases
lr_fn = lambda step: config.d_embed ** (-0.5) * min([(step + 1) ** (-0.5), (step + 1) * warmup_steps ** (-1.5)])
optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
early_stop_count = 2

train_loss, valid_loss = 0, 0
train_loss, valid_loss = train(model, data_loaders, epochs=10)
test_loss = validate(model, data_loaders.test_loader)

print("train set examples:")
train_bleu = evaluate(model, data_loaders.train_loader, 20)
print("validation set examples:")
valid_bleu = evaluate(model, data_loaders.valid_loader)
print("test set examples:")
test_bleu = evaluate(model, data_loaders.test_loader)
print(f'train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, test_loss: {test_loss:.4f}')
print(f'test_bleu: {test_bleu:.4f}, valid_bleu: {valid_bleu:.4f} train_bleu: {train_bleu:.4f}')

# %% md



def translate_this_sentence(text: str):
    'translate the source sentence in string formate into target language'
    input = torch.tensor([[BOS] + tokenizers[SRC](text) + [EOS]]).to(DEVICE)
    output = translate(model, input)
    return decode_sentence(detokenizers[TRG], output[0])


translate_this_sentence("Eine Gruppe von Menschen steht vor einem Iglu.")
