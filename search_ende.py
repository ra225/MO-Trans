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
import random
import copy
import os

import onnx
import onnx.utils
import onnx.version_converter

seed = 13
torch.manual_seed(seed)
np.random.seed(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# %%

SRC = "en"
TRG = "de"
PROBLEM_NAME = "Multi30k"

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
# %%

# indexes of special symbols
UNK, BOS, EOS, PAD = 0, 1, 2, 3

# %% md

# Data processing
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

def dominate(x,y,M):
    lte = 0
    lt = 0
    gte = 0
    gt = 0
    for i in range(M):
        if x[i]<=y[i]:
            lte = lte+1;
        if x[i]<y[i]:
            lt = lt+1;
        if x[i]>=y[i]:
            gte = gte+1;
        if x[i]>y[i]:
            gt = gt+1;
    if lte==M and lt>0:
        return 1
    elif gte==M and gt>0:
        return -1
    else:
        return 0
f
def gte(f,lamda,z,M):
    return max(lamda*abs(f-z))

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

class Encoder_GA(nn.Module):
    '''Encoder = token embedding + positional embedding -> a stack of N EncoderBlock -> layer norm'''

    def __init__(self, gene_encoding, config):
        super().__init__()
        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.encoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_embed))
        self.encoder_blocks = nn.ModuleList([EncoderBlock_GA(gene_encoding.elist[i], config) for i in range(gene_encoding.ne)])
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
        return output_arr


class EncoderBlock_GA(nn.Module):
    '''EncoderBlock: self-attention -> position-wise fully connected feed-forward layer'''

    def __init__(self, one_encoder_gene, config):
        super(EncoderBlock_GA, self).__init__()
        self.block_type = one_encoder_gene[0]
        if self.block_type == 1:
            self.layer1 = nn.Sequential(
                nn.Linear(config.d_embed, 512 if one_encoder_gene[1]==0 else 1024),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(512 if one_encoder_gene[1]==0 else 1024, config.d_embed)
            )
            self.layer2 = nn.Sequential(
                nn.Linear(config.d_embed, 512 if one_encoder_gene[2]==0 else 1024),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(512 if one_encoder_gene[2]==0 else 1024, config.d_embed)
            )
        elif self.block_type == 2:
            self.layer1 = nn.Sequential(
                nn.Linear(config.d_embed, 512 if one_encoder_gene[1]==0 else 1024),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(512 if one_encoder_gene[1]==0 else 1024, config.d_embed)
            )
            self.layer2 = MultiHeadedAttention(4 if one_encoder_gene[2]==0 else 8, config.d_embed, config.dropout)
        elif self.block_type == 3:
            self.layer1 = MultiHeadedAttention(4 if one_encoder_gene[1] == 0 else 8, config.d_embed, config.dropout)
            self.layer2 = nn.Sequential(
                nn.Linear(config.d_embed, 512 if one_encoder_gene[2]==0 else 1024),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(512 if one_encoder_gene[2]==0 else 1024, config.d_embed)
            )
        else:
            self.layer1 = MultiHeadedAttention(4 if one_encoder_gene[1] == 0 else 8, config.d_embed, config.dropout)
            self.layer2 = MultiHeadedAttention(4 if one_encoder_gene[2] == 0 else 8, config.d_embed, config.dropout)
        self.residual1 = ResidualConnection(config.d_embed, config.dropout)
        self.residual2 = ResidualConnection(config.d_embed, config.dropout)

    def forward(self, x, mask=None):
        # self-attention
        if self.block_type < 3:
            x = self.residual1(x, self.layer1)
        else:
            x = self.residual1(x, lambda x: self.layer1(x, x, x, mask=mask))
        if self.block_type%2==1:
            return self.residual2(x, self.layer2)
        else:
            return self.residual2(x, lambda x: self.layer2(x, x, x, mask=mask))


class Decoder_GA(nn.Module):
    '''Decoder = token embedding + positional embedding -> a stack of N DecoderBlock -> fully-connected layer'''

    def __init__(self, gene_encoding, config):
        super().__init__()
        self.nd = gene_encoding.nd
        self.dlist = gene_encoding.dlist
        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.decoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_embed))
        self.dropout = nn.Dropout(config.dropout)
        self.decoder_blocks = nn.ModuleList([DecoderBlock_GA(gene_encoding.dlist[i], config) for i in range(gene_encoding.nd)])
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
            x = layer(memory[self.dlist[i][4]], src_mask, x, trg_mask)
            i+=1
        x = self.norm(x)
        logits = self.linear(x)
        return logits


class DecoderBlock_GA(nn.Module):
    ''' EncoderBlock: self-attention -> position-wise feed-forward (fully connected) layer'''

    def __init__(self, one_decoder_gene, config):
        super().__init__()
        self.block_type = one_decoder_gene[0]
        self.atten1 = MultiHeadedAttention(4 if one_decoder_gene[1]==0 else 8, config.d_embed)
        if self.block_type != 3:
            self.atten2 = MultiHeadedAttention(4 if one_decoder_gene[2] == 0 else 8, config.d_embed)
        else:
            self.feed_forward = nn.Sequential(
                nn.Linear(config.d_embed, 512 if one_decoder_gene[2] == 0 else 1024),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(512 if one_decoder_gene[2] == 0 else 1024, config.d_embed)
            )
        if self.block_type == 3:
            self.atten2 = MultiHeadedAttention(4 if one_decoder_gene[3] == 0 else 8, config.d_embed)
        elif self.block_type == 2:
            self.atten3 = MultiHeadedAttention(4 if one_decoder_gene[3] == 0 else 8, config.d_embed)
        else:
            self.feed_forward = nn.Sequential(
                nn.Linear(config.d_embed, 512 if one_decoder_gene[3] == 0 else 1024),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(512 if one_decoder_gene[3] == 0 else 1024, config.d_embed)
            )
        self.residuals = nn.ModuleList([ResidualConnection(config.d_embed, config.dropout)
                                        for i in range(3)])

    def forward(self, memory, src_mask, decoder_layer_input, trg_mask):
        x = memory
        y = decoder_layer_input
        y = self.residuals[0](y, lambda y: self.atten1(y, y, y, mask=trg_mask))
        if self.block_type == 1:
            # keys and values are from the encoder output
            y = self.residuals[1](y, lambda y: self.atten2(y, x, x, mask=src_mask))
            return self.residuals[2](y, self.feed_forward)
        elif self.block_type == 2:
            y = self.residuals[1](y, lambda y: self.atten2(y, x, x, mask=src_mask))
            return self.residuals[2](y, lambda y: self.atten3(y, x, x, mask=src_mask))
        else:
            y = self.residuals[1](y, self.feed_forward)
            return self.residuals[2](y, lambda y: self.atten2(y, x, x, mask=src_mask))



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

def make_from_individual(gene_encoding, config):
    model = Transformer(Encoder_GA(gene_encoding, config), Decoder_GA(gene_encoding, config)).to(DEVICE)

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
       # print(f'ep: {ep}:')
        #print_performance(model)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        else:
            if scheduler.last_epoch > 2 * warmup_steps:
                early_stop_count -= 1
                if early_stop_count <= 0:
                    return train_loss, valid_loss
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
        #print(min([len(x) for x in refs]))
        bleus.append(sacrebleu.corpus_bleu(cans, [refs]).score)
        # print some examples
        #for i in range(3):
        #    print(f'src:  {decode_sentence(detokenizers[SRC], src[i])}')
        #    print(f'trg:  {decode_sentence(detokenizers[TRG], trg_out[i])}')
        #    print(f'pred: {decode_sentence(detokenizers[TRG], translation[i])}')
        return np.mean(bleus)

def get_and_print_fitness(model):
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
    return 100-test_bleu, math.exp(test_loss)

def make_cross_index(ne,nd,now_d):
    if now_d == nd-1 or now_d >= ne-1:
        return ne-1
    select_weight = np.zeros(ne)
    if ne > nd:
        pos = ne-1-(nd-1-now_d)
    else:
        pos = now_d
    square_count = max(pos, ne - 1 - pos)
    select_weight[pos] = 2 ** square_count
    for i in range(1,max(pos,ne-1-pos)+1):
        if pos - i >= 0:
            select_weight[pos-i] = select_weight[pos] / 2**i
        if pos + i <= ne-1:
            select_weight[pos+i] = select_weight[pos] / 2**i
    s = np.sum(select_weight)
    o = random.randint(1,s)
    s = 0
    for i in range(ne):
        s += select_weight[i]
        if s >= o:
            return i


class individual:
    def __init__(self,base_model=0):
        if base_model < 3:
            self.ne =  random.randint(3,7)
            self.nd =  random.randint(3,7)
            self.elist = []
            for i in range(self.ne):
                arr_e = [random.randint(1,4),random.randint(0,1),random.randint(0,1)]
                self.elist.append(arr_e)
            self.dlist = []
            for i in range(self.nd):
                arr_d = [random.randint(1,3),random.randint(0,1),random.randint(0,1),random.randint(0,1)]
                arr_d.append(make_cross_index(self.ne,self.nd,i))
                self.dlist.append(arr_d)
        else:
            self.ne =  base_model
            self.nd =  base_model
            self.elist = []
            for i in range(self.ne):
                arr_e = [3,1,0]
                self.elist.append(arr_e)
            self.dlist = []
            for i in range(self.nd):
                arr_d = [1,1,1,0,i]
                self.dlist.append(arr_d)


def crossover(a, b):
    c = copy.deepcopy(a)
    d = copy.deepcopy(b)
    p = random.randint(1, 100)
    if p <= 8:
        t = random.randint(0, 1)
        if t == 0:
            return c
        return d
    #encoder
    cross_ne = min(a.ne, b.ne)
    for i in range(cross_ne):
        t = c.elist[i][0]
        c.elist[i][0] = d.elist[i][0]
        d.elist[i][0] = t
    #decoder
    cross_nd = min(a.nd, b.nd)
    for i in range(cross_nd):
        t = c.dlist[i][0]
        c.dlist[i][0] = d.dlist[i][0]
        d.dlist[i][0] = t
    e = copy.deepcopy(c)
    t = random.randint(0, 1)
    if t == 0:
        e.ne = d.ne
        e.elist = d.elist
    t = random.randint(0, 1)
    if t == 0:
        e.nd = d.nd
        e.dlist = d.dlist
    return e

def mutation(a):
    b = copy.deepcopy(a)
    now_ne = b.ne
    for i in range(now_ne-3):
        p = random.randint(1, 100)
        if p <= 15:
            wh = random.randint(0, b.ne-1)
            del b.elist[wh]
            b.ne -= 1
    for i in range(7-b.ne):
        p = random.randint(1, 100)
        if p <= 15:
            arr_e = [random.randint(1, 4), random.randint(0, 1), random.randint(0, 1)]
            b.elist.append(arr_e)
            b.ne += 1
    now_nd = b.nd
    for i in range(now_nd-3):
        p = random.randint(1, 100)
        if p <= 15:
            wh = random.randint(0, b.nd-1)
            del b.dlist[wh]
            b.nd -= 1
    for i in range(7-b.nd):
        p = random.randint(1, 100)
        if p <= 15:
            b.nd += 1
            arr_d = [random.randint(1, 3), random.randint(0, 1), random.randint(0, 1), random.randint(0, 1)]
            arr_d.append(make_cross_index(b.ne, b.nd, b.nd-1))
            b.dlist.append(arr_d)
    for i in range(b.ne):
        p = random.randint(1, 100)
        if p <= 15:
            b.elist[i][0] = random.randint(1, 4)
        for j in range(1,3):
            p = random.randint(1, 100)
            if p <= 15:
                b.elist[i][j] = 1 - b.elist[i][j]
    for i in range(b.nd):
        p = random.randint(1, 100)
        if p <= 15:
            b.dlist[i][0] = random.randint(1, 3)
        for j in range(1,4):
            p = random.randint(1, 100)
            if p <= 15:
                b.dlist[i][j] = 1 - b.dlist[i][j]
    for i in range(b.nd):
        if i == b.nd-1 or i >= b.ne-1:
            b.dlist[i][4] = b.ne - 1
            continue
        if b.dlist[i][4] > b.ne - 1 or random.randint(1, 100) <= 35:
            b.dlist[i][4] = make_cross_index(b.ne, b.nd, i)
    return b

# %% md

# Training

# %%
num_generation = 15
N = 15
M = 2
lamda = np.zeros((N,M))
for i in range(N):
    lamda[i][0] = i/N
    lamda[i][1] = (N-i)/N
T = int(N/5)
mu = 20
theta = 5
B = np.zeros((N,N))
EP = [];
for i in range(N):
    for j in range(N):
        B[i][j] = np.linalg.norm(lamda[i,:]-lamda[j,:]);
    B[i,:] = np.argsort(B[i,:])

now_number = 0;
method = 2 #0:Weight Sum Approach  1:PBI 2:chebycheff
x=[];
EP=[];

config = ModelConfig(encoder_vocab_size=vocab_sizes[SRC],
                     decoder_vocab_size=vocab_sizes[TRG],
                     d_embed=512,
                     d_ff=512,
                     h=8,
                     N_encoder=3,
                     N_decoder=3,
                     max_seq_len=max_seq_len,
                     dropout=0.1
                     )
data_loaders = Dataloaders()
train_size = len(data_loaders.train_loader) * batch_size
warmup_steps = 3 * len(data_loaders.train_loader)
# lr first increases in the warmup steps, and then descreases s
lr_fn = lambda step: config.d_embed ** (-0.5) * min([(step + 1) ** (-0.5), (step + 1) * warmup_steps ** (-1.5)])

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)


model_x = []
for i in range(N):
    if i % 3 == 0:
        x.append(individual(3+int(i/3)))
    else:
        x.append(individual())
    model_x.append(make_from_individual(x[i], config))

a = 0
f = np.zeros((2,N))
for i in range(N):
    optimizer = torch.optim.Adam(model_x[i].parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
    early_stop_count = 2
    print("now_generation: 0, individual:",i)
    print("encoding:",x[i].elist, x[i].dlist)
    model_size = sum([p.numel() for p in model_x[i].parameters()])
    print(f'model_size: {model_size}, train_set_size: {train_size}')
    train_loss, valid_loss = train(model_x[i], data_loaders, epochs=10)
    f[0][i],f[1][i] = get_and_print_fitness(model_x[i])


print(f'average_fitness1: {np.average(f[0,:])}, average_fitness2: {np.average(f[1,:])}')

z = np.zeros(M)
for i in range(M):
   z[i] = min(f[i])

EP = []
for i in range(N):
    flag = 1
    for j in range(N):
        if 1 == dominate(f[:,j],f[:,i],M):
            flag = 0
            break
    if flag == 1:
        new_EP = list(f[:,i])
        new_EP.append(0)
        new_EP.append(i)
        EP.append(new_EP)
        torch.save({'model_state_dict':model_x[i].state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'early_stop_count':early_stop_count
                    },"./ET_ende/0_"+str(i)+".pt")

print('EP=',EP)

for generation_loop in range(num_generation):
    for i in range(N):
        k = B[i,random.randint(0,T-1)]
        l = B[i,random.randint(0,T-1)]
        #while k == l:
        #    l = B[i,random.randint(0,T-1)]
        c = crossover(x[int(k)],x[int(l)])
        c = mutation(c)
        #train the new individual
        new_model = make_from_individual(c, config)
        print("now_generation: %d, individual: %d"%(generation_loop+1, i))
        print("encoding:", c.elist, c.dlist)
        model_size = sum([p.numel() for p in new_model.parameters()])
        print(f'model_size: {model_size}, train_set_size: {train_size}')
        optimizer = torch.optim.Adam(new_model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)
        early_stop_count = 2
        train_loss, valid_loss = train(new_model, data_loaders, epochs=10)
        fj = np.zeros(2)
        fj[0], fj[1] = get_and_print_fitness(new_model)
        for j in range(M):
            if z[j] > fj[j]:
                z[j] = fj[j]
        for j in range(T):
            p = int(B[i,j])
            value_fj = gte(fj,lamda[p,:],z,M)
            value_p = gte(f[:,p].T,lamda[p,:],z,M)
            if value_fj < value_p:
                x[p] = c
                for loop in range(M):
                    f[loop,p] = fj[loop]
        flag = 1
        j = 0
        while j < len(EP):
            if j >= len(EP):
                break
            r = dominate(fj,EP[j][0:M],M);
            if 1 == r:
                os.remove("./ET_ende/"+str(EP[j][M])+"_"+str(EP[j][M+1])+".pt")
                del EP[j]
                print('EP=', EP)
                j -= 1
            elif -1 == r:
                flag = 0
            j += 1
        if flag == 1:
            new_EP = list(fj)
            new_EP.append(generation_loop+1)
            new_EP.append(i)
            torch.save({'model_state_dict': model_x[i].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'early_stop_count': early_stop_count
                        }, "./ET_ende/"+str(generation_loop+1)+"_" + str(i) + ".pt")
            EP.append(new_EP)
            print('EP=', EP)
    print(f'average_fitness1: {np.average(f[0, :])}, average_fitness2: {np.average(f[1, :])}')
    print('EP=', EP)
    a = 0



# %% md
