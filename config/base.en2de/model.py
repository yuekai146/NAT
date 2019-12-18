import copy
import math
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import itertools

from common import config


NEW_ID = itertools.count()


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Encoder_Decoder(nn.Module):

    def __init__(self, encoder, decoder, src_emb, tgt_emb, generator):
        # A standard encoder decoder model
        super(Encoder_Decoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.generator = generator

        if config.share_decoder_generator_embed:
            self.generator.proj.weight = self.tgt_emb[0].emb.weight

        if config.share_all_embeddings:
            self.tgt_emb[0].emb.weight = self.src_emb[0].emb.weight
            self.generator.proj.weight = self.src_emb[0].emb.weight

        # Share position embedding parameters
        self.tgt_emb[1].emb.weight = self.src_emb[1].emb.weight

    def forward(self, src, src_mask, target, target_mask, tgt_mask, log_prob=True):
        
        bsz, src_len = src.size()
        tgt_len = target.size(1)

        # Encode
        enc_out = self.encoder(self.src_emb(src), src_mask)

        # Soft Copy
        dec_input = soft_copy(enc_out, src_mask.view(bsz, src_len), target_mask.view(bsz, tgt_len))
        
        # HSP
        heuristic_pos = hsp(self.tgt_emb[0](target), target_mask.view(bsz, tgt_len), dec_input)

        # Decode
        logits = self.decoder(dec_input + self.tgt_emb[1].emb(heuristic_pos), enc_out, src_mask, tgt_mask)
        logits = self.generator(logits)
        return logits

    def encode(self, src, src_mask):
        return self.encoder(self.src_emb(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, cache=None):
        if cache is not None and 'cur_len' in cache:
            x = self.tgt_emb[0](tgt)
            x = self.tgt_emb[1](x, cache['cur_len'])
            return self.decoder(x, memory, src_mask, tgt_mask, cache)
        else:
            return self.decoder(self.tgt_emb(tgt), memory, src_mask, tgt_mask, cache)


class Generator(nn.Module):

    def __init__(self, d_model, n_vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, n_vocab)
        self.n_vocab = n_vocab

    def forward(self, x, log_prob=True):
        if log_prob:
            return F.log_softmax(self.proj(x), dim=-1)
        else:
            return self.proj(x)


def soft_copy(enc_out, src_mask, tgt_mask, tau=config.tau):
    # enc_out: torch.FloatTensor (bzs, src_len, emb_dim)
    # src_mask: torch.ByteTensor (bsz, src_len)
    # tgt_mask: torch.ByteTensor (bsz, tgt_len)
    # tau: float number

    # Return: FloatTensor (bsz, tgt_len, emb_dim)
    bsz, src_len, emb_dim = enc_out.size()
    tgt_lengths = tgt_mask.sum(-1).long()
    tgt_len = tgt_mask.size(1)
    src_pos = torch.stack(
            [torch.arange(src_len)] * bsz, dim=0
            ).cuda().type_as(enc_out)

    tgt_pos = torch.stack(
            [torch.arange(tgt_len)] * bsz, dim=0
            ).cuda().type_as(enc_out)
    
    # soft_weigths: bsz, tgt_len, src_len
    soft_weights = -torch.abs(
            tgt_pos.unsqueeze(-1) - src_pos.unsqueeze(-2)
            ) / tau
    mask = src_mask.unsqueeze(-2)
    soft_weights = soft_weights.masked_fill(mask == 0, -float('inf'))
    soft_weights = F.softmax(soft_weights.float(), dim=-1).type_as(soft_weights)
    mask = tgt_mask.unsqueeze(-1)
    soft_weights = soft_weights.masked_fill(mask == 0, 0.0)

    return torch.matmul(soft_weights, enc_out)


def hsp(tgt_emb, tgt_mask, dec_input):
    # tgt_emb: torch.FloatTensor (bsz, tgt_len, emb_dim)
    # tgt_mask: torch.ByteTensor (bsz, tgt_len)
    # dec_input: torch.FloatTensor (bsz, tgt_len, emb_dim)

    # Compute cosine matrix
    bsz, tgt_len = tgt_mask.size()
    tgt_lengths = tgt_mask.sum(-1).long()
    dec_input_norm = dec_input.norm(dim=-1).unsqueeze(-1) # bsz, tgt_len, 1
    tgt_norm = tgt_emb.norm(dim=-1).unsqueeze(-2) # bsz, 1, tgt_len
    cosine_matrix = torch.matmul(dec_input, tgt_emb.permute(0, 2, 1)) / torch.matmul(dec_input_norm, tgt_norm)

    # Apply mask
    mask = tgt_mask.unsqueeze(-1) & tgt_mask.unsqueeze(-2)
    consine_matrix = cosine_matrix.masked_fill(mask == 0, 0.0) # bsz, tgt_len, tgt_len

    heuristic_pos = np.zeros((bsz, tgt_len))
    cosine_matrix = cosine_matrix.detach().cpu().numpy()

    # Get heuristic position for one sentence
    def hsp_for_one_sent(cos_mat, slen, sent_idx):
        # cos_mat: slen * slen
        res = []
        for i in range(slen):
            best_pos_real = np.argmax(cos_mat)
            best_row_real = best_pos_real // slen
            best_col_real = best_pos_real % slen
            res.append((best_row_real, best_col_real))
            cos_mat[best_row_real] = -2.0
            cos_mat[:, best_col_real] = -2.0 # cosine must be larger than -2.0

        res = sorted(res, key=lambda item:item[0])
        assert sum([item[1] for item in res]) == slen * (slen - 1) / 2, [item[1] for item in res]
 
        heuristic_pos[sent_idx, :slen] = np.array([item[1] for item in res])
    
    # Get heuristic position for all sentences
    for j in range(bsz):
        slen = tgt_lengths[j].item()
        hsp_for_one_sent(cosine_matrix[j][:slen, :slen], slen, j)

    return torch.from_numpy(heuristic_pos).long().cuda()


def clone(module, N):
    # Create N identical modules
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = Layer_Norm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x


class Layer_Norm(nn.Module):

    def __init__(self, size, eps=1e-12):
        super(Layer_Norm, self).__init__()
        self.param_std = nn.Parameter(torch.ones(size))
        self.param_mean = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        normalized_x = self.param_std * ( ( x - mean ) / ( std + self.eps ) ) + self.param_mean
        return normalized_x


class Sublayer_Connection(nn.Module):

    def __init__(self, size, dropout):
        super(Sublayer_Connection, self).__init__()
        self.norm = Layer_Norm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # norm -> sublayer -> dropout -> residual add
        return x + self.dropout(sublayer(self.norm(x)))


class Encoder_Layer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(Encoder_Layer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clone(Sublayer_Connection(size, dropout), 2)
        self.size = size

    def forward(self, x, x_mask):
        # Self attention sublayer
        x = self.sublayers[0](x, lambda x:self.self_attn(x, None, None, x_mask))
        x = self.sublayers[1](x, self.feed_forward)
        return x


class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = Layer_Norm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, cache=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, cache)
        return self.norm(x)


class Decoder_Layer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(Decoder_Layer, self).__init__()
        self.src_attn = src_attn
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.sublayers = clone(Sublayer_Connection(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask, cache=None):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, None, None, tgt_mask, cache))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, memory, memory, src_mask, cache))
        x = self.sublayers[2](x, self.feed_forward)
        return x


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    query = query / math.sqrt(d_k)
    attn_score = torch.matmul(query, key.transpose(-2, -1))
    
    if mask is not None:
        attn_score = attn_score.masked_fill(mask == 0, -float('inf'))
    attn_score = F.softmax(attn_score.float(), dim=-1).type_as(attn_score)

    if dropout is not None:
        attn_score = dropout(attn_score)

    return torch.matmul(attn_score, value)


class Multi_Head_Attention(nn.Module):

    def __init__(self, d_model, h, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        assert d_model % h == 0
        self.layer_id = None
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key=None, value=None, mask=None, cache=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        
        def shape(x):
            return x.view(n_batches, -1, self.h, self.d_k).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(n_batches, -1, self.d_model)

        q = shape(self.q_lin(query))  # bsz, qlen, h, d_k
        
        assert (key is None and value is None) or (key is not None and value is not None)

        if cache is not None:
            if key is None:
                if self.layer_id in cache:
                    # Inference time self attention, cache been initialized
                    k, v = cache[self.layer_id]
                    key, value = shape(self.k_lin(query)), shape(self.v_lin(query))
                    key, value = torch.cat([k, key], dim=2), torch.cat([v, value], dim=2)
                else:
                    # Inference time self attention, cache needs to be initialized
                    key, value = shape(self.k_lin(query)), shape(self.v_lin(query))
            else:
                if self.layer_id in cache:
                    # Inference time src attention, cache been initiailized
                    key, value = cache[self.layer_id]
                else:
                    # Inference time src attention, cache needs to be initialized
                    key, value = shape(self.k_lin(key)), shape(self.v_lin(value))
            cache[self.layer_id] = (key, value)
            output = attention(q, key, value, mask, self.dropout)
        else:
            if key is None:
                # Training time self attention
                key, value = shape(self.k_lin(query)), shape(self.v_lin(query))
            else:
                # Training time src attention
                key, value = shape(self.k_lin(key)), shape(self.v_lin(value))
                
            output  = attention(q, key, value, mask, self.dropout)
        output = unshape(output)

        return self.out_lin(output)


class Positionwise_Feed_Forward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1, gelu_activation=False):
        super(Positionwise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu_activation = gelu_activation

    def forward(self, x):
        if self.gelu_activation:
            return self.dropout(self.fc2(gelu(self.fc1(x))))
        else:
            return self.fc2(self.dropout(F.relu(self.fc1(x))))


class Embeddings(nn.Module):

    def __init__(self, d_model, n_vocab):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x)


class Positional_Embeddings(nn.Module):

    def __init__(self, d_model, dropout, max_len=512):
        super(Positional_Embeddings, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.emb = nn.Embedding(max_len, d_model)
        
    def forward(self, x, start=0):
        bsz = x.size(0)
        slen = x.size(1)
        pos_emb = x.new_zeros(bsz, slen).long()
        pos_emb[:] = torch.arange(slen) + start
        x = x + self.emb(pos_emb)
        return self.dropout(x)


class Label_Smoothing_Loss(nn.Module):

    def __init__(self, label_smoothing):
        super(Label_Smoothing_Loss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, target, target_mask=None, reduce=True):
        # logits: bsz, length, n_vocab
        # target: bsz, length
        # mask: bsz, length
        target = target.unsqueeze(-1)
        nll_loss = -logits.gather(dim=-1, index=target)
        smooth_loss = -logits.sum(dim=-1, keepdim=True)
        if target_mask is not None:
            non_pad_mask = target_mask.eq(1)
            nll_loss = nll_loss[non_pad_mask]
            smooth_loss = smooth_loss[non_pad_mask]
            n_tokens = target_mask.sum()
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
            n_tokens = target.numel()

        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.label_smoothing / logits.size(-1)
        loss = ( 1. - self.label_smoothing ) * nll_loss + eps_i * smooth_loss
        
        return loss / n_tokens, nll_loss / n_tokens


def dummpy_input():
    # Used for testing network forward and backward
    lengths = np.random.randint(low=3, high=int(config.tokens_per_batch / config.max_batch_size), size=(config.max_batch_size))
    src = []
    for l in lengths:
        src_sent = np.random.randint(low=config.N_SPECIAL_TOKENS, high=config.src_n_vocab, size=(l)).tolist()
        src_sent += np.zeros(int(config.tokens_per_batch / config.max_batch_size) - l).tolist()
        src.append(src_sent)

    lengths = np.random.randint(low=3, high=int(config.tokens_per_batch / config.max_batch_size), size=(config.max_batch_size))
    tgt = []
    for l in lengths:
        tgt_sent = np.random.randint(low=config.N_SPECIAL_TOKENS, high=config.tgt_n_vocab, size=(l)).tolist()
        tgt_sent += np.zeros(int(config.tokens_per_batch / config.max_batch_size) - l).tolist()
        tgt.append(tgt_sent)

    src, tgt = torch.from_numpy(np.array(src)).long(), torch.from_numpy(np.array(tgt)).long()
    src_mask, tgt_mask = (src != 0), (tgt != 0)
    batch = {"src":src, "tgt":tgt, "src_mask":src_mask.unsqueeze(-2), "tgt_mask":tgt_mask.unsqueeze(-2),
             "target":tgt, "target_mask":tgt_mask
            }
    if config.use_cuda:
        from utils import to_cuda
        batch = to_cuda(batch)

    return batch


def get():
    c = copy.deepcopy
    attn = Multi_Head_Attention(config.d_model, config.num_heads)
    ff = Positionwise_Feed_Forward(config.d_model, config.d_ff, config.dropout, config.gelu_activation)
    position = Positional_Embeddings(config.d_model, config.dropout)
    net = Encoder_Decoder(
            Encoder(Encoder_Layer(config.d_model, c(attn), c(ff), config.dropout), config.encoder_num_layers),
            Decoder(Decoder_Layer(config.d_model, c(attn), c(attn), c(ff), config.dropout), config.decoder_num_layers),
            nn.Sequential(Embeddings(config.d_model, config.src_n_vocab), c(position)),
            nn.Sequential(Embeddings(config.d_model, config.tgt_n_vocab), c(position)),
            Generator(config.d_model, config.tgt_n_vocab)
            )
    for name, p in net.named_parameters():
        if "emb" in name:
            nn.init.normal_(p, mean=0, std=config.d_model ** -0.5)
        else:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    for name, module in net.named_modules():
        if name.endswith("attn"):
            assert module.layer_id is None
            module.layer_id = next(NEW_ID)

    criterion = Label_Smoothing_Loss(config.label_smoothing)

    if config.use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()
    return net, criterion


if __name__ == "__main__":
    net, criterion = get()
    batch = dummpy_input()
    
    import optimizer
    opt = optimizer.get(net)
    for i in range(1):
        logits = net(batch['src'], batch['src_mask'], batch['target'], batch['target_mask'], batch['tgt_mask'])
        loss, nll_loss = criterion(logits, batch['target'], batch['target_mask'].squeeze())
        loss.backward()
        opt.step()
        print(nll_loss.item())
