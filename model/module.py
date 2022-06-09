import copy
import torch
import torch.nn as nn
from transformers import AutoModel
from .layer import EncoderLayer, DecoderLayer, get_clones
from .embedding import TransformerEmbedding
from utils.train import create_src_mask, create_trg_mask




class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.n_layers = config.n_layers
        self.layer = EncoderLayer(config)


    def forward(self, src, src_mask):
        for _ in range(self.n_layers):
            src = self.layer(src, src_mask)

        return src




class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.n_layers = config.n_layers
        self.layers = DecoderLayer(config)


    def forward(self, memory, trg, src_mask, trg_mask):
        for _ in range(self.n_layers):
            trg, attn = self.layer(memory, trg, src_mask, trg_mask)
        
        return trg, attn




class NLG_BERT(nn.Module):
    def __init__(self, config):
        super(NLG_BERT, self).__init__()

        self.bert = AutoModel.from_pretrained('bert-base-cased')
        self.bert.resize_token_embeddings(config.input_dim)

        self.embedding = self.bert.embeddings

        self.enocder = Encoder(config)
        self.deocder = Decoder(config)

        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)
        self.device = config.device



    def forward(self, src, trg):
        enc_out = self.encoder(src)
        dec_out = self.decoder(src, enc_out, trg, trg_mask)
        return out