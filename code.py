import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module) :
    def __init__(self, d_model, seq_length) :
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        positional_encoding = self._get_positional_encoding(d_model=d_model, seq_length=seq_length)
        self.register_buffer('positional_encoding', positional_encoding)
    
    def _get_positional_encoding(self, d_model, seq_length):
        postional_encoding = torch.zeros(seq_length, d_model)
        for pos in range(seq_length) :
            for i in range(0, d_model, 2):
                postional_encoding[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                postional_encoding[pos, i+1] = math.cos(pos / (10000 ** (i / d_model)))
        return postional_encoding    
    
    def forward(self, x):
        x = x + self.positional_encoding
        return x
    
    
class LayerNormalization(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.eps = 1e-6
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        normalized_x = (x - mean) / (std + self.eps)
        
        return self.alpha * normalized_x + self.bias
    
class FeedForwardBlock(nn.Module) :
    def __init__(self, d_model, droupout) :
        super().__init__()
        self.w1 = nn.Linear(in_features=d_model, out_features=2048)
        self.w2 = nn.Linear(in_features=2048, out_features=d_model)
        self.relu = nn.ReLU()
        self.droupout = nn.Dropout(droupout)
    def forward(self, x):
        x = self.relu(self.w1(x))
        x = self.droupout(x)
        x = self.w2(x)
        return x
    
class MultiHeadAttention(nn.Module) :
    def __init__(self, d_model, h) :
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dv = self.d_model // self.h
        self.dk = self.d_model // self.h
        
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_o = nn.Linear(in_features=self.dk * self.h, out_features=d_model)
        
    def forward(self, query, key, value, mask = None):
        #shape is batch_size, seq_length, d_model --> batch_size, seq_length, d_model
        q_prime = self.w_q(query)
        k_prime = self.w_k(key)
        v_prime = self.w_v(value)
        
        #shape is batch_size, seq_length, heads, d_v
        q_prime = q_prime.reshape(q_prime.shape[0], q_prime.shape[1], self.h, self.dk)
        k_prime = k_prime.reshape(k_prime.shape[0], k_prime.shape[1], self.h, self.dk)
        v_prime = v_prime.reshape(v_prime.shape[0], v_prime.shape[1], self.h, self.dk)
        
        #shape is batch_size, head, seq_length, d_v
        # q_prime = np.transpose(q_prime, (0, 2, 1, 3))
        # k_prime = np.transpose(k_prime, (0, 2, 1, 3))
        # v_prime = np.transpose(v_prime, (0, 2, 1, 3))
        q_prime = q_prime.transpose(1, 2)
        k_prime = k_prime.transpose(1, 2)
        v_prime = v_prime.transpose(1, 2)
        
        #multiplication of Q * K.T / sqrt(d_v)
        #shape is batch_size, heads, seq_length, seq_length
        attention_score = torch.matmul(q_prime, k_prime.transpose(-2, -1)) / math.sqrt(self.dk)
        
        #for deacoder if outplut is masked then used this
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0,float('-inf'))
        
        # use softmax to normalize the values 
        #shape is batch_size, heads, seq_length, seq_length
        attention_weights = torch.softmax(attention_score, dim=-1)
        
        #multiplication of batch_size, heads, seq_length, seq_length @ batch_size, heads, seq_length,d_v
        #resultatnt shape is batch_size, heads, seq_length, d_v
        attention_output = torch.matmul(attention_weights, v_prime)
        # attention_output = attention_output.transpose(0, 2, 1, 3)
        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.contiguous().view(attention_output.size(0), -1, self.h * self.dv)
        
        output = self.w_o(attention_output)
        
        return output
    
class ResidualBlock(nn.Module) :
    def __init__(self, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.layer_norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sub_layer_output) :
        x_res = x + sub_layer_output
        
        x_norm = self.layer_norm(x_res)
        x_dropout = self.dropout(x_norm)
        
        return x_dropout
    
class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, dropout) :
        super().__init__()
        self.multihead_attention = MultiHeadAttention(d_model=d_model, h=h)
        self.feed_forward = FeedForwardBlock(d_model=d_model, droupout=dropout)
        self.residual1 = ResidualBlock(d_model=d_model, dropout=dropout)
        self.residual2 = ResidualBlock(d_model=d_model, dropout=dropout)
        
    def forward(self, x, mask):
        multihead_attention_output = self.multihead_attention(x, x, x, mask)
        residual_block1 = self.residual1(x, multihead_attention_output)
        
        feed_forward_block_output = self.feed_forward(residual_block1)
        residual_block2 = self.residual2(residual_block1, feed_forward_block_output)
        
        return residual_block2
    
class Encoder(nn.Module) :
    def __init__(self, layers) :
        super().__init__()
        self.layers = layers
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class DecoderBlock(nn.Module) :
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, h=h)
        self.cross_attention = MultiHeadAttention(d_model=d_model, h=h)
        self.feed_forward = FeedForwardBlock(d_model=d_model, droupout=dropout)
        self.residual1 = ResidualBlock(d_model=d_model, dropout=dropout)
        self.residual2 = ResidualBlock(d_model=d_model, dropout=dropout)
        self.residual3 = ResidualBlock(d_model=d_model, dropout=dropout)
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        self_attention_output = self.self_attention(x, x, x, tgt_mask)
        residual_connection_output1 = self.residual1(x, self_attention_output)
        cross_attention_output = self.cross_attention(residual_connection_output1, encoder_output, encoder_output, src_mask)
        residual_connection_output2 = self.residual2(residual_connection_output1, cross_attention_output)
        feed_forward_output = self.feed_forward(residual_connection_output2)
        residual_connection_output3 = self.residual3(residual_connection_output2, feed_forward_output)
        return residual_connection_output3
    
class Decoder(nn.Module) :
    def __init__(self, layers) :
        super().__init__()
        self.layers = layers
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x
    
class InputEmbedding(nn.Module) :
    def __init__(self, d_model, vocab_size) :
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x) :
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        return x
    
class LinearLayer(nn.Module) :
    def __init__(self, d_model, vocab_size) :
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linearLayer = nn.Linear(in_features=d_model, out_features=vocab_size)
        
    def forward(self, x) :
        linearLayerOutput = self.linearLayer(x)
        softmax = torch.softmax(linearLayerOutput, dim=-1)
        return softmax
    
class Transformer(nn.Module) :
    def __init__(self, srcEmbedding, tgtEmbedding, srcPositionalEncoding, tgtPositionalEncoding, encoder, decoder, linearLayer) :
        super().__init__()
        self.srcEmbedding = srcEmbedding
        self.tgtEmbedding = tgtEmbedding
        self.srcPositionalEncoding = srcPositionalEncoding
        self.tgtPositionalEncoding = tgtPositionalEncoding
        self.encoder = encoder
        self.decoder = decoder
        self.linearLayer = linearLayer
        
    def encode(self, src, src_mask) :
        src = self.srcEmbedding(src)
        src = self.srcPositionalEncoding(src)
        encoder = self.encoder(src, src_mask)
        return encoder
    
    def decode(self, tgt, encoder, src_mask, tgt_mask) :
        tgt = self.tgtEmbedding(tgt)
        tgt = self.tgtPositionalEncoding(tgt)
        decoder = self.decoder(tgt, encoder, src_mask, tgt_mask)
        return decoder
    
    def linear_layer(self, x) :
        return self.linearLayer(x)
    
def make_transformer(src_seq_length, tgt_seq_length, src_vocal_size, tgt_vocal_size, N = 6, d_model = 512, h = 8, dropout = 0.1) :
    src_embedding = InputEmbedding(d_model= d_model, vocab_size= src_vocal_size)
    tgt_embedding = InputEmbedding(d_model= d_model, vocab_size= tgt_vocal_size)
    src_positional_encoding = PositionalEncoding(d_model= d_model, seq_length= src_seq_length)
    tgt_positional_encoding = PositionalEncoding(d_model= d_model, seq_length= tgt_seq_length)
    
    encoder_blocks = []
    decoder_blocks = []
    for i in range(N) :
        encoder_block = EncoderBlock(d_model= d_model, h= h, dropout= dropout)
        encoder_blocks.append(encoder_block)
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    
    for i in range(N) :
        decoder_block = DecoderBlock(d_model= d_model, h= h, dropout= dropout)
        decoder_blocks.append(decoder_block)
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    linearLayer = LinearLayer(d_model= d_model, vocab_size= tgt_vocal_size)
    
    transformer = Transformer(srcEmbedding= src_embedding, 
                              tgtEmbedding= tgt_embedding, 
                              srcPositionalEncoding= src_positional_encoding, 
                              tgtPositionalEncoding= tgt_positional_encoding, 
                              encoder= encoder, 
                              decoder= decoder, 
                              linearLayer= linearLayer)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer