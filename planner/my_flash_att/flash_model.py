import numpy as np
import torch
# from performer_pytorch import FastAttention
from torch import nn, autocast
# from flash_attn import flash_attn_qkvpacked_func
from torch.nn import LayerNorm

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.encoding = nn.Parameter(torch.randn(max_len, embed_size))
        self.max_len = max_len
        #self.to(torch.bfloat16)
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        if seq_length > self.max_len:
            raise ValueError("Sequence length exceeds maximum length of positional encoding.")
        positional_encodings = self.encoding[:seq_length, :].unsqueeze(0).repeat(batch_size, 1, 1)
        return x + positional_encodings

class SimpleTransformer(nn.Module):
    def __init__(self, embed_size, heads, num_blocks):
        super(SimpleTransformer, self).__init__()
        self.blocks = nn.ModuleList([SpatialBottleneckAttention(embed_size, heads,5) for _ in range(num_blocks)])
        self.positional_encoding = PositionalEncoding(embed_size, max_len = 128)
    def forward(self, x,attention_mask,attn_mask):
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x,attention_mask,attn_mask)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attention_mask=None, attn_mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply attention mask (for padding)
        if attention_mask is not None:
            # Expand attention_mask to match scores' dimensions
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask.repeat(1, 4, 3, 1)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))


        # Apply causal mask (for preventing future information leakage)
        if attn_mask is not None:
            # Expand attn_mask to match scores' dimensions
            attn_mask = attn_mask.unsqueeze(1)  # [batch_size, 1, seq_length, seq_length]
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out(out)
        return out

class SpatialBottleneckAttention(nn.Module):
    def __init__(self, embed_size, heads, N_prime):
        super(SpatialBottleneckAttention, self).__init__()
        self.N_prime = N_prime
        self.embed_size = embed_size
        self.heads = heads

        self.spatial_ref_points = nn.Parameter(torch.randn(N_prime, embed_size))
        self.mhsa = MultiHeadSelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, Z, attention_mask=None, attn_mask=None):
        # Z: [batch_size, seq_length, embed_size]
        batch_size, seq_length, embed_size = Z.size()

        # Spatial reference points
        IS = self.spatial_ref_points.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, N_prime, embed_size]

        # First MHSA: IS attends to Z
        multihead_attn = nn.MultiheadAttention(100, 4, dropout=0.1,batch_first=True).to(device=Z.device)
        IS_prime,useless1 = multihead_attn(IS, Z, Z)  # [batch_size, N_prime, embed_size]

        # Second MHSA: Z attends to IS_prime
        S,useless = multihead_attn(Z, IS_prime, IS_prime)  # [batch_size, seq_length, embed_size]

        output = self.norm(Z+S)
        return output



class SimpleFlashAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SimpleFlashAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        # self.values = nn.Linear(embed_size, embed_size, bias=False,dtype=torch.bfloat16)
        # self.keys = nn.Linear(embed_size, embed_size, bias=False,dtype=torch.bfloat16)
        # self.queries = nn.Linear(embed_size, embed_size, bias=False,dtype=torch.bfloat16)
        # self.norm = LayerNorm(embed_size,dtype=torch.bfloat16)
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.norm = LayerNorm(embed_size)
        torch.nn.init.xavier_uniform_(self.values.weight)
        torch.nn.init.xavier_uniform_(self.keys.weight)
        torch.nn.init.xavier_uniform_(self.queries.weight)
        # self.to(torch.bfloat16)

    def forward(self, x,attention_mask,attn_mask):
        #x = x.to(torch.bfloat16)
        N, seq_length, _ = x.shape

        try:
            assert not torch.any(torch.isnan(x)), "input x 存在NaN值"
        except AssertionError as a:
            print(a)

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        multihead_attn = nn.MultiheadAttention(100, 4, dropout=0.1,batch_first=True).to(device=x.device)
        output, attn_output_weights = multihead_attn(queries, keys, values,key_padding_mask=attention_mask,attn_mask = attn_mask)
        #multihead_attn = MultiHeadSelfAttention(self.embed_size, self.heads).to(device=x.device)
        #out = multihead_attn(queries, keys, values)

        # embed_per_head = self.embed_size // self.heads
        # q = queries.view(N, seq_length, self.heads, embed_per_head)
        # q = q.permute(0, 2, 1, 3)

        # k = keys.view(N, seq_length, self.heads, embed_per_head)
        # k = k.permute(0, 2, 1, 3)

        # v = values.view(N, seq_length, self.heads, embed_per_head)
        # v = v.permute(0, 2, 1, 3)

        # attn_fn = FastAttention(
        #     dim_heads=embed_per_head,
        #     nb_features=256,
        #     causal=False
        # )

        # # 计算注意力输出
        # out = attn_fn(q, k, v)  # 输出形状为 (1, 8, 512, 64)

        # # 合并头：将 num_heads 和 head_dim 合并到最后一个维度
        # out_reshaped = out.reshape(out.size(0), out.size(2), -1)



        # 使用 Flash Attention 进行注意力计算
        #q = queries.reshape(N, seq_length, self.heads, int(self.embed_size/self.heads))
        #k = keys.reshape(N, seq_length, self.heads, int(self.embed_size / self.heads))
        #v = values.reshape(N, seq_length, self.heads, int(self.embed_size / self.heads))
        #qkv = torch.stack((q, k, v), dim=2).to(torch.bfloat16)

        #try:
        #    assert not torch.any(torch.isnan(qkv)), "qkv 存在NaN值"
        #except AssertionError as a:
        #    print(a)

        #attention_output = flash_attn_qkvpacked_func(qkv)
        #output = attention_output.reshape(N, seq_length, -1)

        output = self.norm(output+x)
        return output


def create_model(embed_size=100, heads=5):
    return SimpleTransformer(embed_size=embed_size, heads=heads,num_blocks=6).cuda()

