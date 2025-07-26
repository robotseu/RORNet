import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Linear, LayerNorm, Conv2d


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate=0.0):
        super(Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_size)
        self.dropout = Dropout(dropout_rate)

        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout_rate):
        super(Block, self).__init__()
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, attention_dropout_rate=0.0)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, mlp_dim, dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

class Embeddings(nn.Module):
    def __init__(self, hidden_size=1024, n_frames=8, dropout_rate=0.1):
        super(Embeddings, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_frames + 1, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers, mlp_dim, dropout_rate):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, num_heads, mlp_dim, dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=8, num_layers=4, mlp_dim=512, dropout_rate=0.1, n_frames=8):
        super(TransformerEncoder, self).__init__()
        self.embeddings = Embeddings(hidden_size, n_frames, dropout_rate)
        self.encoder = Encoder(hidden_size, num_heads, num_layers, mlp_dim, dropout_rate)

    def forward(self, x):
        x = self.embeddings(x)
        encoded = self.encoder(x)
        return encoded[:, 0]


if __name__ == '__main__':
    x = torch.rand(2, 8, 1024)

    model = TransformerEncoder(hidden_size=1024, num_heads=8, num_layers=4, mlp_dim=512, dropout_rate=0.1, n_frames=8)

    encoded = model(x)
    print(encoded.shape)

    total_params = sum([param.nelement() for param in model.parameters()])
    print(total_params / 1e6)