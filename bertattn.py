import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BERTVideo_DividedSpaceTimeAttn(nn.Module):
    def __init__(self, embedding_dimension=256, num_heads=8, split_size=16, batch=5, channels=3, height=128, width=128):
        super(BERTVideo_DividedSpaceTimeAttn, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.split_size = split_size
        self.batch = batch
        self.channels = channels
        self.height = height
        self.width = width

        self.layernorm_temporal = nn.LayerNorm(embedding_dimension)
        self.query_matrices_temporal = nn.Parameter(torch.empty(num_heads, embedding_dimension // num_heads, embedding_dimension // num_heads).normal_(0,0.01), requires_grad=True)
        self.key_matrices_temporal = nn.Parameter(torch.empty(num_heads, embedding_dimension // num_heads, embedding_dimension // num_heads).normal_(0,0.01), requires_grad=True)
        self.value_matrices_temporal = nn.Parameter(torch.empty(num_heads, embedding_dimension // num_heads, embedding_dimension // num_heads).normal_(0,0.01), requires_grad=True)
        self.transform_matrix_temporal = nn.Parameter(torch.empty(embedding_dimension, embedding_dimension).normal_(0,0.01), requires_grad=True)

        self.layernorm_spatial = nn.LayerNorm(embedding_dimension)
        self.query_matrices_spatial = nn.Parameter(torch.empty(num_heads, embedding_dimension // num_heads, embedding_dimension // num_heads).normal_(0,0.01), requires_grad=True)
        self.key_matrices_spatial = nn.Parameter(torch.empty(num_heads, embedding_dimension // num_heads, embedding_dimension // num_heads).normal_(0,0.01), requires_grad=True)
        self.value_matrices_spatial = nn.Parameter(torch.empty(num_heads, embedding_dimension // num_heads, embedding_dimension // num_heads).normal_(0,0.01), requires_grad=True)
        self.transform_matrix_spatial = nn.Parameter(torch.empty(embedding_dimension, embedding_dimension).normal_(0,0.01), requires_grad=True)

        self.layernorm_mlp = nn.LayerNorm(embedding_dimension)
        self.mlp = nn.Linear(embedding_dimension,embedding_dimension)
        # self.initialize_weights()

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.query_matrices_temporal, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.key_matrices_temporal, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.value_matrices_temporal, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.transform_matrix_temporal, nonlinearity="leaky_relu")

        nn.init.kaiming_normal_(self.query_matrices_spatial, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.key_matrices_spatial, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.value_matrices_spatial, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.transform_matrix_spatial, nonlinearity="leaky_relu")

    def forward(self, embeddings):
        # Temporal-Only Attn - dim: num_patches x [num_frames + 1]
        layernorm_embeddings = self.layernorm_temporal(embeddings)
        patches_attnheadsplit = layernorm_embeddings.reshape(-1, self.num_heads, self.embedding_dimension // self.num_heads)

        # QKV Computation
        queries = torch.einsum("abc, dfgh -> dfgh", self.query_matrices_temporal, patches_attnheadsplit.unsqueeze(-2)).squeeze(-2)
        keys = torch.einsum("abc, dfgh -> dfgh", self.key_matrices_temporal, patches_attnheadsplit.unsqueeze(-2)).squeeze(-2)
        values = torch.einsum("abc, dfgh -> dfgh", self.value_matrices_temporal, patches_attnheadsplit.unsqueeze(-2)).squeeze(-2)

        # Attention
        keys_frames = keys[1:].reshape(-1, self.batch, self.num_heads, self.embedding_dimension // self.num_heads)
        keys_tokened = torch.cat([
            keys[0].view(1, 1, self.num_heads, self.embedding_dimension // self.num_heads).repeat((self.height // self.split_size) * (self.width // self.split_size), 1, 1, 1), keys_frames
            ], dim=1)
        queries_frames = queries[1:].reshape(-1, self.batch, self.num_heads, self.embedding_dimension // self.num_heads)
        queries_tokened = torch.cat([
            queries[0].view(1, 1, self.num_heads, self.embedding_dimension // self.num_heads).repeat((self.height // self.split_size) * (self.width // self.split_size), 1, 1, 1), queries_frames
            ], dim=1)

        attn_weights = torch.einsum("abcde, abcef -> abcdf", queries_tokened.unsqueeze(-2), keys_tokened.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        attn_weights = F.softmax(attn_weights.transpose(2,1) / math.sqrt(self.embedding_dimension / self.num_heads), dim=-1)

        # Attn Responses
        values_frames = values[1:].reshape(-1, self.batch, self.num_heads, self.embedding_dimension // self.num_heads)
        attn_response = torch.einsum("acb, abcd -> abcd", attn_weights[:,:,1:], values_frames)
        attn_response = attn_response + torch.einsum("abc, cbd -> acbd", attn_weights[:,:,0].unsqueeze(-1), values[0].unsqueeze(0))
        attn_response = attn_response.reshape(-1, self.embedding_dimension)

        # Token response: value_token + sum(attn_token_other * value_other)
        token_value = values[0].view(-1, self.num_heads, self.embedding_dimension // self.num_heads)
        token_attn_weights = attn_weights[:,:,0].unsqueeze(-1)
        token_response = token_value + torch.einsum("abc, adbe -> cbe", token_attn_weights, values_frames)
        attn_response = torch.cat([token_response.reshape(-1,self.embedding_dimension), attn_response], dim=0)

        # Projections
        projections = attn_response @ self.transform_matrix_temporal + embeddings

        # Spatial-Only Attention - dim: num_patches x [(height // split_size) * (width // split_size) + 1]
        # QKV Computation
        layernorm_embeddings = self.layernorm_spatial(projections)
        patches_attnheadsplit = layernorm_embeddings.reshape(-1, self.num_heads, self.embedding_dimension // self.num_heads)

        queries = torch.einsum("abc, dfgh -> dfgh", self.query_matrices_spatial, patches_attnheadsplit.unsqueeze(-2)).squeeze(-2)
        keys = torch.einsum("abc, dfgh -> dfgh", self.key_matrices_spatial, patches_attnheadsplit.unsqueeze(-2)).squeeze(-2)
        values = torch.einsum("abc, dfgh -> dfgh", self.value_matrices_spatial, patches_attnheadsplit.unsqueeze(-2)).squeeze(-2)

        # Attn
        keys_patches = keys[1:].reshape(self.batch, (self.height // self.split_size) * (self.width // self.split_size), self.num_heads, self.embedding_dimension // self.num_heads)
        keys_tokened = torch.cat([keys[0].unsqueeze(0).repeat(self.batch,1,1).unsqueeze(1), keys_patches], dim=1)
        queries_patches = queries[1:].reshape(self.batch, (self.height // self.split_size) * (self.width // self.split_size), self.num_heads, self.embedding_dimension // self.num_heads)
        queries_tokened = torch.cat([queries[0].unsqueeze(0).repeat(self.batch,1,1).unsqueeze(1), queries_patches], dim=1)

        attn_weights = torch.einsum("abcde, abcef -> abcdf", queries_tokened.unsqueeze(-2), keys_tokened.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        attn_weights = F.softmax(attn_weights.transpose(2,1) / math.sqrt(self.embedding_dimension / self.num_heads), dim=-1)

        # Attn responses
        values_patches = values[1:].reshape(self.batch, (self.height // self.split_size) * (self.width // self.split_size), self.num_heads, self.embedding_dimension // self.num_heads)
        attn_response = torch.einsum("acb, abcd -> abcd", attn_weights[:,:,1:], values_patches)
        attn_response = attn_response + torch.einsum("abc, cbd -> acbd", attn_weights[:,:,0].unsqueeze(-1), values[0].unsqueeze(0))
        attn_response = attn_response.reshape(-1, self.embedding_dimension)

        # Token response: value_token + sum(attn_token_other * value_other)
        token_value = values[0].view(-1, self.num_heads, self.embedding_dimension // self.num_heads)
        token_attn_weights = attn_weights[:,:,0].unsqueeze(-1)
        token_response = token_value + torch.einsum("abc, adbe -> cbe", token_attn_weights, values_patches)
        attn_response = torch.cat([token_response.reshape(-1, self.embedding_dimension), attn_response], dim=0)

        # Spatial Projections
        projections = attn_response @ self.transform_matrix_spatial + projections

        # MLP
        out = self.mlp(self.layernorm_mlp(projections)) + projections
        return out

class BERTVideoModel(nn.Module):
    def __init__(self, depth, embedding_dimension=256, num_heads=8, split_size=16, batch=5, channels=3, height=128, width=128):
        super(BERTVideoModel, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.depth = depth
        self.num_heads = num_heads
        self.split_size = split_size
        self.batch = batch
        self.channels = channels
        self.height = height
        self.width = width
        assert embedding_dimension // num_heads, "embedding_dimension must be divisible by num_heads (example: embedding_dimension=256, num_heads=8"

        self.classification_token = nn.Parameter(torch.empty(1, embedding_dimension), requires_grad=True)
        self.linear_embedding_matrix = nn.Parameter(torch.empty(embedding_dimension, split_size*split_size*channels).normal_(0,0.01), requires_grad=True)
        self.linear_embedding_posencoding = nn.Parameter(torch.empty(batch, height // split_size, width // split_size, embedding_dimension, 1).normal_(0,0.01), requires_grad=True)
        # self.initialize_weights()

        self.layers = nn.ModuleList([nn.Sequential(
            BERTVideo_DividedSpaceTimeAttn(embedding_dimension, num_heads, split_size, batch, channels, height, width),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(embedding_dimension)
        ) for _ in range(depth)])

    def initialize_weights(self):
        nn.init.kaiming_normal_(self.classification_token, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.linear_embedding_matrix, nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.linear_embedding_posencoding, nonlinearity="leaky_relu")

    def forward(self, x):
        batch, channels, height, width = x.size()
        assert batch // self.batch, "unexpected batch size(number of frames) in input tensor"
        assert channels // self.channels, "unexpected number of channels in input tensor"
        assert height // self.height, "unexpected spatial dimensions in input tensor"
        assert width // self.width, "unexpected spatial dimensions in input tensor"
        
        # Initialize new token
        # nn.init.normal_(self.classification_token, mean=0, std=0.01)

        # 16x16 Patches from image
        nonoverlapping_img_patches = x.reshape(self.batch, self.channels, self.split_size, self.height // self.split_size, self.split_size, self.width // self.split_size).permute(0,3,5,2,4,1)
        nonoverlapping_img_patches_flat = nonoverlapping_img_patches.reshape(self.batch, self.height // self.split_size, self.width // self.split_size, -1)

        # Linear Embedding
        embeddings_nontoken = torch.einsum("xy,abcyd->abcxd", self.linear_embedding_matrix, nonoverlapping_img_patches_flat.unsqueeze(-1))
        embeddings_nontoken = (embeddings_nontoken + self.linear_embedding_posencoding).squeeze(-1)
        embeddings_nontoken = embeddings_nontoken.reshape(self.batch * (self.height // self.split_size) * (self.width // self.split_size), self.embedding_dimension)
        embeddings = torch.cat([self.classification_token, embeddings_nontoken], dim=0)

        projections = embeddings
        for layer in self.layers:
            projections = layer(projections) + projections
        return projections[0].view(1, self.embedding_dimension)
        
if __name__ == "__main__":
    device = torch.device('cuda:0')
    embedding_dimension, num_heads, split_size, batch, channels, height, width = 128,8,16,5,3,128,128
    model = BERTVideoModel(4, embedding_dimension, num_heads, split_size, batch, channels, height, width).to(device).double()
    torch.nn.init.normal_(model.classification_token, 0, 0.01)
    out = model(torch.rand(batch, channels, height, width).to(device).double())
    _ = 1