import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

# Src: https://github.com/leaderj1001/Stand-Alone-Self-Attention/blob/master/attention.py
class SelfAttnLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1):
        super(SelfAttnLayer, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.empty(out_channels // 2, 1, 1, kernel_size, 1).normal_(mean=0, std=0.01), requires_grad=True)
        self.rel_w = nn.Parameter(torch.empty(out_channels // 2, 1, 1, 1, kernel_size).normal_(mean=0, std=0.01), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

class SelfAttnLayer_MultipleHeads(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, numheads=1):
        super(SelfAttnLayer_MultipleHeads, self).__init__()
        self.numheads = numheads

        assert in_channels % self.numheads == 0, "in_channels should be divided by numheads. (example: in_channels: 16, numheads: 4)"
        assert out_channels % self.numheads == 0, "out_channels should be divided by numheads. (example: out_channels: 32, numheads: 4)"

        self.selfattns = nn.ModuleList([
            SelfAttnLayer(in_channels // numheads, out_channels // numheads, kernel_size, stride, padding, groups=1, bias=bias) for i in range(numheads)
        ])

    def forward(self, intensor):
        x_split = intensor.split(intensor.size(1) // self.numheads, dim=1)
        responses = []
        idx = 0
        for conv in self.selfattns:
            response = conv(x_split[idx])
            responses.append(response)
            idx += 1
        out = torch.cat(responses, dim=1)       
        return out

class SelfAttnLayer_RandomOverlappingChannelSplits_MultipleHeads(nn.Module):
    def __init__(self, in_channels, channel_overlap, out_channels, kernel_size, stride, padding, numheads, bias=False, groups=1):
        super(SelfAttnLayer_RandomOverlappingChannelSplits_MultipleHeads, self).__init__()
        self.numheads = numheads
        stepsize = math.ceil(in_channels / numheads)
        channel_shuffle = torch.randperm(in_channels).tolist()
        channel_splits = [channel_shuffle[i:i+stepsize] for i in range(0, len(channel_shuffle), stepsize)]
        channel_splits[0].extend(channel_splits[1][:channel_overlap])
        self.channel_splits = channel_splits

        self.selfattns = nn.ModuleList([
            SelfAttnLayer(len(self.channel_splits[i]), out_channels // numheads, kernel_size, stride, padding, groups=1, bias=bias) for i in range(numheads)
        ])

    def forward(self, intensor):
        responses = []
        idx = 0
        for conv in self.selfattns:
            response = conv(intensor[:,self.channel_splits[idx],:,:])
            responses.append(response)
            idx += 1
        out = torch.cat(responses, dim=1)       
        return out

# Src: https://github.com/leaderj1001/Stand-Alone-Self-Attention/blob/master/attention.py
class SelfAttnStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, m=4):
        super(SelfAttnStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        batch, _, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)
        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk->bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        for conv in self.value_conv:
            init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)

# Twins: Revisiting the Design of Spatial Attention in Vision Transformers
class GroupAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, k1=7, k2=7):
        super(GroupAttn, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.k1 = k1
        self.k2 = k2

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.k1, W // self.k2
        total_groups = h_group * w_group
        x = x.reshape(B, h_group, self.k1, w_group, self.k2, C).transpose(2, 3)
        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3,0,1,4,2,5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.k1, self.k2, C)
        x = attn.transpose(2,3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == "__main__":
    """
    inchannels = 32
    model = SelfAttnLayer_RandomOverlappingChannelSplits_4Heads(inchannels, 4, 64, 3, padding=1)
    x = torch.rand(10,inchannels,8,8)
    out = model(x)
    """
    
    """
    model = SelfAttnStem_ValuesComputer()
    x = torch.rand(10,3,128,128)
    x_p = F.pad(x, (1,0,1,0,0,0,0,0))
    out = model(x_p)
    """

    """
    device = torch.device('cuda:0')
    model = SelfAttnLayer_RandomOverlappingChannelSplits_MultipleHeads(8, 2, 16, 3, 1, 1, 4, False).to(device)
    out = model(torch.rand(5,16,32,32).to(device))
    """

    model = GroupAttn(dim=32, k1=7, k2=7)
    out = model(torch.rand(8,28*28, 32), 28, 28).transpose(2,1).view(8,32,28,28)
    _ = 1