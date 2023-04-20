import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
# from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from natten import NeighborhoodAttention1D as NeighborhoodAttention
import math

# def trunc_normal_(tensor, mean=0., std=1.):
#     __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
# -------------------------------------------------------------------------------
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        if (not torch.is_tensor(out)) and len(
                out) == 2:  ## might have problem if the last batch size is 2, needs to be changed
            out_, attn = out
            return out_ + x, attn
        else:
            return out + x
        # return self.fn(x, **kwargs) + x


# -------------------------------------------------------------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        if len(x) == 2:
            x, attn = x
            return self.fn(self.norm(x), **kwargs), attn
        else:
            return self.fn(self.norm(x), **kwargs)


# -------------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, {'attn': attn, 'dot_qk': dots}


# -------------------------------------------------------------------------------

class Cross_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(2, dim=0)
        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, kv) * self.scale
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, kv)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return torch.cat((out, out), dim=0), {'attn': attn, 'dot_qk': dots}

## **********************************************
## code from https://github.com/microsoft/ExtreMA/blob/bf30b64ec0046524d0c373b761df323b9328f8dc/vits.py#L306
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # self.scale = 1

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        attn_o = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls, attn_o


class Block_CA(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Class_Attention,
                 Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, x_cls):
        u = torch.cat((x_cls, x), dim=1)
        out, attn_o = self.attn(self.norm1(u))

        if self.gamma_1 is None:
            x_cls = x_cls + self.drop_path(out)
            x_cls = x_cls + self.drop_path(self.mlp(self.norm2(x_cls)))
        else:
            x_cls = x_cls + self.drop_path(self.gamma_1 * out)
            x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))

        return x_cls, attn_o


## *************************************

def fix_pos_emb(patch_dim, emb_dim):
    position = torch.arange(patch_dim).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
    pe = torch.zeros(1, patch_dim, emb_dim)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


## reducing band_embedding dimension each block
class Transformer_r(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode, use_class_attn=False,
                 spatial_attn=True, cls_token=1):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.spatial_attn = spatial_attn
        self.cls_token = cls_token
        if self.spatial_attn:
            for i in range(depth):
                if i > 0:
                    channel_dim_previous = int(num_channel * (depth - i + 1) / depth)
                else:
                    channel_dim_previous = num_channel
                channel_dim = int(num_channel * (depth - i) / depth)
                self.layers.append(nn.ModuleList([
                    nn.Linear(channel_dim_previous + 1, channel_dim + 1),
                    Residual(PreNorm(dim, NeighborhoodAttention(dim, kernel_size=3, dilation=1, num_heads=heads,
                                                                qkv_bias=True, attn_drop=dropout, proj_drop=dropout))),
                    # Residual(PreNorm(num_channel,
                    #                  NeighborhoodAttention(num_channel, kernel_size=3, dilation=1, num_heads=heads,
                    #                                             qkv_bias=True, attn_drop=dropout, proj_drop=dropout))),
                    # Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                    Residual(PreNorm(channel_dim + self.cls_token,
                                     Attention(channel_dim + self.cls_token, heads=heads, dim_head=dim_head,
                                               dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
                ]))
        else:
            for i in range(depth):
                if i > 0:
                    channel_dim_previous = int(num_channel * (depth - i + 1) / depth)
                else:
                    channel_dim_previous = num_channel
                channel_dim = int(num_channel * (depth - i) / depth)
                self.layers.append(nn.ModuleList([
                    nn.Linear(channel_dim_previous + 1, channel_dim + 1),
                    Residual(PreNorm(dim, NeighborhoodAttention(dim, kernel_size=3, dilation=1, num_heads=heads,
                                                                qkv_bias=True, attn_drop=dropout, proj_drop=dropout))),
                    # Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
                ]))

        self.use_class_attn = use_class_attn
        self.mode = mode
        self.skipcat = nn.ModuleList([])
        self.projs = nn.ModuleList([])
        if mode == 'CAF':
            for i in range(depth - 2):
                # self.skipcat.append(nn.Conv2d(num_channel + self.cls_token, num_channel + self.cls_token, [1, 2], 1, 0))
                if i > 1:
                    channel_dim_previous = int(num_channel * (depth - i + 1) / depth)
                    # channel_dim_previous = channel_dim
                else:
                    channel_dim_previous = num_channel
                channel_dim = int(num_channel * (depth - i - 2 + 1) / depth)
                self.skipcat.append(nn.Conv2d(channel_dim + self.cls_token, channel_dim + self.cls_token, [1, 2], 1, 0))
                self.projs.append(nn.Linear(channel_dim_previous + self.cls_token, channel_dim + self.cls_token))

        if self.use_class_attn:
            self.blocks_token = nn.ModuleList([
                Block_CA(
                    dim=dim, num_heads=heads, mlp_ratio=1 / 8, qkv_bias=False, qk_scale=None,
                    drop=0.1, attn_drop=0.1, drop_path=0.0, norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU)
                for _ in range(1)])

    def forward(self, x, mask=None):
        if self.mode == 'ViT':
            if self.spatial_attn:
                for attn, attn2, ff in self.layers:
                    x, attn_map = attn(x, mask=mask)
                    x, attn_map_s = attn2(x.permute(0, 2, 1), mask=mask)
                    x = x.permute(0, 2, 1)
                    x = ff(x)

                    if self.use_class_attn:
                        for _, blk in enumerate(self.blocks_token):
                            cls_token_ca, _ = blk(x[:, 1:, ], x[:, 0, :].unsqueeze(1))
                        x = torch.cat((cls_token_ca, x[:, 1:, ]), dim=1)  # torch.Size([64, 201, 64])
            else:
                for attn, ff in self.layers:
                    x, attn_map = attn(x, mask=mask)
                    x = ff(x)

                    if self.use_class_attn:
                        for _, blk in enumerate(self.blocks_token):
                            cls_token_ca, _ = blk(x[:, 1:, ], x[:, 0, :].unsqueeze(1))
                        x = torch.cat((cls_token_ca, x[:, 1:, ]), dim=1)  # torch.Size([64, 201, 64])

        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            if self.spatial_attn:
                # for attn, attn2, ff in self.layers:
                for lp, attn, attn2, ff in self.layers:
                    last_output.append(x)
                    if nl > 1:
                        # x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                        # print(x.size(), last_output[nl - 2].size())
                        x = self.skipcat[nl - 2](
                            torch.cat([x.unsqueeze(3),
                                       self.projs[nl - 2](last_output[nl - 2].permute(0, 2, 1)).permute(0, 2,
                                                                                                        1).unsqueeze(
                                           3)], dim=3)).squeeze(3)

                    # ## B1 + NAT
                    # x = attn(x)
                    # x_cls = x[:,0:self.cls_token,]
                    # x = attn2(x[:,self.cls_token:,].permute(0, 2, 1))
                    # x = torch.cat([x_cls, x.permute(0, 2, 1)], dim=1)
                    # x = ff(x)

                    # ## B1 + NAT (only for spectral)
                    # x = attn(x)
                    # x, attm_map = attn2(x.permute(0, 2, 1), mask=mask)
                    # x = x.permute(0, 2, 1)
                    # x = ff(x)

                    ## B1 + NAT (only for spectral)
                    x = attn(x)
                    x = lp(x.permute(0, 2, 1))
                    x, attm_map = attn2(x, mask=mask)
                    x = x.permute(0, 2, 1)
                    x = ff(x)

                    # ## B1: Spectra Attn → Spatial Attn → FF, with class_token in B -> (N,B+1,P) → (N,0,P) →class
                    # x, attn_map_spe = attn(x, mask = mask)
                    # # x_cls = x[:,0,:].unsqueeze(1)
                    # x, attn_map = attn2(x.permute(0, 2, 1), mask=mask)
                    # # x, attn_map_s = attn2(x[:,1:,].permute(0,2,1), mask=mask) ## performed a bit worse: 86.39/74.59/87.79
                    # x = x.permute(0,2,1)
                    # # x = torch.cat((x_cls, x), dim=1)
                    # x = ff(x)

                    # ## B2: Spatial Attn → Spectra Attn → FF, with class_token in B -> (N,B+1,P) → (N,0,P) →class
                    # x, attn_map = attn2(x.permute(0,2,1), mask = mask)
                    # # x_cls = x[:,0,:].unsqueeze(1)
                    # x, attn_map_spe = attn(x.permute(0, 2, 1), mask=mask)
                    # # x, attn_map_s = attn2(x[:,1:,].permute(0,2,1), mask=mask) ## performed a bit worse: 86.39/74.59/87.79
                    # # x = x.permute(0,2,1)
                    # # x = torch.cat((x_cls, x), dim=1)
                    # x = ff(x)

                    if self.use_class_attn:
                        for _, blk in enumerate(self.blocks_token):
                            cls_token_ca, _ = blk(x[:, self.cls_token:, ], x[:, 0:self.cls_token, :])
                        #     cls_token_ca = blk(x[:, 1:, ], x_cls.unsqueeze(1))
                        # cls_token_ca += x[:, 0, ].unsqueeze(1) #performed a bit worse
                        x = torch.cat((cls_token_ca, x[:, self.cls_token:, ]), dim=1)  # torch.Size([64, 201, 64])

                    nl += 1
            else:
                # for attn, ff in self.layers:
                for lp, attn, ff in self.layers:
                    last_output.append(x)
                    if nl > 1:
                        x = self.skipcat[nl - 2](
                            torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
                    x = lp(x.permute(0, 2, 1)).permute(0, 2, 1)
                    x = attn(x)
                    # x, attn_map = attn(x, mask=mask)
                    x = ff(x)

                    if self.use_class_attn:
                        for _, blk in enumerate(self.blocks_token):
                            cls_token_ca, _ = blk(x[:, self.cls_token:, ], x[:, 0:self.cls_token, :])
                        x = torch.cat((cls_token_ca, x[:, self.cls_token:, ]), dim=1)  # torch.Size([64, 201, 64])

                    nl += 1
        return x
        # return x, attn_map


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode, use_class_attn=False,
                 spatial_attn=True, cls_token=1):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.spatial_attn = spatial_attn
        self.cls_token = cls_token
        if self.spatial_attn:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    # Residual(PreNorm(dim, NeighborhoodAttention(dim, kernel_size=3, dilation=1, num_heads=heads,
                    #                                             qkv_bias=True, attn_drop=dropout, proj_drop=dropout))),
                    # Residual(PreNorm(num_channel,
                    #                  NeighborhoodAttention(num_channel, kernel_size=3, dilation=1, num_heads=heads,
                    #                                             qkv_bias=True, attn_drop=dropout, proj_drop=dropout))),
                    Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                    Residual(PreNorm(num_channel + self.cls_token,
                                     Attention(num_channel + self.cls_token, heads=heads, dim_head=dim_head,
                                               dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    # Residual(PreNorm(dim, NeighborhoodAttention(dim, kernel_size=3, dilation=1, num_heads=heads,
                    #                                             qkv_bias=True, attn_drop=dropout, proj_drop=dropout))),
                    Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
                ]))

        self.use_class_attn = use_class_attn
        self.mode = mode
        self.skipcat = nn.ModuleList([])
        if mode == 'CAF':
            for _ in range(depth - 2):
                self.skipcat.append(nn.Conv2d(num_channel + self.cls_token, num_channel + self.cls_token, [1, 2], 1, 0))

        if self.use_class_attn:
            self.blocks_token = nn.ModuleList([
                Block_CA(
                    dim=dim, num_heads=heads, mlp_ratio=1 / 8, qkv_bias=False, qk_scale=None,
                    drop=0.1, attn_drop=0.1, drop_path=0.0, norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU)
                for _ in range(1)])

    def forward(self, x, mask=None):
        if self.mode == 'ViT':
            if self.spatial_attn:
                for attn, attn2, ff in self.layers:
                    x, attn_map = attn(x, mask=mask)
                    x, attn_map_s = attn2(x.permute(0, 2, 1), mask=mask)
                    x = x.permute(0, 2, 1)
                    x = ff(x)

                    if self.use_class_attn:
                        for _, blk in enumerate(self.blocks_token):
                            cls_token_ca, _ = blk(x[:, 1:, ], x[:, 0, :].unsqueeze(1))
                        x = torch.cat((cls_token_ca, x[:, 1:, ]), dim=1)  # torch.Size([64, 201, 64])
            else:
                for attn, ff in self.layers:
                    x, attn_map = attn(x, mask=mask)
                    x = ff(x)

                    if self.use_class_attn:
                        for _, blk in enumerate(self.blocks_token):
                            cls_token_ca, _ = blk(x[:, 1:, ], x[:, 0, :].unsqueeze(1))
                        x = torch.cat((cls_token_ca, x[:, 1:, ]), dim=1)  # torch.Size([64, 201, 64])

        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            if self.spatial_attn:
                for attn, attn2, ff in self.layers:
                    last_output.append(x)
                    if nl > 1:
                        x = self.skipcat[nl - 2](
                            torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)

                    # ## B1 + NAT
                    # x = attn(x)
                    # x_cls = x[:,0:self.cls_token,]
                    # x = attn2(x[:,self.cls_token:,].permute(0, 2, 1))
                    # x = torch.cat([x_cls, x.permute(0, 2, 1)], dim=1)
                    # x = ff(x)

                    # ## B1 + NAT (only for spectral)
                    # x = attn(x)
                    # x, attm_map = attn2(x.permute(0, 2, 1), mask=mask)
                    # x = x.permute(0, 2, 1)
                    # x = ff(x)

                    ## B1: Spectra Attn → Spatial Attn → FF, with class_token in B -> (N,B+1,P) → (N,0,P) →class
                    x, attn_map_spe = attn(x, mask=mask)
                    # x_cls = x[:,0,:].unsqueeze(1)
                    x, attn_map = attn2(x.permute(0, 2, 1), mask=mask)
                    # x, attn_map_s = attn2(x[:,1:,].permute(0,2,1), mask=mask) ## performed a bit worse: 86.39/74.59/87.79
                    x = x.permute(0, 2, 1)
                    # x = torch.cat((x_cls, x), dim=1)
                    x = ff(x)

                    # ## B2: Spatial Attn → Spectra Attn → FF, with class_token in B -> (N,B+1,P) → (N,0,P) →class
                    # x, attn_map = attn2(x.permute(0,2,1), mask = mask)
                    # # x_cls = x[:,0,:].unsqueeze(1)
                    # x, attn_map_spe = attn(x.permute(0, 2, 1), mask=mask)
                    # # x, attn_map_s = attn2(x[:,1:,].permute(0,2,1), mask=mask) ## performed a bit worse: 86.39/74.59/87.79
                    # # x = x.permute(0,2,1)
                    # # x = torch.cat((x_cls, x), dim=1)
                    # x = ff(x)

                    if self.use_class_attn:
                        for _, blk in enumerate(self.blocks_token):
                            cls_token_ca, _ = blk(x[:, self.cls_token:, ], x[:, 0:self.cls_token, :])
                        #     cls_token_ca = blk(x[:, 1:, ], x_cls.unsqueeze(1))
                        # cls_token_ca += x[:, 0, ].unsqueeze(1) #performed a bit worse
                        x = torch.cat((cls_token_ca, x[:, self.cls_token:, ]), dim=1)  # torch.Size([64, 201, 64])

                    nl += 1
            else:
                for attn, ff in self.layers:
                    last_output.append(x)
                    if nl > 1:
                        x = self.skipcat[nl - 2](
                            torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
                    # x = attn(x) #for NAT
                    x, attn_map = attn(x, mask=mask)
                    x = ff(x)

                    if self.use_class_attn:
                        for _, blk in enumerate(self.blocks_token):
                            cls_token_ca, _ = blk(x[:, self.cls_token:, ], x[:, 0:self.cls_token, :])
                        x = torch.cat((cls_token_ca, x[:, self.cls_token:, ]), dim=1)  # torch.Size([64, 201, 64])

                    nl += 1
        # return x ## for NAT
        return x, attn_map


class Transformer_spatial(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode, use_class_attn=False,
                 cls_token=1):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.cls_token = cls_token
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(num_channel + self.cls_token,
                                 Attention(num_channel + self.cls_token, heads=heads, dim_head=dim_head,
                                           dropout=dropout))),
                Residual(PreNorm(num_channel + self.cls_token,
                                 FeedForward(num_channel + self.cls_token, mlp_head, dropout=dropout)))
            ]))

        self.use_class_attn = use_class_attn
        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(dim, dim, [1, 2], 1, 0))

        if self.use_class_attn:
            self.blocks_token = nn.ModuleList([
                Block_CA(
                    dim=dim, num_heads=heads, mlp_ratio=1 / 8, qkv_bias=False, qk_scale=None,
                    drop=0.1, attn_drop=0.1, drop_path=0.0, norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU)
                for _ in range(1)])

    def forward(self, x, mask=None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x, attn_map = attn(x, mask=mask)
                x = ff(x)

                if self.use_class_attn:
                    for _, blk in enumerate(self.blocks_token):
                        cls_token_ca = blk(x[:, 1:, ], x[:, 0, :].unsqueeze(1))
                    x = torch.cat((cls_token_ca, x[:, 1:, ]), dim=1)  # torch.Size([64, 201, 64])

        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    x = self.skipcat[nl - 2](
                        torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
                x, attn_map = attn(x, mask=mask)
                x = ff(x)

                if self.use_class_attn:
                    for _, blk in enumerate(self.blocks_token):
                        cls_token_ca = blk(x[:, 1:, ], x[:, 0, :].unsqueeze(1))
                    x = torch.cat((cls_token_ca, x[:, 1:, ]), dim=1)  # torch.Size([64, 201, 64])

                nl += 1
        return x, attn_map


class VisionTransformerEncoder(nn.Module):

    def __init__(self,
                 image_size,
                 near_band,
                 num_patches,
                 num_classes,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 pool='cls',
                 channels=1,
                 dim_head=16,
                 dropout=0.,
                 emb_dropout=0.,
                 mode='ViT',
                 mask_ratio=0.75,
                 init_scaler=0.,
                 mask_clf=None,
                 mask_method=None,
                 use_class_attn=False,
                 align_loss=None,
                 spatial_attn=False,
                 use_sar=False,
                 use_se=False,
                 ):
        super().__init__()
        self.image_size = image_size
        self.near_band = near_band
        patch_dim = image_size ** 2 * near_band
        band_dim = num_patches
        self.use_cls = True
        self.num_classes = num_classes
        self.num_patches = num_patches
        self.mask_clf = mask_clf
        self.mask_method = mask_method
        self.use_class_attn = use_class_attn
        self.use_sar = use_sar
        self.align_loss = align_loss
        if self.use_cls:
            self.patch_cls = 1  # num_classes
            self.cls_token = nn.Parameter(torch.randn(1, self.patch_cls, dim))
            self.cls_token_ca = nn.Parameter(torch.zeros(1, self.patch_cls, dim))
            # self.cls_proj = nn.Sequential(nn.LayerNorm(num_patches), nn.Linear(num_patches, dim))
        else:
            self.patch_cls = 0
        self.pos_embedding = nn.Parameter(
            torch.randn(1, band_dim + self.patch_cls, dim))  # randomly initialised learnable embedding
        # self.pos_embedding = nn.Parameter(torch.zeros(1, band_dim + self.patch_cls, dim)) #truncated normal initialised learnable embedding
        # self.pos_embedding = fix_pos_emb(band_dim + self.patch_cls, dim).cuda() #sincos fixed position embedding

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # self.band_to_embedding = nn.Linear(num_patches, band_dim)
        self.dropout = nn.Dropout(emb_dropout)
        # if self.mask_clf:
        #     self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, int(num_patches * (1 - mask_ratio)), mode)
        if mode == 'ViT':  ## for pretraining using partial bands
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout,
                                           num_patches - int(num_patches * mask_ratio), mode,
                                           use_class_attn=use_class_attn, spatial_attn=spatial_attn)
        else:  ## for finetuning using full bands
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, band_dim, mode,
                                           use_class_attn=use_class_attn, spatial_attn=spatial_attn,
                                           cls_token=self.patch_cls)
            # self.transformer_s = Transformer_spatial(dim, depth, heads, dim_head, mlp_dim, dropout, band_dim, mode,
            #                                use_class_attn=use_class_attn, cls_token=self.patch_cls)
        # self.pos_embedding_s = nn.Parameter(torch.randn(1, dim, num_patches+1))
        self.pool = pool
        self.to_latent = nn.Identity()
        # self.norm = nn.LayerNorm(dim)
        if num_classes > 0:
            # if self.mask_method in ['high', 'low', 'high_random']:
            #     self.bandvoting_clf = BandVoting(mask_ratio, h_dim=num_patches, mask_method=self.mask_method)
            if use_class_attn:
                ## code from https://github.com/microsoft/ExtreMA/blob/bf30b64ec0046524d0c373b761df323b9328f8dc/vits.py#L306
                self.blocks_token = nn.ModuleList([
                    Block_CA(
                        dim=dim, num_heads=heads, mlp_ratio=1 / 8, qkv_bias=False, qk_scale=None,
                        drop=dropout, attn_drop=emb_dropout, drop_path=0.0, norm_layer=nn.LayerNorm,
                        act_layer=nn.GELU)
                    for _ in range(1)])

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim),
                                      nn.Linear(dim, num_classes)) if num_classes > 0 else nn.Identity()
        # if use_sar:
        #     self.sar = BlobLoss()
        #     # self.sar_spec = BlobLoss_1d()

    def forward_features(self, x):

        # if self.use_cls:
        #     x_cls = x.reshape(x.size(0), x.size(1), self.image_size,self.image_size,self.near_band)[:,:,self.image_size//2,self.image_size//2,self.near_band//2]

        x = self.patch_to_embedding(x)
        # x = self.band_to_embedding(x.permute(0,2,1)).permute(0,2,1)
        b, n, c = x.shape

        if self.use_cls:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            # cls_tokens = self.cls_proj(x_cls).unsqueeze(1)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + self.patch_cls)]
            x = self.dropout(x)
        else:
            x = self.dropout(x + self.pos_embedding)
        return x

    # def forward_features_s(self, x):
    #
    #     # # x = self.patch_to_embedding(x)
    #     b, n, c = x.shape
    #
    #     if self.use_cls:
    #         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
    #         # cls_tokens = self.cls_proj(x_cls).unsqueeze(1)
    #         x = torch.cat((cls_tokens, x), dim=1)
    #         # x += self.pos_embedding_s[:, :(n + 1)]
    #         # x = self.dropout(x)
    #     # else:
    #     #     x = self.dropout(x + self.pos_embedding_s)
    #     return self.dropout(x.permute(0,2,1) + self.pos_embedding_s)

    def forward(self, x, masking_pos=None, mask=None, return_attn=False):
        # x = self.patch_to_embedding(x)
        # x_s = self.forward_features_s(x)
        x = self.forward_features(x)
        b, _, c = x.shape

        # if self.mask_method != None and masking_pos != None and self.num_classes == 0:
        # # if self.mask_method != None and masking_pos != None:
        #     if self.use_cls:
        #         cls_em_tokens = x[:,0:1,:]
        #         x_vis = x[:,1:,:]
        #         x = x_vis[~masking_pos].reshape(b, -1, c)
        #         x = torch.cat((cls_em_tokens, x), dim = 1)
        #     else:
        #         x = x[~masking_pos].reshape(b, -1, c)
        # else:
        #     # ## using mask for finetuning
        #     # if self.mask_clf and self.num_classes > 0:
        #     #     if self.mask_clf not in ['high', 'low', 'high_random'] and masking_pos is None:
        #     #         if self.use_cls:
        #     #             cls_em_tokens = x[:, 0:1, :]
        #     #             x = x[:, 1:, :].reshape(b, -1, c)
        #     #             x = torch.cat((cls_em_tokens, x), dim=1)
        #     #         else:
        #     #             x = x.reshape(b, -1, c)
        #     #     else:
        #     #         if self.use_cls:
        #     #             cls_em_tokens = x[:, 0:1, :]
        #     #             x_vis = x[:, 1:, :]
        #     #             if self.mask_clf in ['high', 'low', 'high_random']:
        #     #                 masking_pos = self.bandvoting_clf(x_vis)
        #     #             x = x_vis[~masking_pos].reshape(b, -1, c)
        #     #             x = torch.cat((cls_em_tokens, x), dim=1)
        #     #         else:
        #     #             if self.mask_clf in ['high', 'low', 'high_random']:
        #     #                 masking_pos = self.bandvoting_clf(x)
        #     #             x = x[~masking_pos].reshape(b, -1, c)
        #     # else:
        #     if self.use_cls:
        #         cls_em_tokens = x[:,0:1,:]
        #         x = x[:,1:,:].reshape(b, -1, c)
        #         x = torch.cat((cls_em_tokens, x), dim = 1)
        #     else:
        #         x = x.reshape(b, -1, c)

        # print('input size of encoder: ', x.size()) #torch.Size([64, 51, 64])
        x, attn_map = self.transformer(x, mask)
        # x = self.transformer(x, mask) ## for NAT
        # x_s, _ = self.transformer_s(x_s, mask)

        # ## class-level fusion: spectra transformer + spatial transformer
        # x_spe, attn_map = self.transformer(x, mask)
        # x_spa, _ = self.transformer_s(x.permute(0,2,1), mask)
        # x = x_spe + x_spa.permute(0,2,1)

        if self.use_sar:
            attn_mat = attn_map['dot_qk'][:, :, 0:self.patch_cls, self.patch_cls:].max(2).values
            # loss_sar = self.sar_spec(attn_mat)
            loss_sar = self.sar(attn_mat)

        # ## add class attention after the encoder
        # if self.num_classes > 0 and self.use_class_attn:
        # # if self.use_class_attn:
        #     # # xx = x
        #     # # print(self.cls_token.size()) #1,1,64
        #     # # print(x.size()) #64,201,64
        #     # cls_token_ca = self.cls_token_ca.expand(x.size(0), -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #     # for _, blk in enumerate(self.blocks_token):
        #     #     cls_token_ca = blk(x,cls_token_ca)
        #     # cls_token_ca = torch.cat((cls_token_ca, x[:,0,:].unsqueeze(1)), dim=1).mean(1).unsqueeze(1)
        #     # x = torch.cat((cls_token_ca, x[:,1:,]), dim=1) #torch.Size([64, 201, 64])
        #
        #     # cls_tokens = self.cls_proj(x_cls).unsqueeze(1)
        #     for _, blk in enumerate(self.blocks_token):
        #         cls_token_ca, _ = blk(x[:, self.patch_cls:, ], x[:, 0:self.patch_cls, :])
        #         # cls_token_ca = blk(x[:,1:,],cls_tokens)
        #     # cls_token_ca = torch.cat((cls_token_ca, x[:, 0, :].unsqueeze(1)), dim=1).mean(1).unsqueeze(1)
        #     x = torch.cat((cls_token_ca, x[:, self.patch_cls:, ]), dim=1)  # torch.Size([64, 201, 64])
        # # x += x_s.permute(0,2,1)

        if self.num_classes > 0:
            if self.align_loss is None:
                x = x.mean(axis=1) if self.pool == 'mean' else self.to_latent(x[:, 0])
                if return_attn:
                    return self.mlp_head(x), attn_map['attn']
                else:
                    return self.mlp_head(x)
            else:
                loss_align = torch.cdist(x[:, 0:self.patch_cls].max(1).values, x[:, self.patch_cls:].mean(1)).mean()
                x = x.mean(axis=1) if self.pool == 'mean' else self.to_latent(x[:, 0:self.patch_cls].max(1).values)
                if self.use_sar:
                    return self.mlp_head(x), loss_align, loss_sar
                else:
                    return self.mlp_head(x), loss_align
        elif (self.use_cls and self.num_classes == 0):
            if self.align_loss is None:
                # x = x[:,1:]
                return self.mlp_head(x)
            else:
                # return self.mlp_head(x), x
                loss_align = torch.cdist(x[:, 0], x[:, 1:].mean(1))
                loss_align = torch.nn.functional.normalize(loss_align).mean()
                return self.mlp_head(x), loss_align
        else:
            x = self.to_latent(x)
        return self.mlp_head(x)


