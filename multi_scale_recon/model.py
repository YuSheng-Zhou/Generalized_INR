import torch.nn as nn
import torch
import math
import numpy as np


############################# PosEncoding ##################################
class PosEncoding(nn.Module):
    """reference: Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"""
    def __init__(self, in_dim, num_frequencies, k, include_coord=False):
        super().__init__()
        B = torch.randn(in_dim, num_frequencies) * k
        self.register_buffer("B", B)
        self.out_dim = num_frequencies * 2
        self.out_dim = self.out_dim + in_dim if include_coord else self.out_dim
        self.include_coord = include_coord

    def forward(self, x):
        x_proj = torch.matmul(2 * math.pi * x, self.B)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        if self.include_coord:
            out = torch.cat([x, out], dim=-1)
        return out


############################# ResEncoder ##################################
def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"invalid mode for variance scaling initializer: {mode}")
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init

def default_init(scale=1.):

    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')


class ResBlock(nn.Module):

    def __init__(self, hidden_dim, act, kernel_size=(5, 5), embed_dim=None):
        super().__init__()

        self.act = act
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(hidden_dim // 4, 32), num_channels=hidden_dim, eps=1e-6)
        self.Conv_0 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=(kernel_size[0] // 2,))
        if embed_dim is not None:
            self.Dense_0 = nn.Linear(embed_dim, hidden_dim)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(hidden_dim // 4, 32), num_channels=hidden_dim, eps=1e-6)
        self.Conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=(kernel_size[0] // 2,))

    def forward(self, x, scale_embedding=None):

        h = self.Conv_0(self.act(self.GroupNorm_0(x)))
        if scale_embedding is not None:
            h += self.Dense_0(self.act(scale_embedding))[:, :, None, None]
        h = self.Conv_1(self.act(self.GroupNorm_1(h)))

        return x + h

class ResEncoder(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, act, block_num=5, kernel_size=(5, 5), embed_dim=None):
        super().__init__()

        model = [nn.ReflectionPad2d(kernel_size[0] // 2),
                 nn.Conv2d(in_dim, hidden_dim, kernel_size)]
        for _ in range(block_num):
            model.append(ResBlock(hidden_dim, act, kernel_size, embed_dim))
        model += [nn.ReflectionPad2d(kernel_size[0] // 2),
                  nn.Conv2d(hidden_dim, out_dim, kernel_size)]
        self.model = nn.ModuleList(model)
        self.block_num = block_num

    def forward(self, x, scale_embedding=None):

        h = self.model[0](x)
        h = self.model[1](h)
        for i in range(self.block_num):
            h = self.model[2 + i](h, scale_embedding)
        h = self.model[-2](h)
        h = self.model[-1](h)

        return h


############################# our inr network ##################################
class network(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.act = nn.SiLU()

        # positional encoding
        if config.pos_encoding:
            self.pos_encoding = PosEncoding(2,
                                            num_frequencies=config.pos_fre_num,
                                            k=config.pos_scale,
                                            include_coord=config.include_coord)
            in_dim = self.pos_encoding.out_dim
        else:
            self.pos_encoding = None
            in_dim = 2

        # scale embedding
        if config.scale_embed:
            self.embedding_projection = PosEncoding(in_dim=1,
                                                    num_frequencies=config.pos_fre_num // 2,
                                                    k=config.pos_scale)
            embed_dim = self.embedding_projection.out_dim
        else:
            embed_dim = None

        # encoder
        self.encoder = ResEncoder(in_dim=config.image_in_dim,
                                  hidden_dim=config.enc_hidden_dim,
                                  out_dim=config.enc_out_dim,
                                  act=self.act,
                                  block_num=config.enc_block_num,
                                  kernel_size=config.enc_kernel_size,
                                  embed_dim=embed_dim)
        in_dim += config.enc_out_dim

        # MLP
        self.skips = config.mlp_skips
        MLP = [nn.Conv2d(in_channels=in_dim, out_channels=config.mlp_hidden_dim, kernel_size=(1, 1))]
        for i in range(1, config.mlp_num_layer - 1):
            if i in self.skips:
                MLP.append(nn.Conv2d(in_channels=config.mlp_hidden_dim + in_dim, out_channels=config.mlp_hidden_dim, kernel_size=(1, 1)))
            else:
                MLP.append(nn.Conv2d(in_channels=config.mlp_hidden_dim, out_channels=config.mlp_hidden_dim, kernel_size=(1, 1)))
        self.MLP = nn.ModuleList(MLP)
        self.out_linear1 = nn.Conv2d(in_channels=config.mlp_hidden_dim, out_channels=config.image_out_dim, kernel_size=(1, 1))
        self.out_linear2 = nn.Conv2d(in_channels=config.mlp_hidden_dim + config.image_in_dim, out_channels=config.image_out_dim, kernel_size=(1, 1))


    def forward(self, coords, prior_intensity, prior_phase, scale=None):

        # the scale project to a vector
        if scale is not None:
            scale_embed = self.embedding_projection(scale)
        else:
            scale_embed = None

        # encoding prior_intensity
        prior_intensity = self.encoder(prior_intensity, scale_embed)

        # position encoding
        if self.pos_encoding is not None:
            coords = self.pos_encoding(coords)

        # concat position and prior intensity
        coords = coords.permute(0, 3, 1, 2)
        features = torch.cat([coords, prior_intensity], dim=1).type(coords.dtype)

        # MLP reconstruction
        h = features
        for i, layer in enumerate(self.MLP):
            h = self.act(self.MLP[i](h))
            if i+1 in self.skips:
                h = torch.cat([features, h], dim=1)
        intensity = self.out_linear1(h)
        phase = self.out_linear2(torch.cat([h, prior_phase], dim=1))

        return intensity, phase
