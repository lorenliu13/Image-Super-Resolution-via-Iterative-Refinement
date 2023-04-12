import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    """
    A feature-wise affine transformation.
    """
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        """
        use_affine_level: whether to use affine level transformation
        """

        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        # a linear layer: y = Wx + b
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
            # in_channels: the number of input features
            # out_channels * (1 + self.use_affine_level): the number of output features
            # if use_affine_level is ture, the output dimension is doubled
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0] # extract the batch size

        # if use affine level transform
        if self.use_affine_level:
            # apply noise_function to the noise embedding
            # reshape the result, splitting it along the channel dimension
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            # apply feature wise affine transform
            x = (1 + gamma) * x + beta
        else:
            # if not use affine level transform
            # apply noise_func(noise_embed) on x
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    """
    Upsample layer

    """
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest") # use nearest to do upsampling
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    """
    Downsample layer

    """
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules
class Block(nn.Module):
    """
    A basic building block. It consists of a normalization layer, activation function, optional drop out, and a conv layer
    """
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim), # a group normalization layer
            Swish(), # a activation function, f(x) = x * sigmod(x)
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(), # randomly set a fraction of input to 0
            nn.Conv2d(dim, dim_out, 3, padding=1) # 2d conv layer with 3*3 kernel pad 1
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    """
    A resnet block class
    """
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):

        super().__init__() # call "nn.Module"'s constructor

        #
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        # add one block
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        # add another block
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        # 1*1 conv layer to match the input and output dimensions
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x) # pass the tensor through block 1
        h = self.noise_func(h, time_emb) # apply the noise function
        h = self.block2(h) # pass the tensor through block 2
        return h + self.res_conv(x) # add the result to input tensor, create a residual connection


class SelfAttention(nn.Module):
    """
    A self-attention layer to process the feature maps and compute attention weights between different spatial positions
    """
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        # a group normalization layer normalizes the input features across specified number of groups
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        # 1 * 1 convolution, responsible to compute the query, key, and value matrices
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        # produce the final output feature maps
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape # extract the dimension of input tensor
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input) # normalize the input tensor using the group normalization layer
        # compute the query, key, value tensors using the 1*1 conv layer.
        # reshape the result
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        # this qkv layer is initialized with random weights, as the model is trained, these weights are updated
        # through backpropagation
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        # matrix multiplication of the query and key tensors
        # scaling the result by the square root of the channel size to get the raw attention scores
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        # reshape the raw attention scores
        attn = attn.view(batch, n_head, height, width, -1)
        # apply the softmax function along the last dimension to get attention weights
        attn = torch.softmax(attn, -1)
        # reshape teh attention weights back to the original dimensions
        attn = attn.view(batch, n_head, height, width, height, width)
        # matrix multiplication of the attention weights and the value tensor
        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        # pass the output through 1*1 convolutional layer to get final output map
        out = self.out(out.view(batch, channel, height, width))
        # add the input tensor to the output to create a residual connection
        return out + input


class ResnetBlocWithAttn(nn.Module):
    """
    ResNet block with optional attention
    """
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        """
        Initialize the class and accepts several parameters
        dim: the number of input channels
        dim_out: the number of output channels
        noise_level_emb_dim: the dimension of noise level embedding
        norm_groups: the number of groups for group normalization
        dropout: the dropout rate
        with_attn: whether or not to include a self-attention layer
        """
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups) # append a SelfAttention layer

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):
    """
    Defines a U-Net architecture
    """
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
    ):
        """
        Initialize U-Net with various arguments
        in_channels:
        out_channels:
        norm_groups:
        channel_mults:
        attn_res: attention resolution
        res_block: number of "ResnetBlocWithAttn" layers
        dropout:
        with_noise_level_emb:
        image_size:
        """

        super().__init__()

        if with_noise_level_emb: # if include a noise level multiple layer perceptron
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            ) # a PositionalEncoding layer + linear layer + swish activation + linear layer
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        # start to create the downsampling part
        num_mults = len(channel_mults)
        pre_channel = inner_channel # the first channel number
        feat_channels = [pre_channel]
        now_res = image_size # current spatial resolution of the feature maps
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)] # the first layer, input is input dimension, output is the first channel number
        for ind in range(num_mults): # for each number in the channel multipliers
            is_last = (ind == num_mults - 1) # check if it is the last multiplier
            use_attn = (now_res in attn_res) # if the current resolution is in the attention resolution list
            channel_mult = inner_channel * channel_mults[ind] # get the target channel number
            for _ in range(0, res_blocks): # repeat "res_blocks" times
                # append a ResnetBlocWithAttn layer
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult) # append the channel number to feat_channels
                pre_channel = channel_mult # update the pre_channel to current channel size
            if not is_last: # if it is not the last multiplier, reduce the spatial dimension by a factor of 2
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs) # add the downsampling layers to ModuleList

        # middle parts of the U-Net
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True), # use the attention mechanism
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False) # not use the attention mechanism
        ])

        ups = []
        for ind in reversed(range(num_mults)): # reverse the multiplier
            is_last = (ind < 1) # if the current iteration is less than 1
            use_attn = (now_res in attn_res) # check if the current spatial resolution is in the attn_res list
            # to decide whether to use the attention mechanism
            channel_mult = inner_channel * channel_mults[ind] # calculate the number of output channels
            for _ in range(0, res_blocks+1): # add "res_blocks+1" number of ResnetBlockWithAttn layers
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last: # if it is not the last one, add an upsample layer
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time):
        """
        Forward pass of the U-Net, process the input tensor through downsampling, middle, and upsampling layer
        x: input tensor
        time: noise level tensor
        """
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)
