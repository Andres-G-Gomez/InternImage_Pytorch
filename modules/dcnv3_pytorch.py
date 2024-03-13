import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)
    

class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)
    
    
def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')
    
    
def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))

    return (n & (n - 1) == 0) and n != 0


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = torch.nn.functional.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale

    
class DCNv3_pytorch(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0, # what is this?
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.center_feature_scale = center_feature_scale

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        
        self.offset = nn.Linear( 
            channels,
            group * kernel_size * kernel_size * 2)
        
        self.mask = nn.Linear(
            channels,
            group * kernel_size * kernel_size)
        
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()

        # creates a trainable parameter center_feature_scale_proj_weight with shape 
        # (group, channels) and parameter center_feature_scale_proj_bias with shape 
        # (group, ) initialized with zeros.
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape

        # Linear projection of the input feature map
        x = self.input_proj(input)
        x_proj = x # Preserve a reference for later use

        # Permute input dimensions for depthwise convolution
        x1 = input.permute(0, 3, 1, 2) # (N, C, H, W)
        x1 = self.dw_conv(x1) # Apply depthwise convolution
        offset = self.offset(x1) # Compute offsets for deformable convolution

        # self.mask(x1) outputs (N, H, W, self.group * kernel_size * kernel_size)
        # The output is reshaped to have dimensions (N, H, W, self.group, -1). 
        # This reshaping is performed to create a set of masks for each group and position 
        # in the output feature map.
        mask = self.mask(x1).reshape(N, H, W, self.group, -1) 
        # Applies the softmax function along the last dimension
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)

        # Perform deformable convolution using the core operation
        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale)

        # Optionally, apply center feature scaling
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                # linearly projects x1
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            
            # reshapes and repeats 'center_feature_scale' to be compatible with 'x'
            # N, H, W, groups -> N, H, W, groups, 1 -> 
            # N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)

            # Apply center feature scaling to the output
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
            #  modulate the importance of features at different spatial locations ?
        
        # Linear projection of the output
        x = self.output_proj(x)
      
        return x
    

def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    _, H_, W_, _ = spatial_shapes

    # Calculate the output dimensions after convolution
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    # Generate grid of reference points in the input space
    # ref_x and ref_y represent ~ HxW
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            # pad_h + 0.5,
            # H_ - pad_h - 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            # pad_w + 0.5,
            # W_ - pad_w - 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device))

    # Flattens and normalizes the reference points to the range [0, 1]
    ref_y = ref_y.reshape(-1)[None] / H_ 
    ref_x = ref_x.reshape(-1)[None] / W_

    # Stack the reference points and reshape for compatibility with deformable convolution
    ref = torch.stack((ref_x, ref_y), -1).reshape(
        1, H_out, W_out, 1, 2)

    return ref

def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes
    points_list = []

    # Generate a meshgrid of coordinates based on kernel size and dilation
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) +
            (kernel_w - 1) * dilation_w, kernel_w,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) +
            (kernel_h - 1) * dilation_h, kernel_h,
            dtype=torch.float32,
            device=device))

    # Normalize the coordinates to the range [-1, 1]
    points_list.extend([x / W_, y / H_])

    # Stack the normalized coordinates and reshape for deformable convolution compatibility
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).\
        repeat(1, group, 1).permute(1, 0, 2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)

    return grid

def dcnv3_core_pytorch(
        input, offset, mask, kernel_h,
        kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w, group,
        group_channels, offset_scale):
    
    # Pad input feature map
    input = F.pad(
        input,
        [0, 0, pad_h, pad_h, pad_w, pad_w])

    # Extract input dimensions
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    # Compute reference points and dilation grids
    ref = _get_reference_points(
        input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w) # (1, H_out, W_out, 1, 2)
    
    grid = _generate_dilation_grids(
        input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device) # (1, 1, 1, group * kernel_h * kernel_w, 2).
    
    # Compute spatial normalization factors
    # (1, 1, 1, group * kernel_h * kernel_w * 2) ??????
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).\
        repeat(1, 1, 1, group*kernel_h*kernel_w).to(input.device)
    
    # Compute sampling locations
    # (N, H_out, W_out, group * kernel_h * kernel_w * 2)
    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1).flatten(3, 4) + offset * offset_scale / spatial_norm
    
    # Calculate constants
    P_ = kernel_h * kernel_w
    sampling_grids = 2 * sampling_locations - 1

    
    # Reshape input 
    # N_, H_in, W_in, group*group_channels -> N_, H_in*W_in, group*group_channels -> 
    # N_, group*group_channels, H_in*W_in -> N_*group, group_channels, H_in, W_in
    input_ = input.view(N_, H_in*W_in, group*group_channels).transpose(1, 2).\
        reshape(N_*group, group_channels, H_in, W_in)
    # (group*N, group_channels, H_out + padding, W_out + padding)
    
    # Reshape sampling grid 
    # N_, H_out, W_out, group*P_*2 -> N_, H_out*W_out, group, P_, 2 -> 
    # N_, group, H_out*W_out, P_, 2 -> N_*group, H_out*W_out, P_, 2
    sampling_grid_ = sampling_grids.view(N_, H_out*W_out, group, P_, 2).transpose(1, 2).\
        flatten(0, 1)
    # (group*N, H_out * W_out, kernel_size * kernel_size, 2)

    # Reshape sampling_grid, perform bilinear interpretation if points are not available
    # N_*group, group_channels, H_out*W_out, P_
    sampling_input_ = F.grid_sample(
        input_, sampling_grid_, mode='bilinear', padding_mode='zeros', align_corners=False) # (group*N, group_channels, H_out * W_out, kernel_size * kernel_size)
    
    # Reshape mask
    # (N_, H_out, W_out, group*P_) -> N_, H_out*W_out, group, P_ -> 
    # (N_, group, H_out*W_out, P_) -> (N_*group, 1, H_out*W_out, P_)
    mask = mask.view(N_, H_out*W_out, group, P_).transpose(1, 2).\
        reshape(N_*group, 1, H_out*W_out, P_)
    # (group*N, 1, H_out * W_out, kernel_size * kernel_size)

    # for all {i,j}, sum over groups: x_g(p_{i,j} + location-aware offsets) * m_{g,k}(i,j)
    output = (sampling_input_ * mask).sum(-1).view(N_,
                                                   group*group_channels, H_out*W_out) # (N, channels, H_out * W_out)
    
    # Transpose and reshape the output
    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous() # (N, H_out, W_out, channels)


