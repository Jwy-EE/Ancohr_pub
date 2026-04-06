# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from layers import DCNv4_1D, PeriodEstimator

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

class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FGDM(nn.Module):
    """
    Frequency-Guided Deformable Module (FGDM) from the ANCHOR architecture.
    Provides physical navigation for time-domain geometric deformations 
    by injecting RFFT-extracted explicit dominant periods.
    """
    def __init__(self, dim):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a1 = nn.Sequential(
            nn.Conv1d(dim // 4, dim // 4, 1),
            nn.GELU(),
            # Deformable Operator (DefOp) with Gaussian Interpolation
            DCNv4_1D(
                channels=dim//4,
                kernel_size=7,
                dilation=1, # dynamically updated by FFT estimator
                group=dim // 4,
            )
        )
        self.v1 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.v11 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.v12 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.conv3_1 = nn.Conv1d(dim // 4, dim // 4, 3, padding=1, groups=dim//4)

        self.norm2 = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.a2 = nn.Sequential(
            nn.Conv1d(dim // 2, dim // 2, 1),
            nn.GELU(),
            # Deformable Operator (DefOp) with Gaussian Interpolation
            DCNv4_1D(
                channels=dim//2,
                kernel_size=9,
                dilation=1, # dynamically updated by FFT estimator
                group=dim // 2,
            )
        )
        self.v2 = nn.Conv1d(dim//2, dim//2, 1)
        self.v21 = nn.Conv1d(dim // 2, dim // 2, 1)
        self.v22 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.proj2 = nn.Conv1d(dim // 2, dim // 4, 1)
        self.conv3_2 = nn.Conv1d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.norm3 = LayerNorm(dim * 3 // 4, eps=1e-6, data_format="channels_first")
        self.a3 = nn.Sequential(
            nn.Conv1d(dim * 3 // 4, dim * 3 // 4, 1),
            nn.GELU(),
            # Deformable Operator (DefOp) with Gaussian Interpolation
            DCNv4_1D(
                channels=dim * 3 // 4,
                kernel_size=11,
                dilation=1,  # dynamically updated by FFT estimator
                group=dim * 3 // 4,
            )
        )
        self.v3 = nn.Conv1d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v31 = nn.Conv1d(dim * 3 // 4, dim * 3 // 4, 1)
        self.v32 = nn.Conv1d(dim // 4, dim // 4, 1)
        self.proj3 = nn.Conv1d(dim * 3 // 4, dim // 4, 1)
        self.conv3_3 = nn.Conv1d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4)

        self.dim = dim

    def forward(self, x):
        x = self.norm1(x)
        # Orthogonal subspace partitions as described in the ANCHOR paper
        x_split = torch.split(x, self.dim // 4, dim=1)
        
        # Stage 1: Asymmetric routing mechanism - small kernels / high frequency
        a1 = self.a1(x_split[0])
        if a1.shape[1] != self.dim // 4:  
            a1 = a1.permute(0, 2, 1)  # (N, L, C//4) -> (N, C//4, L)
        mul1 = a1 * self.v1(x_split[0])
        mul1 = self.v11(mul1)
        x1 = self.conv3_1(self.v12(x_split[1]))
        x1 = x1 + a1
        x1 = torch.cat((x1, mul1), dim=1)

        # Stage 2
        x1 = self.norm2(x1)
        a2 = self.a2(x1)  
        if a2.shape[1] != self.dim // 2:  
            a2 = a2.permute(0, 2, 1)  # (N, L, C//2) -> (N, C//2, L)
        mul2 = a2 * self.v2(x1)
        mul2 = self.v21(mul2)
        x2 = self.conv3_2(self.v22(x_split[2]))
        x2 = x2 + self.proj2(a2)
        x2 = torch.cat((x2, mul2), dim=1)

        # Stage 3: Large kernels / low frequency integration
        x2 = self.norm3(x2)
        a3 = self.a3(x2)  
        if a3.shape[1] != self.dim * 3 // 4:  
            a3 = a3.permute(0, 2, 1)  # (N, L, C*3//4) -> (N, C*3//4, L)
        mul3 = a3 * self.v3(x2)
        mul3 = self.v31(mul3)
        x3 = self.conv3_3(self.v32(x_split[3]))
        x3 = x3 + self.proj3(a3)
        x = torch.cat((x3, mul3), dim=1)

        return x

class Block(nn.Module):
    def __init__(self, dim,
                 drop=0.,
                 drop_path=0.,
                 mlp_ratio=4,
                 layer_scale_init_value=1e-5,
                 ):
        super().__init__()

        self.fgdm = FGDM(dim)
        self.mlp = MLPLayer(in_features=dim,
                            hidden_features=int(dim * mlp_ratio),
                            drop=drop)
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True)
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                        requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = build_norm_layer(dim, 'LN')
        self.norm2 = build_norm_layer(dim, 'LN')

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale.unsqueeze(-1)* self.fgdm(x))
        x = x.permute(0, 2, 1)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x.permute(0, 2, 1)

class Model(nn.Module):
    """
    ANCHOR: Adaptive Network based on Cascaded Harmonic Offset Routing
    Main Backbone Network compatible with Time-Series-Library.
    """
    def __init__(self, configs): 
        super().__init__()
        # 1. Acquire physical parameters and task instructions from configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # 2. Maintain hyperparameter defaults
        depths = getattr(configs, 'depths', [2, 2, 8, 2])
        dims = getattr(configs, 'dims', [64, 128, 256, 512])
        drop_path_rate = getattr(configs, 'drop_path', 0.)
        layer_scale_init_value = 1e-6
        drop = getattr(configs, 'dropout', 0.)

        self.fft_estimator = PeriodEstimator(top_k=3)

        # ====== Feature extraction network (ANCHOR Cascade Architecture) ======
        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
            nn.Conv1d(self.enc_in, dims[0] // 2, kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv1d(dims[0] // 2, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Dropout(drop)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv1d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() 
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        # Exact calculation of the true temporal stride after downsampling
        self.last_len = self.seq_len
        for _ in range(3):
            self.last_len = (self.last_len - 1) // 2 + 1
        
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(dims[-1] * self.last_len, self.seq_len * self.enc_in)
        elif self.task_name == 'classification':
            self.num_class = getattr(configs, 'num_class', 10)
            self.dropout_rate = getattr(configs, 'dropout', 0.1)
            self.class_head = MLPLayer(
                in_features=dims[-1] * self.last_len,
                hidden_features=dims[-1],
                out_features=self.num_class,
                drop=self.dropout_rate
            )
        elif self.task_name == 'short_term_forecast':
            self.head = nn.Linear(dims[-1] * self.last_len, self.pred_len * self.enc_in)
        else:
            raise ValueError(f"Not available: {self.task_name}")

        self.apply(self._init_weights)
        self.apply(self._init_deform_weights)
    
    # RFFT explicit physical prior injection
    def update_stage_dilations(self, stage, dilations):
        safe_p = list(dilations) if not isinstance(dilations, list) else dilations.copy()
        
        # Extreme degradation defense
        if len(safe_p) == 0:
            safe_p = [1, 1, 1]
            
        # Padding mechanism
        while len(safe_p) < 3:
            safe_p.append(safe_p[-1])

        # Execution of dynamic phase alignment prior
        for block in stage:
            if hasattr(block, 'fgdm'):
                fgdm = block.fgdm
                fgdm.a1[2].dilation = safe_p[0] 
                fgdm.a2[2].dilation = safe_p[1]
                fgdm.a3[2].dilation = safe_p[2]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def _init_deform_weights(self, m):
        if isinstance(m, DCNv4_1D):
            m._reset_parameters()

    def forward_features(self, x):
        topk_periods = self.fft_estimator(x) 
        current_stride=1
        
        for i in range(4):
            x = self.downsample_layers[i](x)
            if i>0:
                current_stride *=2
            stage_dilations=[max(1,p//current_stride) for p in topk_periods]
            self.update_stage_dilations(self.stages[i], stage_dilations)
            x = self.stages[i](x)
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        
        x = x_enc.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        
        # 2. FFT + FGDM Backbone (ANCHOR Architecture)
        x = self.forward_features(x)
        x = x.reshape(x.shape[0], -1)  # flatten
        
        # 3. Task distribution
        if self.task_name in ['anomaly_detection', 'imputation']:
            x = self.projection(x)
            x = x.reshape(x.shape[0], self.out_len, self.enc_in)
            return x

        elif self.task_name == 'classification':
            x = self.class_head(x)
            return x
            
        elif self.task_name in ['long_term_forecast', 'short_term_forecast', 'few_shot_forecast', 'zero_shot_forecast']:
            x = self.head(x)
            x = x.reshape(x.shape[0], self.pred_len, self.enc_in)
            return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x