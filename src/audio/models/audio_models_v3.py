import sys

sys.path.append('src')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from transformers import AutoConfig

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

from mamba_ssm.modules.mamba2 import Mamba2

from audio.models.common import TransformerLayer, ClassificationHead
from audio.utils.common import AttrDict


class AudioModelV3(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelV3, self).__init__()
        self.config = config
        
        self.f_size = 256
        self.feature_downsample = nn.Linear(1024, self.f_size)
        
        self.xlstm = xLSTMBlockStack(xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    num_heads=4,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=199,
            num_blocks=7,
            embedding_dim=256,
            slstm_at=[1, 4],
        ))
        
        self.selu = nn.SELU()
        self.cl_head = ClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
    def forward(self, x):
        x = self.feature_downsample(x)
        
        x = self.xlstm(x)
        x = self.selu(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)


class AudioModelV4(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelV4, self).__init__()
        self.config = config

        self.f_size = 1024
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(input_dim=self.f_size, num_heads=8, dropout=0.1, positional_encoding=True) for i in range(5)
        ])
        
        self.selu = nn.SELU()
        self.cl_head = ClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
    def forward(self, x):
        for i, l in enumerate(self.transformer_layers):
            x = l(query=x, key=x, value=x)
            x = self.selu(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)
    
    
class AudioModelV5(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelV5, self).__init__()
        self.config = config

        self.f_size = 1024
        self.mamba_layers = nn.ModuleList([
            Mamba2(d_model=self.f_size, d_state=64, d_conv=4, expand=2) for i in range(5)
        ])
        
        self.selu = nn.SELU()
        self.cl_head = ClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
    
    def forward(self, x):
        for i, l in enumerate(self.mamba_layers):
            x = l(x)
            x = self.selu(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)
    
    
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inp_v = torch.zeros((4, 199, 1024)).to(device)
    model_clses = [AudioModelV3, AudioModelV4, AudioModelV5]

    for m_cls in model_clses:
        config = AttrDict()
        config.out_emo = 7
        config.out_sen = 3

        model = m_cls(config=config)
    
        model.to(device)

        res = model(inp_v)
        print(res)
        print(res['emo'].shape)
        print(res['sen'].shape)