import sys

sys.path.append('src')

import torch
import torch.nn as nn

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

from audio.models.common import ClassificationHead, SmallClassificationHead
from audio.utils.common import AttrDict

from audio.data.common import define_context_length

class AudioModelX1(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelX1, self).__init__()
        self.config = config
        
        self.f_size = 1024
        
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
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=config.context_length,
            num_blocks=3,
            embedding_dim=1024,
            slstm_at=[1],
        ))
        
        self.selu = nn.SELU()
        self.cl_head = SmallClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
    def forward(self, x):        
        x = self.xlstm(x)
        x = self.selu(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)
    

class AudioModelX2(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelX2, self).__init__()
        self.config = config
        
        self.f_size = 1024
        
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
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=config.context_length,
            num_blocks=3,
            embedding_dim=1024,
            slstm_at=[1],
        ))
        
        self.selu = nn.SELU()
        self.cl_head = ClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
    def forward(self, x):        
        x = self.xlstm(x)
        x = self.selu(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)
    
    
class AudioModelX3(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelX3, self).__init__()
        self.config = config
        
        self.f_size = 1024
        
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
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=config.context_length,
            num_blocks=2,
            embedding_dim=1024,
            slstm_at=[1],
        ))
        
        self.selu = nn.SELU()
        self.cl_head = SmallClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
    def forward(self, x):        
        x = self.xlstm(x)
        x = self.selu(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)
    
    
class AudioModelX4(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelX4, self).__init__()
        self.config = config
        
        self.f_size = 1024
        
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
                ),
            ),
            context_length=config.context_length,
            num_blocks=2,
            embedding_dim=1024,
            slstm_at=[1],
        ))
        
        self.selu = nn.SELU()
        self.cl_head = SmallClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
    def forward(self, x):        
        x = self.xlstm(x)
        x = self.selu(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)


class AudioModelX5(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelX5, self).__init__()
        self.config = config
        
        self.f_size = 1024
        
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
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=config.context_length,
            num_blocks=3,
            embedding_dim=1024,
            slstm_at=[0, 2],
        ))
        
        self.selu = nn.SELU()
        self.cl_head = SmallClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
    def forward(self, x):        
        x = self.xlstm(x)
        x = self.selu(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)
    
    
class AudioModelX6(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelX6, self).__init__()
        self.config = config
        
        self.f_size = 1024
        
        self.xlstm = xLSTMBlockStack(xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
                )
            ),
            context_length=config.context_length,
            num_blocks=2,
            embedding_dim=1024,
            slstm_at=[],
            add_post_blocks_norm=True,
        ))
        
        self.selu = nn.SELU()
        self.cl_head = SmallClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
    def forward(self, x):        
        x = self.xlstm(x)
        x = self.selu(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)
    

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inp_v = torch.zeros((4, define_context_length(1), 1024)).to(device)
    m_clses = [AudioModelX1, AudioModelX2, AudioModelX3, AudioModelX4, AudioModelX5, AudioModelX6]

    for m_cls in m_clses:
        config = AttrDict()
        config.out_emo = 7
        config.out_sen = 3

        model = m_cls(config=config)
    
        model.to(device)

        res = model(inp_v)
        print(res)
        print(res['emo'].shape)
        print(res['sen'].shape)