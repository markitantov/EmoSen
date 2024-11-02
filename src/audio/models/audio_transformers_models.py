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

from common.models.common import TransformerLayer, ClassificationHead, SmallClassificationHead
from common.utils.common import AttrDict
from common.data.utils import define_context_length

class AudioModelT1(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelT1, self).__init__()
        self.config = config
        
        self.f_size = 1024
        
        self.tl1 = TransformerLayer(input_dim=self.f_size, num_heads=4, dropout=0.1, positional_encoding=True)
        self.tl2 = TransformerLayer(input_dim=self.f_size, num_heads=4, dropout=0.1, positional_encoding=True)
        
        self.selu = nn.SELU()
        self.cl_head = SmallClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
    def forward(self, x):        
        x = self.tl1(x, x, x)
        x = self.selu(x)
        
        x = self.tl2(x, x, x)
        x = self.selu(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)
    

class AudioModelT2(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelT2, self).__init__()
        self.config = config
        
        self.f_size = 1024
        
        self.tl1 = TransformerLayer(input_dim=self.f_size, num_heads=4, dropout=0.1, positional_encoding=True)
        self.tl2 = TransformerLayer(input_dim=self.f_size, num_heads=4, dropout=0.1, positional_encoding=True)
        
        self.selu = nn.SELU()
        self.cl_head = ClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
    def forward(self, x):        
        x = self.tl1(x, x, x)
        x = self.selu(x)
        
        x = self.tl2(x, x, x)
        x = self.selu(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)
    
    
class AudioModelT3(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelT3, self).__init__()
        self.config = config
        
        self.f_size = 1024
        
        self.tl1 = TransformerLayer(input_dim=self.f_size, num_heads=2, dropout=0.1, positional_encoding=True)
        self.tl2 = TransformerLayer(input_dim=self.f_size, num_heads=4, dropout=0.1, positional_encoding=True)
        self.tl3 = TransformerLayer(input_dim=self.f_size, num_heads=8, dropout=0.1, positional_encoding=True)
        
        self.selu = nn.SELU()
        self.cl_head = SmallClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
    def forward(self, x):        
        x = self.tl1(x, x, x)
        x = self.selu(x)
        
        x = self.tl2(x, x, x)
        x = self.selu(x)
        
        x = self.tl3(x, x, x)
        x = self.selu(x)
        
        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)
    
    
class AudioModelT4(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelT4, self).__init__()
        self.config = config
        
        self.f_size = 1024
        
        self.tl1 = TransformerLayer(input_dim=self.f_size, num_heads=8, dropout=0.1, positional_encoding=True)
        self.tl2 = TransformerLayer(input_dim=self.f_size, num_heads=8, dropout=0.1, positional_encoding=True)
        
        self.selu = nn.SELU()
        self.cl_head = SmallClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
    def forward(self, x):        
        x = self.tl1(x, x, x)
        x = self.selu(x)
        
        x = self.tl2(x, x, x)
        x = self.selu(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)


class AudioModelT5(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelT5, self).__init__()
        self.config = config
        
        self.f_size = 1024
        
        self.tl1 = TransformerLayer(input_dim=self.f_size, num_heads=4, dropout=0.1, positional_encoding=True)
        self.tl2 = TransformerLayer(input_dim=self.f_size, num_heads=4, dropout=0.1, positional_encoding=True)
        self.tl3 = TransformerLayer(input_dim=self.f_size, num_heads=4, dropout=0.1, positional_encoding=True)
        
        self.selu = nn.SELU()
        self.cl_head = SmallClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
    def forward(self, x):        
        x = self.tl1(x, x, x)
        x = self.selu(x)
        
        x = self.tl2(x, x, x)
        x = self.selu(x)
        
        x = self.tl3(x, x, x)
        x = self.selu(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)
    
    
class AudioModelT6(nn.Module):
    def __init__(self, config) -> None:
        super(AudioModelT6, self).__init__()
        self.config = config
        
        self.f_size = 1024
        
        self.tl1 = TransformerLayer(input_dim=self.f_size, num_heads=8, dropout=0.1, positional_encoding=True)
        self.tl2 = TransformerLayer(input_dim=self.f_size, num_heads=8, dropout=0.1, positional_encoding=True)
        self.tl3 = TransformerLayer(input_dim=self.f_size, num_heads=8, dropout=0.1, positional_encoding=True)
        
        self.selu = nn.SELU()
        self.cl_head = SmallClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
    def forward(self, x):        
        x = self.tl1(x, x, x)
        x = self.selu(x)
        
        x = self.tl2(x, x, x)
        x = self.selu(x)
        
        x = self.tl3(x, x, x)
        x = self.selu(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)
    

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inp_v = torch.zeros((4, define_context_length(1), 1024)).to(device)
    m_clses = [AudioModelT1, AudioModelT2, AudioModelT3, AudioModelT4, AudioModelT5, AudioModelT6]

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