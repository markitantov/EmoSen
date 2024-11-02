import sys

sys.path.append('src')

import torch
import torch.nn as nn

from transformers import AutoConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from mamba_ssm.modules.mamba2 import Mamba2

from common.utils.common import AttrDict
from common.data.utils import define_context_length

from common.models.common import SmallClassificationHead, TransformerLayer

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class AudioModelWT(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)

        self.f_size = 1024

        self.tl1 = TransformerLayer(input_dim=self.f_size, num_heads=4, dropout=0.1, positional_encoding=True)
        self.tl2 = TransformerLayer(input_dim=self.f_size, num_heads=4, dropout=0.1, positional_encoding=True)

        self.fc1 = nn.Linear(1024, 1)
        self.dp = nn.Dropout(p=.5)
        
        self.selu = nn.SELU()
        self.relu = nn.ReLU()
        self.cl_head = SmallClassificationHead(input_size=199, 
                                               out_emo=config.out_emo, 
                                               out_sen=config.out_sen)
        
        self.init_weights()
        
        # freeze conv
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        for param in self.wav2vec2.feature_extractor.conv_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = self.wav2vec2(x)

        x = self.tl1(outputs[0], outputs[0], outputs[0])
        x = self.selu(x)
        
        features = self.tl2(x, x, x)
        x = self.selu(features)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dp(x)

        x = x.view(x.size(0), -1)
        
        return self.cl_head(x)
    

class AudioModelWM(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)

        self.f_size = 1024

        self.ml1 = Mamba2(d_model=self.f_size, d_state=64, d_conv=4, expand=2)
        self.ml2 = Mamba2(d_model=self.f_size, d_state=128, d_conv=4, expand=4)
        self.ml3 = Mamba2(d_model=self.f_size, d_state=64, d_conv=4, expand=8)

        self.fc1 = nn.Linear(1024, 1)
        self.dp = nn.Dropout(p=.5)
        
        self.selu = nn.SELU()
        self.relu = nn.ReLU()
        self.cl_head = SmallClassificationHead(input_size=199, 
                                               out_emo=config.out_emo, 
                                               out_sen=config.out_sen)
        
        self.init_weights()
        
        # freeze conv
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        for param in self.wav2vec2.feature_extractor.conv_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = self.wav2vec2(x)

        x = self.ml1(outputs[0])
        x = self.selu(x)
        
        x = self.ml2(x)
        x = self.selu(x)
        
        x = self.ml3(x)
        x = self.selu(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dp(x)

        x = x.view(x.size(0), -1)
        
        return self.cl_head(x)
    

class AudioModelWX(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)

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

        self.fc1 = nn.Linear(1024, 1)
        self.dp = nn.Dropout(p=.5)
        
        self.selu = nn.SELU()
        self.relu = nn.ReLU()
        self.cl_head = SmallClassificationHead(input_size=199, 
                                               out_emo=config.out_emo, 
                                               out_sen=config.out_sen)
        
        self.init_weights()
        
        # freeze conv
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        for param in self.wav2vec2.feature_extractor.conv_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = self.wav2vec2(x)
        
        x = self.xlstm(outputs[0])
        x = self.selu(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dp(x)

        x = x.view(x.size(0), -1)
        
        return self.cl_head(x)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inp_v = torch.zeros((4, 64000)).to(device)
    m_clses = [AudioModelWT]

    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'

    for m_cls in m_clses:
        config = AutoConfig.from_pretrained(model_name)

        config.out_emo = 7
        config.out_sen = 3

        model = m_cls.from_pretrained(model_name, config=config)

        print(model)
    
        model.to(device)

        res = model(inp_v)
        print(res)
        print(res['emo'].shape)
        print(res['sen'].shape)