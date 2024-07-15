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

from audio.models.common import TransformerLayer, ClassificationHead


class AudioModelV3(Wav2Vec2PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        
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
            slstm_at=[1],
        ))
        
        self.tanh = nn.Tanh()
        self.cl_head = ClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
        self.init_weights()
        self.freeze_conv_only()
        
    def freeze_conv_only(self):
        # freeze conv
        for param in self.wav2vec2.feature_extractor.conv_layers.parameters():
            param.requires_grad = False
            
    def unfreeze_last_n_blocks(self, num_blocks: int) -> None:
        # freeze all wav2vec
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # unfreeze last n transformer blocks
        for i in range(0, num_blocks):
            for param in self.wav2vec2.encoder.layers[-1 * (i + 1)].parameters():
                param.requires_grad = True

    def get_features(self, x):
        x = self.wav2vec2(x)[0]
        return x

    def forward(self, x):
        x = self.get_features(x)
        x = self.feature_downsample(x)
        
        x = self.xlstm(x)
        x = self.tanh(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)


class AudioModelV4(Wav2Vec2PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)

        self.f_size = 1024
        self.tl1 = TransformerLayer(input_dim=self.f_size, num_heads=8, dropout=0.1, positional_encoding=True)
        self.tl2 = TransformerLayer(input_dim=self.f_size, num_heads=16, dropout=0.1, positional_encoding=True)

        self.tanh = nn.Tanh()
        self.cl_head = ClassificationHead(input_size=self.f_size, 
                                          out_emo=config.out_emo, 
                                          out_sen=config.out_sen)
        
        self.init_weights()
        self.freeze_conv_only()
    
    def freeze_conv_only(self):
        # freeze conv
        for param in self.wav2vec2.feature_extractor.conv_layers.parameters():
            param.requires_grad = False
            
    def unfreeze_last_n_blocks(self, num_blocks: int) -> None:
        # freeze all wav2vec
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        # unfreeze last n transformer blocks
        for i in range(0, num_blocks):
            for param in self.wav2vec2.encoder.layers[-1 * (i + 1)].parameters():
                param.requires_grad = True

    def get_features(self, x):
        x = self.wav2vec2(x)[0]
        return x

    def forward(self, x):
        x = self.get_features(x)

        x = self.tl1(query=x, key=x, value=x)
        x = self.tl2(query=x, key=x, value=x)
        x = self.tanh(x)

        x = torch.mean(x, dim=1)
        
        return self.cl_head(x)
    
    
if __name__ == "__main__":
    sampling_rate = 16000
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inp_v = torch.zeros((4, sampling_rate * 4)).to(device)
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'

    model_cls = AudioModelV4

    config = AutoConfig.from_pretrained(model_name)
    config.out_emo = 7
    config.out_sen = 3

    model = model_cls.from_pretrained(model_name, config=config)
    
    model.to(device)

    res = model(inp_v)
    print(res)
    print(res['emo'].shape)
    print(res['sen'].shape)