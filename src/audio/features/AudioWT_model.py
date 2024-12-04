import sys

sys.path.append('src')

import torch.nn as nn

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from common.models.common_layers import SmallClassificationHead, TransformerLayer


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

    def forward(self, x, with_features=False):
        outputs = self.wav2vec2(x)

        x = self.tl1(outputs[0], outputs[0], outputs[0])
        x = self.selu(x)
        
        features = self.tl2(x, x, x)
        x = self.selu(features)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dp(x)

        x = x.view(x.size(0), -1)
        
        if with_features:
            return self.cl_head(x), features
        else:
            return self.cl_head(x)
