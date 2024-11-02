import sys

sys.path.append('src')

import torch
import torch.nn as nn

from transformers import AutoConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from common.utils.common import AttrDict
from common.data.utils import define_context_length


class Audio2023Model(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.gru = nn.GRU(input_size=1024, hidden_size=256, dropout=.5, num_layers=2, batch_first=True)
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(256, 1)
        self.dp = nn.Dropout(p=.5)

        self.fc_emo = nn.Linear(199, self.config.out_emo)
        self.fc_sen = nn.Linear(199, self.config.out_sen)
        
        self.init_weights()
        
        # freeze conv
        for param in self.wav2vec2.feature_extractor.conv_layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = self.wav2vec2(x)

        x, h = self.gru(outputs[0])
        x = self.relu(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dp(x)

        x = x.view(x.size(0), -1)
        
        emo = self.fc_emo(x)
        sen = self.fc_sen(x)
        return {'emo': emo, 'sen': sen}


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inp_v = torch.zeros((4, 64000)).to(device)
    m_clses = [Audio2023Model]

    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'

    for m_cls in m_clses:
        config = AutoConfig.from_pretrained(model_name)

        config.out_emo = 7
        config.out_sen = 3

        model = m_cls.from_pretrained(model_name, config=config)
    
        model.to(device)

        res = model(inp_v)
        print(res)
        print(res['emo'].shape)
        print(res['sen'].shape)