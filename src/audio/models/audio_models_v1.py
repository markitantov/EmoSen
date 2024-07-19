import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)


class AudioModelV16(Wav2Vec2PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.tanh_feats = nn.Tanh()
        
        self.f_size = 128
        self.feature_downsample = nn.Linear(1024, self.f_size)
        
        self.time_downsample = torch.nn.Sequential(
            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=3),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.MaxPool1d(3),
            torch.nn.ReLU(),
            
            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=3),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.MaxPool1d(3),
            torch.nn.ReLU(),

            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=3),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.ReLU(),
        )
        
        cfg = xLSTMBlockStackConfig(
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
            embedding_dim=128,
            slstm_at=[1],
        )
        self.xlstm = xLSTMBlockStack(cfg)
        
        self.fc_emo1 = nn.Linear(128, 64)
        self.fc_emo2 = nn.Linear(64, 6)
        
        self.fc_sen1 = nn.Linear(128, 64)
        self.fc_sen2 = nn.Linear(64, 3)
        
        self.init_weights()
        self.unfreeze_last_n_blocks(0)
        
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
        x = self.tanh_feats(x)
        return x

    def forward(self, x):
        x = self.get_features(x)
        x = self.feature_downsample(x)
        
        x = self.xlstm(x)
        
        x = x.permute(0, 2, 1)
        x = self.time_downsample(x)
        x = x.squeeze()
        
        emo = self.fc_emo1(x)
        emo = self.fc_emo2(emo)
        
        sen = self.fc_sen1(x)
        sen = self.fc_sen2(sen)
        
        return {'emo': emo, 'sen': sen}
    
    
class AudioModelV26(Wav2Vec2PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.tanh_feats = nn.Tanh()
        
        self.f_size = 128
        self.feature_downsample = nn.Linear(512, self.f_size)
        
        self.time_downsample = torch.nn.Sequential(
            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=3),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.MaxPool1d(3),
            torch.nn.ReLU(),
            
            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=3),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.MaxPool1d(3),
            torch.nn.ReLU(),

            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=3),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.ReLU(),
        )
        
        cfg = xLSTMBlockStackConfig(
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
            embedding_dim=128,
            slstm_at=[1],
        )
        self.xlstm = xLSTMBlockStack(cfg)
        
        self.fc_emo1 = nn.Linear(128, 64)
        self.fc_emo2 = nn.Linear(64, 6)
        
        self.fc_sen1 = nn.Linear(128, 64)
        self.fc_sen2 = nn.Linear(64, 3)
        
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
        x = self.wav2vec2.feature_extractor(x)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        x = self.get_features(x)
        x = self.feature_downsample(x)
        
        x = self.xlstm(x)
        
        x = x.permute(0, 2, 1)
        x = self.time_downsample(x)
        x = x.squeeze()
        
        emo = self.fc_emo1(x)
        emo = self.fc_emo2(emo)
        
        sen = self.fc_sen1(x)
        sen = self.fc_sen2(sen)
        
        return {'emo': emo, 'sen': sen}
    
    
class AudioModelV17(Wav2Vec2PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.tanh_feats = nn.Tanh()
        
        self.f_size = 128
        self.feature_downsample = nn.Linear(1024, self.f_size)
        
        self.time_downsample = torch.nn.Sequential(
            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=3),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.MaxPool1d(3),
            torch.nn.ReLU(),
            
            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=3),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.MaxPool1d(3),
            torch.nn.ReLU(),

            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=3),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.ReLU(),
        )
        
        cfg = xLSTMBlockStackConfig(
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
            embedding_dim=128,
            slstm_at=[1],
        )
        self.xlstm = xLSTMBlockStack(cfg)
        
        self.fc_emo1 = nn.Linear(128, 64)
        self.fc_emo2 = nn.Linear(64, 7)
        
        self.fc_sen1 = nn.Linear(128, 64)
        self.fc_sen2 = nn.Linear(64, 3)
        
        self.init_weights()
        self.unfreeze_last_n_blocks(0)
        
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
        x = self.tanh_feats(x)
        return x

    def forward(self, x):
        x = self.get_features(x)
        x = self.feature_downsample(x)
        
        x = self.xlstm(x)
        
        x = x.permute(0, 2, 1)
        x = self.time_downsample(x)
        x = x.squeeze()
        
        emo = self.fc_emo1(x)
        emo = self.fc_emo2(emo)
        
        sen = self.fc_sen1(x)
        sen = self.fc_sen2(sen)
        
        return {'emo': emo, 'sen': sen}
    
    
class AudioModelV27(Wav2Vec2PreTrainedModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.tanh_feats = nn.Tanh()
        
        self.f_size = 128
        self.feature_downsample = nn.Linear(512, self.f_size)
        
        self.time_downsample = torch.nn.Sequential(
            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=3),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.MaxPool1d(3),
            torch.nn.ReLU(),
            
            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=3),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.MaxPool1d(3),
            torch.nn.ReLU(),

            torch.nn.Conv1d(self.f_size, self.f_size, kernel_size=3),
            torch.nn.BatchNorm1d(self.f_size),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.ReLU(),
        )
        
        cfg = xLSTMBlockStackConfig(
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
            embedding_dim=128,
            slstm_at=[1],
        )
        self.xlstm = xLSTMBlockStack(cfg)
        
        self.fc_emo1 = nn.Linear(128, 64)
        self.fc_emo2 = nn.Linear(64, 7)
        
        self.fc_sen1 = nn.Linear(128, 64)
        self.fc_sen2 = nn.Linear(64, 3)
        
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
        x = self.wav2vec2.feature_extractor(x)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        x = self.get_features(x)
        x = self.feature_downsample(x)
        
        x = self.xlstm(x)
        
        x = x.permute(0, 2, 1)
        x = self.time_downsample(x)
        x = x.squeeze()
        
        emo = self.fc_emo1(x)
        emo = self.fc_emo2(emo)
        
        sen = self.fc_sen1(x)
        sen = self.fc_sen2(sen)
        
        return {'emo': emo, 'sen': sen}
    
    
if __name__ == "__main__":      
    sampling_rate = 16000
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inp_v = torch.zeros((4, sampling_rate * 4)).to(device)
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    model_cls = AudioModelV27

    model = model_cls.from_pretrained(model_name)
    
    model.to(device)

    res = model(inp_v)
    print(res)
    print(res['emo'].shape)
    print(res['sen'].shape)