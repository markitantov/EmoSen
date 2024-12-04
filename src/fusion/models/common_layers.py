import sys

sys.path.append('src')

import torch
import torch.nn as nn
import torch.nn.functional as F
    

from common.models.common_layers import PermuteLayer, StatPoolLayer, TransformerLayer


class MultimodalDownsamplerS32Mean(nn.Module):
    def __init__(self, out_t_size=6, out_f_size=512, features_only=False):
        super(MultimodalDownsamplerS32Mean, self).__init__()
        
        a_modules = [PermuteLayer((0, 2, 1)), nn.Conv1d(1024, out_f_size, 1), nn.GELU()]
        v_modules = [nn.Identity()]
        t_modules = [PermuteLayer((0, 2, 1)), nn.Conv1d(1024, out_f_size, 1), nn.GELU(), PermuteLayer((0, 2, 1))]

        if features_only:
            a_modules.extend([PermuteLayer((0, 2, 1))])
        else: 
            a_modules.extend([
                nn.Conv1d(out_f_size, out_f_size, kernel_size=7, stride=3), 
                nn.GroupNorm(num_groups=out_f_size, num_channels=out_f_size, affine=True),
                nn.GELU(),
                nn.Conv1d(out_f_size, out_f_size, kernel_size=7, stride=3),
                nn.GELU(),
                PermuteLayer((0, 2, 1)),
            ])
            
            t_modules.extend([
                PermuteLayer((0, 2, 1)), 
                nn.Conv1d(out_f_size, out_f_size, kernel_size=7),
                nn.GroupNorm(num_groups=out_f_size, num_channels=out_f_size, affine=True),
                nn.GELU(),
                nn.Conv1d(out_f_size, out_f_size, kernel_size=7),
                nn.GELU(),
                PermuteLayer((0, 2, 1)),
            ])
            
        self.a_downsampler = nn.Sequential(*a_modules)
        self.v_downsampler = nn.Sequential(*v_modules)
        self.t_downsampler = nn.Sequential(*t_modules)

    def forward(self, x):
        a_features, v_features, t_features = x
        a_features = self.a_downsampler(a_features)
        v_features = self.v_downsampler(v_features)        
        t_features = self.t_downsampler(t_features)
        
        return a_features, v_features, t_features
    
    
class MultimodalDownsamplerS32MeanSTD(nn.Module):
    def __init__(self, out_t_size=6, out_f_size=512, features_only=False):
        super(MultimodalDownsamplerS32MeanSTD, self).__init__()
        
        a_modules = [PermuteLayer((0, 2, 1)), nn.Conv1d(2048, out_f_size, 1), nn.GELU()]
        v_modules = [nn.Identity()]
        t_modules = [nn.Identity()]

        if features_only:
            a_modules.extend([PermuteLayer((0, 2, 1))])
        else: 
            a_modules.extend([
                nn.Conv1d(out_f_size, out_f_size, kernel_size=7, stride=3), 
                nn.GroupNorm(num_groups=out_f_size, num_channels=out_f_size, affine=True),
                nn.GELU(),
                nn.Conv1d(out_f_size, out_f_size, kernel_size=7, stride=3),
                nn.GELU(),
                PermuteLayer((0, 2, 1)),
            ])
            
            t_modules.extend([
                PermuteLayer((0, 2, 1)), 
                nn.Conv1d(out_f_size, out_f_size, kernel_size=7),
                nn.GELU(),
                nn.Conv1d(out_f_size, out_f_size, kernel_size=7),
                nn.GELU(),
                PermuteLayer((0, 2, 1)),
            ])
            
        self.a_downsampler = nn.Sequential(*a_modules)
        self.v_downsampler = nn.Sequential(*v_modules)
        self.t_downsampler = nn.Sequential(*t_modules)

    def forward(self, x):
        a_features, v_features, t_features = x
        a_features = self.a_downsampler(a_features)
        v_features = self.v_downsampler(v_features)        
        t_features = self.t_downsampler(t_features)
        
        return a_features, v_features, t_features
    

class PredictionsUpsampler(nn.Module):
    def __init__(self, inp_size=10, out_t_size=20, out_f_size=512, trainable=True):
        super(PredictionsUpsampler, self).__init__()
        upsample_modules = [nn.Identity()]
        if trainable:        
            upsample_modules.extend([
                nn.ConvTranspose1d(inp_size, out_f_size, out_t_size),
                nn.ReLU(),
                PermuteLayer((0, 2, 1)),
            ])
        else:
            upsample_modules.extend([
                nn.Upsample(scale_factor=out_t_size), 
                PermuteLayer((0, 2, 1)), 
                nn.Upsample(scale_factor=out_f_size / inp_size)
            ])

        self.prediction_upsampler = nn.Sequential(*upsample_modules)

    def forward(self, x):
        predicts = x.unsqueeze(-1) # bs, 10, 1
        return self.prediction_upsampler(predicts)


class TripleFusion(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(TripleFusion, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.block_a_v = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_a_t = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        
        self.block_v_a= TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_v_t = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        
        self.block_t_a= TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_t_v = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)

        # cross attention between avt and vta.
        self.block_avt_vta = TransformerLayer(input_dim=self.in_features * 3, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_vta_avt = TransformerLayer(input_dim=self.in_features * 3, num_heads=8, dropout=0.1, positional_encoding=True)

        self.stp = StatPoolLayer(dim=1)
        self.cross_att_dense_layer = torch.nn.Linear(self.in_features * 3 * 2 * 2, self.out_features)
        self.cross_att_batch_norm = torch.nn.BatchNorm1d(self.out_features)
        self.cross_att_activation = torch.nn.Tanh()

    def forward(self, x):
        a, v, t = x
        # a - main feature; v, t - supportive features
        a_v = self.block_a_v(query=v, key=a, value=a)
        a_t = self.block_a_t(query=t, key=a, value=a)

        # v - main feature; a, t - supportive features
        v_a = self.block_v_a(query=a, key=v, value=v)
        v_t = self.block_v_t(query=t, key=v, value=v)

        # t - main feature; a, v - supportive features
        t_a = self.block_t_a(query=a, key=t, value=t)
        t_v = self.block_t_v(query=v, key=t, value=t)

        avt = torch.cat((a_v, v_t, t_a), dim=-1)
        vta = torch.cat((v_a, t_v, a_t), dim=-1)

        avt_vta = self.block_avt_vta(query=vta, key=avt, value=avt)
        vta_avt = self.block_vta_avt(query=avt, key=vta, value=vta)

        a_v_t = torch.cat((avt_vta, vta_avt), dim=-1) # b, 6, 512 * 3 * 2
        mean_std_pool = self.stp(a_v_t) # Output size (batch_size, in_features * 3 * 2 * 2)

        # dense layer + batch norm + activation
        output = self.cross_att_dense_layer(mean_std_pool)  # Output size (batch_size, 256)
        output = self.cross_att_batch_norm(output)  # Output size (batch_size, 256)
        output = self.cross_att_activation(output)  # Output size (batch_size, 256)
        return output
    

class TripleFusionO1(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(TripleFusionO1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.block_a_v = CrossModalAttention(embed_dim=self.in_features, num_heads=8, dropout=0.1)
        self.block_a_t = CrossModalAttention(embed_dim=self.in_features, num_heads=8, dropout=0.1)
        
        self.block_v_a = CrossModalAttention(embed_dim=self.in_features, num_heads=8, dropout=0.1)
        self.block_v_t = CrossModalAttention(embed_dim=self.in_features, num_heads=8, dropout=0.1)
        
        self.block_t_a= CrossModalAttention(embed_dim=self.in_features, num_heads=8, dropout=0.1)
        self.block_t_v = CrossModalAttention(embed_dim=self.in_features, num_heads=8, dropout=0.1)

        # cross attention between avt and vta.
        self.block_avt_vta = CrossModalAttention(embed_dim=self.in_features, num_heads=8, dropout=0.1)
        self.block_vta_avt = CrossModalAttention(embed_dim=self.in_features, num_heads=8, dropout=0.1)

        self.stp = StatPoolLayer(dim=1)
        self.cross_att_dense_layer = torch.nn.Linear(self.in_features * 4, self.out_features)
        self.cross_att_batch_norm = torch.nn.BatchNorm1d(self.out_features)
        self.cross_att_activation = torch.nn.Tanh()

    def forward(self, x):
        a, v, t = x
        # a - main feature; v, t - supportive features
        a_v = self.block_a_v(query=v, key=a, value=a)
        a_t = self.block_a_t(query=t, key=a, value=a)

        # v - main feature; a, t - supportive features
        v_a = self.block_v_a(query=a, key=v, value=v)
        v_t = self.block_v_t(query=t, key=v, value=v)

        # t - main feature; a, v - supportive features
        t_a = self.block_t_a(query=a, key=t, value=t)
        t_v = self.block_t_v(query=v, key=t, value=t)

        avt = torch.cat((a_v, v_t, t_a), dim=1) # b, 225, 256
        vta = torch.cat((v_a, t_v, a_t), dim=1)

        avt_vta = self.block_avt_vta(query=vta, key=avt, value=avt)
        vta_avt = self.block_vta_avt(query=avt, key=vta, value=vta)

        a_v_t = torch.cat((avt_vta, vta_avt), dim=-1) # b, 225, 512
        mean_std_pool = self.stp(a_v_t) # Output size (batch_size, in_features * 4)

        # dense layer + batch norm + activation
        output = self.cross_att_dense_layer(mean_std_pool)  # Output size (batch_size, 256)
        output = self.cross_att_batch_norm(output)  # Output size (batch_size, 256)
        output = self.cross_att_activation(output)  # Output size (batch_size, 256)
        return output    


class DoubleFusion(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(DoubleFusion, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # we unite separately a+t and v+t features.
        # The supportive features in both cases are t features.
        self.block_a_t = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_v_t = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)

        # finally, we do cross attention between a_t and v_t
        self.block_a_t_v_t = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_v_t_a_t = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        
        # at the end of the cross attention we want to calculate 1D avg pooling and 1D max pooling to 'aggregate'
        # the temporal information
        # at the time of the pooling, embeddings from all different cross-attention blocks will be concatenated
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.cross_att_dense_layer = torch.nn.Linear(self.in_features * 2 * 2, self.out_features)
        self.cross_att_batch_norm = torch.nn.BatchNorm1d(self.out_features)
        self.cross_att_activation = torch.nn.Tanh()

    def forward(self, x):
        a, v, t = x
        # a, v - main features, t - supportive feature
        a_t = self.block_a_t(query=t, key=a, value=a)
        v_t = self.block_v_t(query=t, key=v, value=v)
        
        a_t_v_t = self.block_a_t_v_t(query=a_t, key=v_t, value=v_t)
        v_t_a_t = self.block_v_t_a_t(query=v_t, key=a_t, value=a_t)
        
        a_v_t = torch.cat((a_t_v_t, v_t_a_t), dim=-1)
        # permute it to (batch_size, in_features, sequence_length) for calculating 1D avg pooling and 1D max pooling
        a_v_t = a_v_t.permute(0, 2, 1)  # Output size (batch_size, in_features, sequence_length)
        # 1D avg pooling and 1D max pooling
        avg_pool = self.avg_pool(a_v_t)  # Output size (batch_size, in_features, 1)
        max_pool = self.max_pool(a_v_t)  # Output size (batch_size, in_features, 1)
        # get rid of the dimension with size 1 (last dimension)
        avg_pool = avg_pool.squeeze(-1)  # Output size (batch_size, in_features)
        max_pool = max_pool.squeeze(-1)  # Output size (batch_size, in_features)
        # concat avg_pool and max_pool
        output = torch.cat((avg_pool, max_pool), dim=-1)  # Output size (batch_size, in_features * 2)
        # dense layer + batch norm + activation
        output = self.cross_att_dense_layer(output)  # Output size (batch_size, 256)
        output = self.cross_att_batch_norm(output)  # Output size (batch_size, 256)
        output = self.cross_att_activation(output)  # Output size (batch_size, 256)
        return output
    
    
class DoubleFusionO1(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(DoubleFusionO1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # we unite separately a+t and v+t features.
        # The supportive features in both cases are t features.
        self.block_a_t= CrossModalAttention(embed_dim=self.in_features, num_heads=8, dropout=0.1)
        self.block_v_t = CrossModalAttention(embed_dim=self.in_features, num_heads=8, dropout=0.1)

        # finally, we do cross attention between a_t and v_t
        self.block_a_t_v_t = CrossModalAttention(embed_dim=self.in_features, num_heads=8, dropout=0.1)
        self.block_v_t_a_t = CrossModalAttention(embed_dim=self.in_features, num_heads=8, dropout=0.1)
        
        # at the end of the cross attention we want to calculate 1D avg pooling and 1D max pooling to 'aggregate'
        # the temporal information
        # at the time of the pooling, embeddings from all different cross-attention blocks will be concatenated
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.cross_att_dense_layer = torch.nn.Linear(self.in_features * 2 * 2, self.out_features)
        self.cross_att_batch_norm = torch.nn.BatchNorm1d(self.out_features)
        self.cross_att_activation = torch.nn.Tanh()

    def forward(self, x):
        a, v, t = x
        # a, v - main features, t - supportive feature
        a_t = self.block_a_t(query=t, key=a, value=a)
        v_t = self.block_v_t(query=t, key=v, value=v)
        
        a_t_v_t = self.block_a_t_v_t(query=a_t, key=v_t, value=v_t)
        v_t_a_t = self.block_v_t_a_t(query=v_t, key=a_t, value=a_t)
        
        a_v_t = torch.cat((a_t_v_t, v_t_a_t), dim=-1)
        # permute it to (batch_size, in_features, sequence_length) for calculating 1D avg pooling and 1D max pooling
        a_v_t = a_v_t.permute(0, 2, 1)  # Output size (batch_size, in_features, sequence_length)
        # 1D avg pooling and 1D max pooling
        avg_pool = self.avg_pool(a_v_t)  # Output size (batch_size, in_features, 1)
        max_pool = self.max_pool(a_v_t)  # Output size (batch_size, in_features, 1)
        # get rid of the dimension with size 1 (last dimension)
        avg_pool = avg_pool.squeeze(-1)  # Output size (batch_size, in_features)
        max_pool = max_pool.squeeze(-1)  # Output size (batch_size, in_features)
        # concat avg_pool and max_pool
        output = torch.cat((avg_pool, max_pool), dim=-1)  # Output size (batch_size, in_features * 2)
        # dense layer + batch norm + activation
        output = self.cross_att_dense_layer(output)  # Output size (batch_size, 256)
        output = self.cross_att_batch_norm(output)  # Output size (batch_size, 256)
        output = self.cross_att_activation(output)  # Output size (batch_size, 256)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.input_proj(x)  # [batch_size, seq_len, embed_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embed_dim] for transformer
        x = self.transformer_encoder(x)  # [seq_len, batch_size, embed_dim]
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, embed_dim]
        x = self.dropout(x)
        return x


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # query, key, value: [batch_size, seq_len, embed_dim]
        attn_output, _ = self.attention(query, key, value)
        # Add & Norm
        out = self.layer_norm(query + self.dropout(attn_output))
        return out
