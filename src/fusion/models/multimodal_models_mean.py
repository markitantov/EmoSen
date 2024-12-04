import sys

sys.path.append('src')

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from common.models.common_layers import SmallClassificationHead, TransformerLayer, PermuteLayer
from fusion.models.common_layers import MultimodalDownsamplerS32Mean, PredictionsUpsampler, TripleFusion, DoubleFusion
    
    
class LabelEncoderFusionDFAV(torch.nn.Module):
    def __init__(self, out_emo: int = 6, out_sen: int = 3):
        super(LabelEncoderFusionDFAV, self).__init__()
        self.downsampler = MultimodalDownsamplerS32Mean(features_only=False, out_f_size=512)
        self.ap_upsampler = PredictionsUpsampler(inp_size=(out_emo + out_sen))
        self.vp_upsampler = PredictionsUpsampler(inp_size=(out_emo + out_sen))
        self.tp_upsampler = PredictionsUpsampler(inp_size=(out_emo + out_sen))

        self.in_features = 512
        self.block_avtp = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_vtap = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_tavp = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        
        self.double_fusion = DoubleFusion(in_features=self.in_features, out_features=256)
        self.classifier = SmallClassificationHead(256, out_emo=out_emo, out_sen=out_sen)

    def forward(self, x):
        # cross-attention
        a, v, t = self.downsampler(x[0:3])
        ap, vp, tp = self.ap_upsampler(x[3]), self.vp_upsampler(x[4]), self.tp_upsampler(x[5])

        av = torch.mean(torch.stack((a, v), dim=-1), dim=-1)
        vt = torch.mean(torch.stack((v, t), dim=-1), dim=-1)
        ta = torch.mean(torch.stack((t, a), dim=-1), dim=-1)
        
        av_tp = self.block_avtp(query=tp, key=av, value=av)
        vt_ap = self.block_vtap(query=ap, key=vt, value=vt)
        ta_vp = self.block_tavp(query=vp, key=ta, value=ta)

        a_v_t = self.double_fusion((av_tp, vt_ap, ta_vp))
        output = self.classifier(a_v_t)
        return output
    

class LabelEncoderFusionDFAT(torch.nn.Module):
    def __init__(self, out_emo: int = 6, out_sen: int = 3):
        super(LabelEncoderFusionDFAT, self).__init__()
        self.downsampler = MultimodalDownsamplerS32Mean(features_only=False, out_f_size=512)
        self.ap_upsampler = PredictionsUpsampler(inp_size=(out_emo + out_sen))
        self.vp_upsampler = PredictionsUpsampler(inp_size=(out_emo + out_sen))
        self.tp_upsampler = PredictionsUpsampler(inp_size=(out_emo + out_sen))

        self.in_features = 512
        self.block_avtp = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_vtap = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_tavp = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        
        self.double_fusion = DoubleFusion(in_features=self.in_features, out_features=256)
        self.classifier = SmallClassificationHead(256, out_emo=out_emo, out_sen=out_sen)

    def forward(self, x):
        # cross-attention
        a, v, t = self.downsampler(x[0:3])
        ap, vp, tp = self.ap_upsampler(x[3]), self.vp_upsampler(x[4]), self.tp_upsampler(x[5])
        
        av = torch.mean(torch.stack((a, v), dim=-1), dim=-1)
        vt = torch.mean(torch.stack((v, t), dim=-1), dim=-1)
        ta = torch.mean(torch.stack((t, a), dim=-1), dim=-1)
        
        av_tp = self.block_avtp(query=tp, key=av, value=av)
        vt_ap = self.block_vtap(query=ap, key=vt, value=vt)
        ta_vp = self.block_tavp(query=vp, key=ta, value=ta)

        a_v_t = self.double_fusion((av_tp, ta_vp, vt_ap))
        output = self.classifier(a_v_t)
        return output
    

class LabelEncoderFusionDFVT(torch.nn.Module):
    def __init__(self, out_emo: int = 6, out_sen: int = 3):
        super(LabelEncoderFusionDFVT, self).__init__()
        self.downsampler = MultimodalDownsamplerS32Mean(features_only=False, out_f_size=512)
        self.ap_upsampler = PredictionsUpsampler(inp_size=(out_emo + out_sen))
        self.vp_upsampler = PredictionsUpsampler(inp_size=(out_emo + out_sen))
        self.tp_upsampler = PredictionsUpsampler(inp_size=(out_emo + out_sen))

        self.in_features = 512
        self.block_avtp = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_vtap = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_tavp = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        
        self.double_fusion = DoubleFusion(in_features=self.in_features, out_features=256)
        self.classifier = SmallClassificationHead(256, out_emo=out_emo, out_sen=out_sen)

    def forward(self, x):
        # cross-attention
        a, v, t = self.downsampler(x[0:3])
        ap, vp, tp = self.ap_upsampler(x[3]), self.vp_upsampler(x[4]), self.tp_upsampler(x[5])
        
        av = torch.mean(torch.stack((a, v), dim=-1), dim=-1)
        vt = torch.mean(torch.stack((v, t), dim=-1), dim=-1)
        ta = torch.mean(torch.stack((t, a), dim=-1), dim=-1)
        
        av_tp = self.block_avtp(query=tp, key=av, value=av)
        vt_ap = self.block_vtap(query=ap, key=vt, value=vt)
        ta_vp = self.block_tavp(query=vp, key=ta, value=ta)

        a_v_t = self.double_fusion((vt_ap, ta_vp, av_tp))
        output = self.classifier(a_v_t)
        return output


class LabelEncoderFusionTF(torch.nn.Module):
    def __init__(self, out_emo: int = 6, out_sen: int = 3):
        super(LabelEncoderFusionTF, self).__init__()
        self.downsampler = MultimodalDownsamplerS32Mean(features_only=False, out_f_size=512)
        self.ap_upsampler = PredictionsUpsampler(inp_size=(out_emo + out_sen))
        self.vp_upsampler = PredictionsUpsampler(inp_size=(out_emo + out_sen))
        self.tp_upsampler = PredictionsUpsampler(inp_size=(out_emo + out_sen))

        self.in_features = 512
        self.block_avtp = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_vtap = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_tavp = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)

        self.triple_fusion = TripleFusion(in_features=self.in_features, out_features=256)
        self.classifier = SmallClassificationHead(256, out_emo=out_emo, out_sen=out_sen)

    def forward(self, x):
        # cross-attention
        a, v, t = self.downsampler(x[0:3])
        ap, vp, tp = self.ap_upsampler(x[3]), self.vp_upsampler(x[4]), self.tp_upsampler(x[5])
        
        av = torch.mean(torch.stack((a, v), dim=-1), dim=-1)
        vt = torch.mean(torch.stack((v, t), dim=-1), dim=-1)
        ta = torch.mean(torch.stack((t, a), dim=-1), dim=-1)
        
        av_tp = self.block_avtp(query=tp, key=av, value=av)
        vt_ap = self.block_vtap(query=ap, key=vt, value=vt)
        ta_vp = self.block_tavp(query=vp, key=ta, value=ta)
        
        a_v_t = self.triple_fusion((av_tp, vt_ap, ta_vp))
        output = self.classifier(a_v_t)
        return output


class AttentionFusionTF(torch.nn.Module):
    def __init__(self, out_emo: int = 6, out_sen: int = 3):
        super(AttentionFusionTF, self).__init__()
        self.downsampler = MultimodalDownsamplerS32Mean(features_only=False, out_f_size=512)
        self.in_features = 512

        self.triple_fusion = TripleFusion(in_features=self.in_features, out_features=256)
        self.classifier = SmallClassificationHead(256, out_emo=out_emo, out_sen=out_sen)

    def forward(self, x):
        # cross-attention
        a, v, t = self.downsampler(x[0:3])
        a_v_t = self.triple_fusion((a, v, t))
        output = self.classifier(a_v_t)
        return output
    

class AttentionFusionDFAV(torch.nn.Module):
    def __init__(self, out_emo: int = 6, out_sen: int = 3):
        super(AttentionFusionDFAV, self).__init__()
        self.downsampler = MultimodalDownsamplerS32Mean(features_only=False, out_f_size=512)
        self.in_features = 512
        
        # create cross-attention blocks
        self.block_a = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_v = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_t = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)

        self.double_fusion = DoubleFusion(in_features=self.in_features, out_features=256)
        self.classifier = SmallClassificationHead(256, out_emo=out_emo, out_sen=out_sen)

    def forward(self, x):
        a, v, t = self.downsampler(x[0:3])
        # cross-attention
        a = self.block_a(query=a, key=a, value=a)
        v = self.block_v(query=v, key=v, value=v)
        t = self.block_t(query=t, key=t, value=t)
        
        a_v_t = self.double_fusion((a, v, t))

        # last classification layer
        output = self.classifier(a_v_t)
        return output
    

class AttentionFusionDFAT(torch.nn.Module):
    def __init__(self, out_emo: int = 6, out_sen: int = 3):
        super(AttentionFusionDFAT, self).__init__()
        self.downsampler = MultimodalDownsamplerS32Mean(features_only=False, out_f_size=512)
        self.in_features = 512
        
        # create cross-attention blocks
        self.block_a = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_v = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_t = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)

        self.double_fusion = DoubleFusion(in_features=self.in_features, out_features=256)
        self.classifier = SmallClassificationHead(256, out_emo=out_emo, out_sen=out_sen)

    def forward(self, x):
        a, v, t = self.downsampler(x[0:3])
        # cross-attention
        a = self.block_a(query=a, key=a, value=a)
        v = self.block_v(query=v, key=v, value=v)
        t = self.block_t(query=t, key=t, value=t)
        
        a_v_t = self.double_fusion((a, t, v))

        # last classification layer
        output = self.classifier(a_v_t)
        return output
    

class AttentionFusionDFVT(torch.nn.Module):
    def __init__(self, out_emo: int = 6, out_sen: int = 3):
        super(AttentionFusionDFVT, self).__init__()
        self.downsampler = MultimodalDownsamplerS32Mean(features_only=False, out_f_size=512)
        self.in_features = 512
        
        # create cross-attention blocks
        self.block_a = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_v = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)
        self.block_t = TransformerLayer(input_dim=self.in_features, num_heads=8, dropout=0.1, positional_encoding=True)

        self.double_fusion = DoubleFusion(in_features=self.in_features, out_features=256)
        self.classifier = SmallClassificationHead(256, out_emo=out_emo, out_sen=out_sen)

    def forward(self, x):
        a, v, t = self.downsampler(x[0:3])
        # cross-attention
        a = self.block_a(query=a, key=a, value=a)
        v = self.block_v(query=v, key=v, value=v)
        t = self.block_t(query=t, key=t, value=t)
        
        a_v_t = self.double_fusion((v, t, a))

        # last classification layer
        output = self.classifier(a_v_t)
        return output
    
    
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')       
    batch_size, out_emo, out_sen = 16, 7, 3
    feats = []
    for feats_size in [(batch_size, 199, 1024), (batch_size, 20, 512), (batch_size, 32, 1024), 
                       (batch_size, (out_emo + out_sen)), 
                       (batch_size, (out_emo + out_sen)), 
                       (batch_size, (out_emo + out_sen))]:
        feats.append(torch.zeros(feats_size).to(device))
    
    models = [
        AttentionFusionDFAV, AttentionFusionDFAT, AttentionFusionDFVT,
        LabelEncoderFusionDFAV, LabelEncoderFusionDFAT, LabelEncoderFusionDFVT,
        AttentionFusionTF, LabelEncoderFusionTF
    ]
    
    for m in models:
        print(str(m))
        model = m(out_emo=out_emo, out_sen=out_sen)
        model.to(device)
        res = model(tuple(feats))
        for t in res:
            print(res[t].shape)