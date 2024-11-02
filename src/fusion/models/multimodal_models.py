from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
    
    
from common.models.common import SmallClassificationHead

    
class SimpleMultimodalModel(nn.Module):
    def __init__(self, out_emo=6, out_sen=3):
        super(SimpleMultimodalModel, self).__init__()
        
        self.cl_head = SmallClassificationHead(input_size=199, 
                                               out_emo=out_emo, 
                                               out_sen=out_sen)

    def forward(self, x):
        a_features, v_features, t_features = x
        return self.cl_head(x)