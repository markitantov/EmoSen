import torch


class ModalityDropAugmentation(torch.nn.Module):
    """Randomly dropes one of the modality zeroing out all elements
    Generates value (uniform distribution) on specified limits
        
    Args:
        limits (list[tuple[int, int]], optional): Limits of generated value. Defaults to [(0, 20), (20, 80), (80, 100)].
    """
    def __init__(self, limits: list[tuple[int, int]] = None) -> None:
        super(ModalityDropAugmentation, self).__init__()
        self.limits = limits if limits else [(0, 16), (16, 33), (33, 50), (50, 66), (66, 83), (83, 100)]
        self.min_l = self.limits[0][0]
        self.max_l = self.limits[-1][1] + self.limits[0][1] // 2
    
    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Generates value (uniform distribution) on specified limits
        and drop (zeroing out) modalities

        Args:
            x (list[torch.Tensor]): Input (audio, video) tensor

        Returns:
            list[torch.Tensor]: Modified (audio, video) tensor
        """
        a, v, t = x
        # generate uniformly distributed value on [min_l, max_l].
        choise = torch.FloatTensor(1).uniform_(self.min_l, self.max_l)
        
        limits_idx = -1
        for l_idx, l in enumerate(self.limits):
            if l[0] <= choise < l[1]:
                limits_idx = l_idx
            else:
                continue
            
        match limits_idx:
            case 0:
                v, t = torch.zeros(v.shape), torch.zeros(t.shape)
            case 1:
                a, t = torch.zeros(a.shape), torch.zeros(t.shape)
            case 2:
                a, v = torch.zeros(a.shape), torch.zeros(v.shape)
            case 3:
                t = torch.zeros(t.shape)
            case 4:
                v = torch.zeros(v.shape)
            case 5:
                a = torch.zeros(a.shape)
            case _:
                pass
        
        return a, v, t


class ModalityRemover(torch.nn.Module):
    """Remove one of the modality zeroing out all elements
        
    Args:
        Modalities
    """
    def __init__(self, used_modalities: str) -> None:
        super(ModalityRemover, self).__init__()
        self.removed_modalities = set(['A', 'V', 'T']) - set([c for c in used_modalities])
    
    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            x (list[torch.Tensor]): Input (audio, video, text) tensor

        Returns:
            list[torch.Tensor]: Modified (audio, video, text) tensor
        """
        a, v, t = x
        if 'A' in self.removed_modalities:
            a = torch.zeros(a.shape)
        
        if 'V' in self.removed_modalities:
            v = torch.zeros(v.shape)

        if 'T' in self.removed_modalities:
            t = torch.zeros(t.shape)  
        
        return a, v, t