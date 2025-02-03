import torch
import torch.nn.functional as F
import torch.nn as nn


class MLMTLoss(nn.Module):
    def __init__(self, 
                 emotion_weights: torch.Tensor, sentiment_weights: torch.Tensor, 
                 emotion_alpha: float = 1, sentiment_alpha: float = 1) -> None:
        """Multilabel Multitask loss function
    
        Args:
            emotion_weights (torch.Tensor): Weights for emotion
            sentiment_weights (torch.Tensor): Weights for sentiment
            emotion_alpha (float, optional): Weighted coefficient for emotion. Defaults to 1.
            sentiment_alpha (float, optional): Weighted coefficient for sentiment. Defaults to 1.
        """
        super(MLMTLoss, self).__init__()
        self.emotion_alpha = emotion_alpha
        self.sentiment_alpha = sentiment_alpha
        
        self.emotion_loss = torch.nn.BCEWithLogitsLoss(weight=emotion_weights)
        self.sentiment_loss = torch.nn.CrossEntropyLoss(weight=sentiment_weights)
        
        self.loss_values = {
            'emo': 0,
            'sen': 0,
        }

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes MLMTLoss, which is sum of BCEWithLogitsLoss and CrossEntropyLoss

        Args:
            predicts (torch.Tensor): Input tensor
            targets (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: loss value
        """
        self.loss_values['emo'] = self.emotion_loss(predicts['emo'], targets['emo'])
        self.loss_values['sen'] = self.sentiment_loss(predicts['sen'], targets['sen'])
        return self.emotion_alpha * self.loss_values['emo'] + self.sentiment_alpha * self.loss_values['sen']
    
    
class MTLoss(nn.Module):
    def __init__(self, 
                 emotion_weights: torch.Tensor, sentiment_weights: torch.Tensor, 
                 emotion_alpha: float = 1, sentiment_alpha: float = 1) -> None:
        """Multitask loss function
    
        Args:
            emotion_weights (torch.Tensor): Weights for emotion
            sentiment_weights (torch.Tensor): Weights for sentiment
            emotion_alpha (float, optional): Weighted coefficient for emotion. Defaults to 1.
            sentiment_alpha (float, optional): Weighted coefficient for sentiment. Defaults to 1.
        """
        super(MTLoss, self).__init__()
        self.emotion_alpha = emotion_alpha
        self.sentiment_alpha = sentiment_alpha
        
        self.emotion_loss = torch.nn.CrossEntropyLoss(weight=emotion_weights)             
        self.sentiment_loss = torch.nn.CrossEntropyLoss(weight=sentiment_weights)
        
        self.loss_values = {
            'emo': 0,
            'sen': 0,
        }

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes MTLoss, which is sum of two CrossEntropyLoss

        Args:
            predicts (torch.Tensor): Input tensor
            targets (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: loss value
        """
        self.loss_values['emo'] = self.emotion_loss(predicts['emo'], targets['emo'])
        self.loss_values['sen'] = self.sentiment_loss(predicts['sen'], targets['sen'])
        return self.emotion_alpha * self.loss_values['emo'] + self.sentiment_alpha * self.loss_values['sen']
    

class MTEmoLoss(nn.Module):
    def __init__(self, 
                 emotion_weights: torch.Tensor, sentiment_weights: torch.Tensor, 
                 emotion_alpha: float = 1, sentiment_alpha: float = 1) -> None:
        """Multitask loss function
    
        Args:
            emotion_weights (torch.Tensor): Weights for emotion
            sentiment_weights (torch.Tensor): Weights for sentiment
            emotion_alpha (float, optional): Weighted coefficient for emotion. Defaults to 1.
            sentiment_alpha (float, optional): Weighted coefficient for sentiment. Defaults to 1.
        """
        super(MTEmoLoss, self).__init__()
        self.emotion_alpha = emotion_alpha
        self.sentiment_alpha = sentiment_alpha
        
        self.emotion_loss = torch.nn.CrossEntropyLoss(weight=emotion_weights)             
        self.sentiment_loss = torch.nn.CrossEntropyLoss(weight=sentiment_weights)
        
        self.loss_values = {
            'emo': 0,
            'sen': 0,
        }

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Computes MTLoss, which is sum of two CrossEntropyLoss

        Args:
            predicts (torch.Tensor): Input tensor
            targets (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: loss value
        """
        self.loss_values['emo'] = self.emotion_loss(predicts['emo'], targets['emo'])
        self.loss_values['sen'] = self.sentiment_loss(predicts['sen'], targets['sen'])
        return self.emotion_alpha * self.loss_values['emo']