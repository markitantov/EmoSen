import torch
from transformers import Wav2Vec2Processor

from audio.features.audio_models_features import EmotionModelV1


class BaseFeatureExtractor:
    def __init__(self) -> None:
        """Base Feature Extractor class
        """
        pass
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Base Feature Extractor implementation

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Features
        """
        return x
    
    
class AudeeringFeatureExtractor(BaseFeatureExtractor): 
    def __init__(self, sr: int = 16000, win_max_length: int = 4) -> None:
        """Audeering Feature Extractor
        https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim

        Args:
            sr (int, optional): Sample rate of audio. Defaults to 16000.
            win_max_length (int, optional): Max length of window. Defaults to 4.
        """
        self.sr = sr
        self.win_max_length = win_max_length
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        self.model = EmotionModelV1.from_pretrained(model_name).to(self.device)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extracts features
        Apply padding to max length of audio

        Args:
            waveform (torch.Tensor): Input waveform

        Returns:
            torch.Tensor: Extracted features
        """
        waveform = self.processor(waveform, sampling_rate=self.sr, return_tensors="pt", 
                                  padding='max_length', max_length=self.sr * self.win_max_length)["input_values"]
        
        waveform = waveform.to(self.device)
        with torch.no_grad():
            features = self.model.extract_features(waveform)
            
        return features.detach().cpu().squeeze()
