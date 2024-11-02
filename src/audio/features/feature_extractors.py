import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from data.data_preprocessors import Wav2Vec2DataPreprocessor
from common.data.utils import define_context_length


from audio.features.W2V2_model import EmotionModel
from audio.features.ExHuBERT_model import ExHuBERT
from audio.features.AudioWT_model import AudioModelWT

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
    
    
class AudeeringW2V2FeatureExtractor(BaseFeatureExtractor): 
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
        
        self.model = EmotionModel.from_pretrained(model_name).to(self.device)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extracts features
        Apply padding to max length of audio

        Args:
            waveform (torch.Tensor): Input waveform

        Returns:
            torch.Tensor: Extracted features
        """
        waveform = self.processor(waveform, sampling_rate=self.sr, return_tensors="pt", truncation=True,
                                  padding='max_length', max_length=self.sr * self.win_max_length)["input_values"]
        
        waveform = waveform.to(self.device)
        with torch.no_grad():
            features = self.model.extract_features(waveform)
            
        return features.detach().cpu().squeeze()
    
    
class ExHuBERTFeatureExtractor(BaseFeatureExtractor): 
    def __init__(self, sr: int = 16000, win_max_length: int = 4) -> None:
        """ExHuBERT Feature Extractor
        https://huggingface.co/amiriparian/ExHuBERT

        Args:
            sr (int, optional): Sample rate of audio. Defaults to 16000.
            win_max_length (int, optional): Max length of window. Defaults to 4.
        """
        self.sr = sr
        self.win_max_length = win_max_length
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        model_name = 'amiriparian/ExHuBERT'
        
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        
        self.model = ExHuBERT.from_pretrained(model_name, trust_remote_code=True, revision="b158d45ed8578432468f3ab8d46cbe5974380812")
        
        self.model = self.model.to(self.device)
        
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extracts features
        Apply padding to max length of audio

        Args:
            waveform (torch.Tensor): Input waveform

        Returns:
            torch.Tensor: Extracted features
        """
        waveform = self.processor(waveform, sampling_rate=self.sr, return_tensors="pt", truncation=True,
                                  padding='max_length', max_length=self.sr * self.win_max_length)["input_values"]
        
        waveform = waveform.to(self.device)
        with torch.no_grad():
            features = self.model.extract_features(waveform)
            
        return features.detach().cpu().squeeze()
    
    
class AudioFeatureExtractor(BaseFeatureExtractor): 
    def __init__(self, sr: int = 16000, win_max_length: int = 4) -> None:
        """
        Args:
            sr (int, optional): Sample rate of audio. Defaults to 16000.
            win_max_length (int, optional): Max length of window. Defaults to 4.
        """
        self.sr = sr
        self.win_max_length = win_max_length
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        model_config = AutoConfig.from_pretrained(model_name)

        model_config.out_emo = 7
        model_config.out_sen = 3
        model_config.context_length = define_context_length(self.win_max_length)
        
        self.processor = Wav2Vec2DataPreprocessor(model_name)
        
        self.model = AudioModelWT.from_pretrained(pretrained_model_name_or_path=model_name, config=model_config)
        
        checkpoint = torch.load('/media/maxim/WesternDigital/RAMAS2024/multicorpus/w-AudioModelWT-2024.08.26-02.33.04/models/epoch_100.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extracts acoustic features
        Apply padding to max length of audio

        Args:
            waveform (torch.Tensor): Input waveform

        Returns:
            torch.Tensor: Extracted features
        """
        waveform = self.processor(waveform).unsqueeze(0).to(self.device)
    
        with torch.no_grad():        
            preds, features = self.model(waveform, with_features=True)
            
            predicts = {
                'emo': F.softmax(preds['emo'], dim=-1).detach().cpu().squeeze(), 
                'sen': F.softmax(preds['sen'], dim=-1).detach().cpu().squeeze()
            }
        
        return predicts, features.detach().cpu().squeeze()
