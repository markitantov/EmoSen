import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from soft.models.models import AudioModelWT

class AudioFeatureExtractor: 
    def __init__(self, 
                 checkpoint_path: str,
                 device: torch.device, sr: int = 16000, win_max_length: int = 4, 
                 with_features: bool = False) -> None:
        """
        Args:
            sr (int, optional): Sample rate of audio. Defaults to 16000.
            win_max_length (int, optional): Max length of window. Defaults to 4.
            with_features (bool, optional): Extract features or not
        """
        self.device = device
        self.sr = sr
        self.win_max_length = win_max_length
        self.with_features = with_features
        
        model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        model_config = AutoConfig.from_pretrained(model_name)

        model_config.out_emo = 7
        model_config.out_sen = 3
        model_config.context_length = 199
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        
        self.model = AudioModelWT.from_pretrained(pretrained_model_name_or_path=model_name, config=model_config)
        
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

    def preprocess_wave(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts features for wav2vec  
        Apply padding to max length of audio

        Args:
            x (torch.Tensor): Input data

        Returns:
            np.ndarray: Preprocessed data
        """
        a_data = self.processor(x, sampling_rate=self.sr, return_tensors="pt", 
                                padding='max_length', max_length=self.sr * self.win_max_length)
        return a_data["input_values"][0]
        
    def __call__(self, waveform: torch.Tensor) -> tuple[dict[torch.Tensor], torch.Tensor]:
        """Extracts acoustic features
        Apply padding to max length of audio

        Args:
            wave (torch.Tensor): wave

        Returns:
            torch.Tensor: Extracted features
        """ 
        waveform = self.preprocess_wave(waveform).unsqueeze(0).to(self.device)
    
        with torch.no_grad():
            if self.with_features:
                preds, features = self.model(waveform, with_features=self.with_features)
            else:
                preds = self.model(waveform, with_features=self.with_features)

            predicts = {
                'emo': F.softmax(preds['emo'], dim=-1).detach().cpu().squeeze(), 
                'sen': F.softmax(preds['sen'], dim=-1).detach().cpu().squeeze()
            }

        return (predicts, features.detach().cpu().squeeze()) if self.with_features else (predicts, None)
