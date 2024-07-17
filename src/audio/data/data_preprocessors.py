import numpy as np
from transformers import AutoProcessor
import torch


class BaseDataPreprocessor:
    def __init__(self) -> None:
        """Base Data Preprocessor class
        """
        pass
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Base Data Preprocessor implementation

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Preprocessed data
        """
        return x


class Wav2VecDataPreprocessor(BaseDataPreprocessor): 
    def __init__(self, preprocessor_name: str = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim', sr: int = 16000) -> None:
        """Wav2Vec Data Preprocessor

        Args:
            preprocessor_name (str, optional): Preprocessor name in transformers library. 
                                               Defaults to 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'.
            sr (int, optional): Sample rate of audio. Defaults to 16000.
        """
        self.sr = sr
        self.processor = AutoProcessor.from_pretrained(preprocessor_name)
    
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts features for wav2vec using 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim' preprocessor 
        from transformers library

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Preprocessed data
        """
        a_data = self.processor(x, sampling_rate=self.sr)
        return a_data['input_values'][0].squeeze()