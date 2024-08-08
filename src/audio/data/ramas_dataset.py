import sys

sys.path.append('src')

import os
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torchvision

from torch.utils.data import Dataset

from audio.configs.singlecorpus_config import data_config as conf

from audio.data.common import load_data, save_data, slice_audio, find_intersections, emo_to_label, ohe_sentiment, read_audio, generate_dump_filename
from audio.data.data_preprocessors import BaseDataPreprocessor
from audio.features.feature_extractors import BaseFeatureExtractor


class RAMASDataset(Dataset):
    def __init__(self, audio_root: str, metadata: pd.DataFrame, dump_filepath: str, 
                 vad_metadata: dict[list] = None, include_neutral: bool = True, 
                 sr: int = 16000, win_max_length: int = 4, win_shift: int = 2, win_min_length: int = 0, 
                 feature_extractor: BaseFeatureExtractor = None,
                 transform: torchvision.transforms.transforms.Compose = None,
                 data_preprocessor: BaseDataPreprocessor = None) -> None:
        """RAMAS dataset
        Preprocesses labels and features during initialization

        Args:
            audio_root (str): Audio root dir
            metadata (pd.DataFrame): Pandas labels
            dump_filepath (str): File path for fast load features
            vad_metadata (dict[list], optional): VAD information. Defaults to None.
            include_neutral (bool, optional): Include neutral emotion or not. Defaults to True.
            sr (int, optional): Sample rate of audio. Defaults to 16000.
            win_max_length (int, optional): Max length of window. Defaults to 4.
            win_shift (int, optional): Shift length of window. Defaults to 2.
            win_min_length (int, optional): Min length of window. Defaults to 0.
            feature_extractor (BaseFeatureExtractor, optional): Feature extractor. Defaults to None.
            transform (torchvision.transforms.transforms.Compose, optional): Augmentation methods. Defaults to None.
            data_preprocessor (BaseDataProcessor, optional): Data preprocessor. Defaults to None.
        """
        self.audio_root = audio_root
        self.metadata = metadata
        self.vad_metadata = vad_metadata
        self.include_neutral = include_neutral
        
        self.sr = sr
        self.win_max_length = win_max_length
        self.win_shift = win_shift
        self.win_min_length = win_min_length
        
        self.transform = transform
        self.feature_extractor = feature_extractor
        self.data_preprocessor = data_preprocessor

        partial_dump_filename = generate_dump_filename(vad_metadata=self.vad_metadata,  
                                                       win_max_length=self.win_max_length, 
                                                       win_shift=self.win_shift,
                                                       win_min_length=self.win_min_length, 
                                                       feature_extractor=self.feature_extractor)

        self.full_dump_path = os.path.join(os.path.dirname(self.audio_root), 'features',
                                           '{}_{}'.format(dump_filepath, partial_dump_filename))
        full_dump_filename = os.path.join(self.full_dump_path, 'stats.pickle')

        if not os.path.exists(self.full_dump_path):
            os.makedirs(self.full_dump_path)
        
        self.info = load_data(full_dump_filename)

        if not self.info:
            self.prepare_data()
            save_data(self.info, full_dump_filename)

    def prepare_data(self) -> None:
        """Prepares data
        - Reads audio
        - Slices audio
        - Finds windows intersections according to VAD
        - Dumps features
        - Calculates label statistics
        """
        self.info = {
            'stats': {
                'db': 'RAMAS',
                'fns': {},
                'majority_class': {
                    'emo_7': 0,
                    'emo_6': 0,
                    'sen_3': 0
                },
                'counts': {
                    'emo_7': [],
                    'emo_6': [],
                    'sen_3': [],
                },
            },
            'samples': [],
        }

        for sample in tqdm(self.metadata.values):
            sample_fp = os.path.join(self.audio_root, sample[1].replace('.mov', '.wav'))

            sample_emo = sample[6:13].astype(int)
            sample_sen = sample[5]
            
            full_wave = read_audio(sample_fp, self.sr)

            audio_windows = slice_audio(start_time=0, end_time=int(len(full_wave)),
                                        win_max_length=int(self.win_max_length * self.sr), win_shift=int(self.win_shift * self.sr), win_min_length=int(self.win_min_length * self.sr))
                
            self.info['stats']['fns'][os.path.basename(sample_fp)] = {
                'emo_7': emo_to_label(sample_emo, True),
                'emo_6': emo_to_label(sample_emo, False),
                'sen_3': ohe_sentiment(sample_sen)
            }
            
            if not audio_windows:
                continue
            
            if self.vad_metadata:
                vad_info = self.vad_metadata[os.path.basename(sample_fp)]
                intersections = find_intersections(audio_windows, vad_info)
            else:
                intersections = audio_windows

            for w_idx, window in enumerate(intersections):
                wave = full_wave[window['start']: window['end']].clone()
                
                if self.feature_extractor:
                    wave = self.feature_extractor(wave)
                
                self.info['samples'].append({
                    'fp': sample_fp,
                    'w_idx': w_idx,
                    'start': window['start'],
                    'end': window['end'],
                    'emo': sample_emo,
                    'sen': sample_sen,
                })
                
                save_data(wave, os.path.join(self.full_dump_path, 
                                             os.path.basename(sample_fp).replace('.wav', 
                                                                                 '_{0}.dat'.format(w_idx))))
                
        emo_7 = self.metadata[['neutral', 'happy', 'sad', 'anger','surprise', 'disgust', 'fear']].values
        self.info['stats']['counts']['emo_7'] = np.unique(np.argmax(emo_7, axis=1), return_counts=True)[1]
        self.info['stats']['majority_class']['emo_7'] = int(np.argmax(self.info['stats']['counts']['emo_7']))
        
        emo_6 = self.metadata[['happy', 'sad', 'anger','surprise', 'disgust', 'fear']].values
        self.info['stats']['counts']['emo_6'] = np.unique(np.argmax(emo_6, axis=1), return_counts=True)[1]
        self.info['stats']['majority_class']['emo_6'] = int(np.argmax(self.info['stats']['counts']['emo_6']))
        
        sen_3 = self.metadata[['sentiment']].values
        self.info['stats']['counts']['sen_3'] = np.unique(sen_3, return_counts=True)[1]
        self.info['stats']['majority_class']['sen_3'] = int(np.argmax(self.info['stats']['counts']['sen_3']))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[torch.Tensor], dict]:
        """Gets sample from dataset:
        - Pads the obtained values to `win_max_length` seconds
        - Augments the obtained window
        - Extracts preliminary deep features if `processor` is set

        Args:
            index (int): Index of sample from info list

        Returns:
            tuple[torch.Tensor, dict[torch.Tensor], dict]: x, Y[emo, sen], sample_info
        """
        data = self.info['samples'][index]
        a_data = load_data(os.path.join(self.full_dump_path, 
                                        os.path.basename(data['fp']).replace('.wav', 
                                                                             '_{0}.dat'.format(data['w_idx']))))

        if self.transform:
            a_data = self.transform(a_data)

        if self.data_preprocessor:
            a_data = self.data_preprocessor(a_data)
        else:
            a_data = torch.nn.functional.pad(a_data, (0, max(0, int(self.win_max_length * self.sr) - len(a_data))), mode='constant')

        # OHE
        emo_values = emo_to_label(data['emo'], self.include_neutral)
        sen_values = ohe_sentiment(data['sen'])

        sample_info = {
            'filename': os.path.basename(data['fp']),
            'start_t': data['start'] / self.sr,
            'end_t': data['end'] / self.sr,
            'start_f': data['start'],
            'end_f': data['end'],
            'db': self.info['stats']['db'],
        }

        y = {'emo': torch.FloatTensor(emo_values), 'sen': torch.FloatTensor(sen_values)}
        return torch.FloatTensor(a_data), y, [sample_info]

    def __len__(self) -> int:
        """Return number of all samples in dataset

        Returns:
            int: Length of info list
        """
        return len(self.info['samples'])


if __name__ == "__main__":
    data_config = conf['RAMAS']

    for ds in ['train', 'test']:
        labels = pd.read_csv(os.path.join(data_config['DATA_ROOT'], data_config['LABELS_FILE']))
        labels = labels[labels['subset'] == ds]
        with open(os.path.join(data_config['DATA_ROOT'], data_config['VAD_FILE']), 'rb') as handle:
            vad_metadata = pickle.load(handle)

        dump_filepath = os.path.join(data_config['DATA_ROOT'], 'RAMAS_{}_{}'.format(ds.upper(), data_config['FEATURES_DUMP_FILE']))

        rd = RAMASDataset(audio_root=os.path.join(data_config['DATA_ROOT'], data_config['VOCALS_ROOT']),
                          metadata=labels, dump_filepath=dump_filepath,
                          vad_metadata=vad_metadata)

        dl = torch.utils.data.DataLoader(rd, batch_size=8, shuffle=False, num_workers=8)

        for d in dl:
            pass
            
        print('{0} is OK'.format(ds))