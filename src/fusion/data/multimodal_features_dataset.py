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

from configs.singlecorpus_config import data_config as conf

from common.data.utils import load_data, save_data, slice_audio, emo_to_label, ohe_sentiment, read_audio, generate_dump_filename, find_intersections
from audio.features.feature_extractors import BaseFeatureExtractor, AudioFeatureExtractor


class MultimodalFeaturesDataset(Dataset):
    def __init__(self, audio_root: str, labels_metadata: pd.DataFrame, 
                 features_root: str, features_file_name: str,
                 corpus_name: str, include_neutral: bool = False,
                 vad_metadata: dict[list] = None,
                 sr: int = 16000, win_max_length: int = 4, win_shift: int = 2, win_min_length: int = 4,
                 feature_extractor: BaseFeatureExtractor = None,
                 transform: torchvision.transforms.transforms.Compose = None) -> None:
        self.audio_root = audio_root
        self.labels_metadata = labels_metadata
        self.vad_metadata = vad_metadata
        
        self.corpus_name = corpus_name
        self.include_neutral = include_neutral
        
        self.sr = sr
        self.win_max_length = win_max_length
        self.win_shift = win_shift
        self.win_min_length = win_min_length
        
        self.transform = transform
        self.feature_extractor = feature_extractor

        partial_features_file_name = generate_dump_filename(vad_metadata=self.vad_metadata,
                                                            win_max_length=self.win_max_length, 
                                                            win_shift=self.win_shift,
                                                            win_min_length=self.win_min_length,
                                                            feature_extractor=self.feature_extractor)
                
        
        self.full_features_path = os.path.join(features_root, '{}_{}_a'.format(features_file_name, partial_features_file_name))
        full_features_file_name = os.path.join(features_root, '{}_{}_stats_a.pickle'.format(features_file_name, partial_features_file_name))

        if not os.path.exists(self.full_features_path):
            os.makedirs(self.full_features_path)
        
        self.info = load_data(full_features_file_name)
        self.v_info = load_data(full_features_file_name.replace('_a.', '_v.'))
        self.t_info = load_data(full_features_file_name.replace('_a.', '_t.'))

        if not self.info:
            self.prepare_data()
            save_data(self.info, full_features_file_name)
            
        self.info = self.join_avt(self.info, self.v_info, self.t_info)
        self.info, self.stats = self.filter_samples(self.info, self.include_neutral)

    def prepare_data(self) -> None:
        """Prepares data
        - Reads audio
        - Slices audio
        - Finds windows intersections according to VAD
        - [Optional] Extracts deep features
        - Dumps features
        - Calculates label statistics
        """
        self.info = []
        
        audio_feature_extractor = AudioFeatureExtractor()
        for sample in tqdm(self.labels_metadata.to_dict('records')):
            if 'CMUMOSEI' in self.corpus_name:
                sample_fp = os.path.join(self.audio_root, '{0}.wav'.format(sample['video_name']))
                sample_filename = '{0}_{1}_{2}.wav'.format(sample['video_name'], 
                                                           sample['start_time'], 
                                                           sample['end_time'])
            elif 'MELD' in self.corpus_name:
                sample_fp = os.path.join(self.audio_root, '{0}.wav'.format(sample['video_name']))
                sample_filename = os.path.basename(sample_fp)
            else:
                sample_fp = os.path.join(self.audio_root, sample['video_name'].replace('.mov', '.wav'))
                sample_filename = os.path.basename(sample_fp)

            sample_emo = [float(sample[emo]) for emo in ['neutral', 'happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']]
            sample_sen = sample['sentiment']
            
            full_wave = read_audio(sample_fp, self.sr)
            
            audio_windows = slice_audio(start_time=int(sample['start_time'] * self.sr) if 'CMUMOSEI' in self.corpus_name else 0, 
                                        end_time=int(sample['end_time'] * self.sr) if 'CMUMOSEI' in self.corpus_name else int(len(full_wave)),
                                        win_max_length=int(self.win_max_length * self.sr), 
                                        win_shift=int(self.win_shift * self.sr), 
                                        win_min_length=int(self.win_min_length * self.sr))
            
            if not audio_windows:
                continue               

            for w_idx, window in enumerate(audio_windows):
                wave = full_wave[window['start']: window['end']].clone()
                
                if self.feature_extractor:
                    predicts, features = audio_feature_extractor(wave)
                else:
                    predicts, features = (None, None)
                    
                data = {
                    'start': window['start'] - sample['start_time'] * self.sr if 'CMUMOSEI' in self.corpus_name else window['start'],
                    'end': window['end'] - sample['start_time'] * self.sr if 'CMUMOSEI' in self.corpus_name else window['end'],
                    'target_emo': sample_emo,
                    'target_sen': sample_sen,
                    'predict_emo': predicts['emo'] if predicts is not None else None,
                    'predict_sen': predicts['sen'] if predicts is not None else None,
                    'features': features if features is not None else wave,
                    'data_available': find_intersections(x=[window], 
                                                         y=self.vad_metadata[os.path.basename(sample_fp)], 
                                                         min_length=0) if self.vad_metadata else [window]
                }
            
                self.info.append({
                    'fp': sample_fp,
                    'fn': sample_filename,
                    'w_idx': w_idx,
                    'start': window['start'] - sample['start_time'] * self.sr if 'CMUMOSEI' in self.corpus_name else window['start'],
                    'end': window['end'] - sample['start_time'] * self.sr if 'CMUMOSEI' in self.corpus_name else window['end'],
                    't_emo': sample_emo,
                    't_sen': sample_sen,
                    'p_emo': predicts['emo'] if predicts is not None else None,
                    'p_sen': predicts['sen'] if predicts is not None else None,
                })

                save_data(data, os.path.join(self.full_features_path, 
                                             sample_filename.replace('.wav', '_{0}.dat'.format(w_idx))))

    def join_avt(self, a_info: list[dict], v_info: list[dict], t_info: list[dict]) -> list[dict]:
        all_samples = {}
        for idx, sample in enumerate(a_info):
            sample_name = sample['fn'].replace('.wav', '_{0}.dat'.format(sample['w_idx']))
            all_samples[sample_name] = sample

        for idx, sample in enumerate(v_info):
            sample_name = sample['fn'].replace('.wav', '_{0}.dat'.format(sample['w_idx']))
            if sample_name in all_samples:
                continue

            all_samples[sample_name] = sample

        for idx, sample in enumerate(t_info):
            sample_name = sample['fn'].replace('.wav', '_{0}.dat'.format(sample['w_idx']))
            if sample_name in all_samples:
                continue
            
            all_samples[sample_name] = sample

        return list(all_samples.values())
                
    def filter_samples(self, info: list[dict], include_neutral: int) -> tuple[list[dict], dict]:
        """Filters samples and calculate db statistics after filtering

        Args:
            info (list[dict]): List of samples with all information
            include_neutral (int): Include neutral emotion or not.

        Returns:
            tuple[list[dict], dict]: Filtered info dictionary and statistics for all samples
        """
        stats = {
            'db': self.corpus_name,
            'fns': {},
            'majority_class': {
                'emo_7': 0,
                'emo_6': 0,
                'sen_3': 0
            },
            'counts': {
                'emo_7': np.zeros(7, dtype=float),
                'emo_6': np.zeros(6, dtype=float),
                'sen_3': np.zeros(3, dtype=int),
            },
        }
        
        new_info = []
        for sample_info in info:                        
            sample_emo = emo_to_label(sample_info['t_emo'], include_neutral)
            sample_sen = ohe_sentiment(sample_info['t_sen'])
           
            new_info.append({
                'fp': sample_info['fp'],
                'fn': sample_info['fn'],
                'w_idx': sample_info['w_idx'],
                'start': sample_info['start'],
                'end': sample_info['end'],
                't_emo': sample_emo,
                't_sen': sample_sen,
                'p_emo': sample_info['p_emo'],
                'p_sen': sample_info['p_sen'],
            })
            
            stats['fns'][sample_info['fn']] = {
                'emo_7': np.asarray(emo_to_label(sample_info['t_emo'], True)),
                'emo_6': np.asarray(emo_to_label(sample_info['t_emo'], False)),
                'sen_3': np.asarray(ohe_sentiment(sample_info['t_sen']))
            }                  
            
            stats['counts']['emo_7'] += np.asarray(emo_to_label(sample_info['t_emo'], True))
            stats['counts']['emo_6'] += np.asarray(emo_to_label(sample_info['t_emo'], False))
            stats['counts']['sen_3'] += np.asarray(ohe_sentiment(sample_info['t_sen']))
            
        stats['majority_class']['emo_7'] = int(np.argmax(stats['counts']['emo_7']))
        stats['majority_class']['emo_6'] = int(np.argmax(stats['counts']['emo_6']))
        stats['majority_class']['sen_3'] = int(np.argmax(stats['counts']['sen_3']))        
        return new_info, stats

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
        data = self.info[index]
                    
        if os.path.exists(os.path.join(self.full_features_path, data['fn'].replace('.wav', '_{0}.dat'.format(data['w_idx'])))):
            a_data = load_data(os.path.join(self.full_features_path, data['fn'].replace('.wav', '_{0}.dat'.format(data['w_idx']))))
        else:
            a_data = torch.zeros(199, 1024)

        if os.path.exists(os.path.join(self.full_features_path.replace('_a', '_v'), data['fn'].replace('.wav', '_{0}.dat'.format(data['w_idx'])))):
            v_data = torch.FloatTensor(load_data(os.path.join(self.full_features_path.replace('_a', '_v'), data['fn'].replace('.wav', '_{0}.dat'.format(data['w_idx'])))))
        else:
            v_data = torch.zeros(20, 512)

        if os.path.exists(os.path.join(self.full_features_path.replace('_a', '_t'), data['fn'].replace('.wav', '_{0}.dat'.format(data['w_idx'])))):
            t_data = torch.FloatTensor(load_data(os.path.join(self.full_features_path.replace('_a', '_t'), data['fn'].replace('.wav', '_{0}.dat'.format(data['w_idx'])))))
        else:
            t_data = torch.zeros(20, 512) # TODO

        if self.transform:
            a_data, v_data, t_data = self.transform(a_data, v_data, t_data)

        # OHE
        emo_values = data['t_emo']
        sen_values = data['t_sen']

        sample_info = {
            'filename': data['fn'],
            'start_t': data['start'] / self.sr,
            'end_t': data['end'] / self.sr,
            'start_f': data['start'],
            'end_f': data['end'],
            'db': self.corpus_name,
        }

        y = {'emo': torch.FloatTensor(emo_values), 'sen': torch.FloatTensor(sen_values)}
        return [torch.FloatTensor(a_data), torch.FloatTensor(v_data), torch.FloatTensor(t_data)], y, [sample_info]

    def __len__(self) -> int:
        """Returns number of all samples in dataset

        Returns:
            int: Length of info list
        """
        return len(self.info)
