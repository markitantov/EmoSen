import sys

sys.path.append('src')

import os
import gc
import pprint
import pickle
import datetime
from copy import deepcopy

import pandas as pd

import torch
import torch.nn as nn
from torchvision import transforms

from configs.multimodal_config import data_config as dconf
from configs.multimodal_config import training_config as tconf

from audio.features.feature_extractors import AudioFeatureExtractor

from common.data.grouping import singlecorpus_grouping
from common.data.utils import define_context_length
from common.loss.loss_v2 import MTLoss
from common.utils.accuracy import *
from common.net_trainer.net_trainer_v2 import NetTrainer, LabelType
from common.utils.common import get_source_code, define_seed, AttrDict

from fusion.data.multimodal_features_dataset import MultimodalFeaturesDataset
from fusion.augmentation.modality_augmentation import ModalityRemover
from fusion.models.multimodal_models import *


def main(d_config: dict, t_config: dict, used_modalities: str) -> None:
    """Trains with configuration in the following steps:
    - Defines datasets names
    - Defines data augmentations
    - Defines data preprocessor
    - Defines datasets
    - Defines dataloaders
    - Defines measures
    - Defines NetTrainer
    - Defines model
    - Defines weighted loss, optimizer, scheduler
    - Runs NetTrainer 

    Args:
        d_config (dict): Data configuration dictionary
        t_config (dict): Training configuration dictionary
    """            
    # Defining class names
    c_names = d_config['CMUMOSEI']['C_NAMES']
    include_neutral = d_config['CMUMOSEI']['INCLUDE_NEUTRAL']
    
    logs_root = t_config['LOGS_ROOT']
    
    aug = t_config['AUGMENTATION']
    
    features_root = t_config['FEATURE_EXTRACTOR']['FEATURES_ROOT']
    features_file_name = t_config['FEATURE_EXTRACTOR']['FEATURES_FILE_NAME']
    win_max_length = t_config['FEATURE_EXTRACTOR']['WIN_MAX_LENGTH']
    win_shift = t_config['FEATURE_EXTRACTOR']['WIN_SHIFT']
    win_min_length = t_config['FEATURE_EXTRACTOR']['WIN_MIN_LENGTH']
    sr = t_config['FEATURE_EXTRACTOR']['SR']
    
    feature_extractor_cls = t_config['FEATURE_EXTRACTOR']['cls']
    feature_extractor_args = t_config['FEATURE_EXTRACTOR']['args']
        
    model_cls = t_config['MODEL']['cls']
    model_args = t_config['MODEL']['args']
    
    num_epochs = t_config['NUM_EPOCHS']
    batch_size = t_config['BATCH_SIZE']
    augmentation = t_config['AUGMENTATION']
    
    source_code = 'Data configuration:\n{0}\nTraining configuration:\n{1}\n\nSource code:\n{2}'.format(
        pprint.pformat(d_config),
        pprint.pformat(t_config),
        get_source_code([main, model_cls, MultimodalFeaturesDataset, feature_extractor_cls, NetTrainer]))
    
    # Defining datasets 
    ds_names = {
        'RAMAS': {
            'train': 'train', 
            'test': 'test',
        },
        'MELD': {
            'train': 'train', 
            'devel': 'dev', 
            'test': 'test',
        },
        'CMUMOSEI': {
            'train': 'train', 
            'devel': 'dev',
            'test': 'test',
        }
    }
    
    c_names_to_display = {}
    for task, class_names in c_names.items():
        c_names_to_display[task] = [cn.capitalize() for cn in class_names]

    all_transforms = {}
    for ds in ds_names['CMUMOSEI']:
        if 'train' in ds:
            all_transforms[ds] = [
                ModalityRemover(used_modalities=used_modalities)
            ]
        else:
            all_transforms[ds] = ModalityRemover(used_modalities=used_modalities)
        
    # Defining feature extractor
    feature_extractor = feature_extractor_cls(**feature_extractor_args)
    
    # Defining metadata and data augmentations
    metadata_info = {}
    for corpus_name in ds_names:
        metadata_info[corpus_name] = {}
        corpus_labels = pd.read_csv(os.path.join(d_config[corpus_name]['DATA_ROOT'], 
                                                 d_config[corpus_name]['LABELS_FILE']))

        for ds in ds_names[corpus_name]:
            audio_root = os.path.join(d_config[corpus_name]['DATA_ROOT'], 
                                      d_config[corpus_name]['VOCALS_ROOT'])
            
            vad_metadata_filename = d_config[corpus_name]['VAD_FILE'].replace('.pickle', '_{0}.pickle'.format(ds_names[corpus_name][ds])) if 'MELD' in corpus_name else d_config[corpus_name]['VAD_FILE']

            with open(os.path.join(d_config[corpus_name]['DATA_ROOT'], vad_metadata_filename), 'rb') as handle:
                vad_metadata = pickle.load(handle)
            
            metadata_info[corpus_name][ds] = {
                'audio_root': os.path.join(audio_root, ds_names[corpus_name][ds]) if 'MELD' in corpus_name else audio_root,
                'labels_metadata': corpus_labels[corpus_labels['subset'] == ds_names[corpus_name][ds]],
                'features_file_name': '{0}_{1}_{2}'.format(corpus_name, ds_names[corpus_name][ds].upper(), features_file_name),
                'vad_metadata': vad_metadata
            }
    
    # Define datasets
    datasets = {}
    datasets_stats = {}
    for corpus_name in ds_names:
        datasets[corpus_name] = {}
        datasets_stats[corpus_name] = {}
        
        for ds in ds_names[corpus_name]:
            if 'train' in ds:
                datasets[corpus_name][ds] = torch.utils.data.ConcatDataset([
                    MultimodalFeaturesDataset(
                        audio_root=metadata_info[corpus_name][ds]['audio_root'],
                        labels_metadata=metadata_info[corpus_name][ds]['labels_metadata'], 
                        features_root=features_root, features_file_name=metadata_info[corpus_name][ds]['features_file_name'],
                        vad_metadata=metadata_info[corpus_name][ds]['vad_metadata'],
                        corpus_name=corpus_name, include_neutral=include_neutral, load_in_ram=True,
                        sr=sr, win_max_length=win_max_length, win_shift=win_shift, win_min_length=win_min_length,
                        feature_extractor=feature_extractor,
                        transform=t) for t in all_transforms[ds]
                ])

                datasets_stats[corpus_name][ds] = datasets[corpus_name][ds].datasets[0].stats
            else:
                datasets[corpus_name][ds] = MultimodalFeaturesDataset(
                    audio_root=metadata_info[corpus_name][ds]['audio_root'],
                    labels_metadata=metadata_info[corpus_name][ds]['labels_metadata'], 
                    features_root=features_root, features_file_name=metadata_info[corpus_name][ds]['features_file_name'],
                    vad_metadata=metadata_info[corpus_name][ds]['vad_metadata'],
                    corpus_name=corpus_name, include_neutral=include_neutral, load_in_ram=True,
                    sr=sr, win_max_length=win_max_length, win_shift=win_shift, win_min_length=win_min_length,
                    feature_extractor=feature_extractor,
                    transform=all_transforms[ds])

                datasets_stats[corpus_name][ds] = datasets[corpus_name][ds].stats

    # Defining dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([datasets['RAMAS']['train'], datasets['MELD']['train'], datasets['CMUMOSEI']['train']]),
            batch_size=batch_size,
            shuffle=True
        ),
        'devel_MELD':  torch.utils.data.DataLoader(datasets['MELD']['devel'], batch_size=batch_size, shuffle=False),
        'devel_CMUMOSEI':  torch.utils.data.DataLoader(datasets['CMUMOSEI']['devel'], batch_size=batch_size, shuffle=False),
        'test_RAMAS':  torch.utils.data.DataLoader(datasets['RAMAS']['test'], batch_size=batch_size, shuffle=False),
        'test_MELD':  torch.utils.data.DataLoader(datasets['MELD']['test'], batch_size=batch_size, shuffle=False),
        'test_CMUMOSEI':  torch.utils.data.DataLoader(datasets['CMUMOSEI']['test'], batch_size=batch_size, shuffle=False),
    }
    
    # Defining dataloaders
    dataloaders = {}
    for corpus_name in ds_names:
        for ds in ds_names[corpus_name]:
            if 'train' in ds:
                if ds not in dataloaders:
                    dataloaders[ds] = []
                
                dataloaders[ds].append(datasets[corpus_name][ds])
            else:
                dataloaders['{0}_{1}'.format(ds, corpus_name)] = torch.utils.data.DataLoader(
                    datasets[corpus_name][ds],
                    batch_size=batch_size,
                    shuffle=False,
                )
    
    dataloaders['train'] = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(dataloaders['train']),
        batch_size=batch_size,
        shuffle=True
    )
    
    # Defining measures, measure with 0 index is main measure
    measures = [
        MultilabelEmoSenCombinedMeasure('emo_sen_combined'), # main measure

        MeanWeightedAccuracyMeasure('emo_mWA', protection='CMUMOSEI'),
        MeanWeightedF1Measure('emo_mWF1', protection='CMUMOSEI'),
        MeanMacroF1Measure('emo_mMacroF1', protection='CMUMOSEI'),
        MeanWARMeasure('emo_mA(WAR)', protection='CMUMOSEI'),
        WARMeasure('sen_A(WAR)', protection='CMUMOSEI'),
        UARMeasure('sen_UAR', protection='CMUMOSEI'),
        WeightedF1Measure('sen_WF1', protection='CMUMOSEI'),
        MacroF1Measure('sen_MacroF1', protection='CMUMOSEI'),

        WARMeasure('emo_A(WAR)', protection='RAMAS'),
        UARMeasure('emo_UAR', protection='RAMAS'),
        WeightedF1Measure('emo_WF1', protection='RAMAS'),
        MacroF1Measure('emo_MacroF1', protection='RAMAS'),
        WARMeasure('sen_A(WAR)', protection='RAMAS'),
        UARMeasure('sen_UAR', protection='RAMAS'),
        WeightedF1Measure('sen_WF1', protection='RAMAS'),
        MacroF1Measure('sen_MacroF1', protection='RAMAS'),

        WARMeasure('emo_A(WAR)', protection='MELD'),
        UARMeasure('emo_UAR', protection='MELD'),
        WeightedF1Measure('emo_WF1', protection='MELD'),
        MacroF1Measure('emo_MacroF1', protection='MELD'),
        WARMeasure('sen_A(WAR)', protection='MELD'),
        UARMeasure('sen_UAR', protection='MELD'),
        WeightedF1Measure('sen_WF1', protection='MELD'),
        MacroF1Measure('sen_MacroF1', protection='MELD')
    ]
    
    define_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    experiment_name = 'wMultimodal{0}{1}-{2}'.format(used_modalities,
                                                     model_cls.__name__.replace('-', '_').replace('/', '_'),
                                                     datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
    
    # Defining NetTrainer 
    net_trainer = NetTrainer(log_root=logs_root,
                             experiment_name=experiment_name,
                             c_names=c_names,
                             measures=measures,
                             device=device,
                             final_activations={'emo': nn.Softmax(dim=-1), 'sen': nn.Softmax(dim=-1)},
                             label_type=LabelType.SINGLELABEL,                    
                             group_predicts_fn=singlecorpus_grouping, # this is OK
                             source_code=source_code,
                             c_names_to_display=c_names_to_display)
    
    # Defining model
    model = model_cls(**model_args)
    model.to(device)
    
    # Defining weighted loss
    class_sample_count_emo = [sum(x) for x in zip(*[datasets_stats[corpus_name]['train']['counts']['emo_7'] for corpus_name in ds_names])]
    class_sample_count_sen = [sum(x) for x in zip(*[datasets_stats[corpus_name]['train']['counts']['sen_3'] for corpus_name in ds_names])]
    loss = MTLoss(emotion_weights=torch.Tensor(class_sample_count_emo / sum(class_sample_count_emo)).to(device), emotion_alpha=1,
                  sentiment_weights=torch.Tensor(class_sample_count_sen / sum(class_sample_count_sen)).to(device), sentiment_alpha=1)
    
    # Defining optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)

    # Defining scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=10, T_mult=2)

    model, max_perf = net_trainer.run(model=model, loss=loss, optimizer=optimizer, scheduler=scheduler,
                                      num_epochs=num_epochs, dataloaders=dataloaders, datasets_stats=datasets_stats, 
                                      log_epochs=list(range(0, num_epochs + 1)))

    for phase, perf in max_perf.items():
        if 'train' in phase:
            continue

        print()
        print(phase.capitalize())
        print('Epoch: {}, Max performance:'.format(max_perf[phase]['epoch']))
        print([metric for metric in max_perf[phase]['performance']])
        print([max_perf[phase]['performance'][metric] for metric in max_perf[phase]['performance']])
        print()
    

def run_expression_training() -> None:
    """Wrapper for training 
    """
    d_config = dconf

    m_clses = [AttentionFusionTF, LabelEncoderFusionTF]
    all_modalities = ['A', 'V', 'T', 'AV', 'VT', 'TA']
        
    for used_modalities in all_modalities:      
        for m_cls in m_clses:
            t_config = deepcopy(tconf)
            t_config['AUGMENTATION'] = False
                
            t_config['FEATURE_EXTRACTOR']['cls'] = AudioFeatureExtractor
            t_config['FEATURE_EXTRACTOR']['args'] = {}
                
            t_config['MODEL']['cls'] = m_cls
            t_config['MODEL']['args'] = {
                'out_emo': len(d_config['RAMAS']['C_NAMES']['emo']),
                'out_sen': len(d_config['RAMAS']['C_NAMES']['sen'])
            }
                            
            main(d_config=d_config, t_config=t_config, used_modalities=used_modalities)

    
if __name__ == "__main__":
    run_expression_training()