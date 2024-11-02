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
from torchvision import transforms
from transformers import AutoConfig

from configs.multicorpus_config import data_config as dconf
from configs.multicorpus_config import training_config as tconf

from audio.augmentation.wave_augmentation import RandomChoice, PolarityInversion, WhiteNoise, Gain

from audio.data.ramas_dataset import RAMASDataset
from audio.data.meld_dataset import MELDDataset
from audio.data.cmumosei_dataset import CMUMOSEIDataset
from common.data.grouping import singlecorpus_grouping
from common.data.utils import define_context_length

from audio.features.feature_extractors import *
from audio.data.data_preprocessors import *

from audio.models.audio_2023_model import *
from audio.models.audio_2024_model import *

from common.loss.loss import MTLoss

from common.utils.accuracy import *

from common.net_trainer.net_trainer import NetTrainer, LabelType

from common.utils.common import get_source_code, define_seed, AttrDict
  

def main(d_config: dict, t_config: dict) -> None:
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
    
    features_dump_file = t_config['FEATURE_EXTRACTOR']['FEATURES_DUMP_FILE']
    win_max_length = t_config['FEATURE_EXTRACTOR']['WIN_MAX_LENGTH']
    win_shift = t_config['FEATURE_EXTRACTOR']['WIN_SHIFT']
    win_min_length = t_config['FEATURE_EXTRACTOR']['WIN_MIN_LENGTH']
    sr = t_config['FEATURE_EXTRACTOR']['SR']
    
    feature_extractor_cls = t_config['FEATURE_EXTRACTOR']['cls']
    feature_extractor_args = t_config['FEATURE_EXTRACTOR']['args']
    
    data_preprocessor_cls = t_config['DATA_PREPROCESSOR']['cls']
    data_preprocessor_args = t_config['DATA_PREPROCESSOR']['args']
    
    model_cls = t_config['MODEL']['cls']
    model_args = t_config['MODEL']['args']
    
    num_epochs = t_config['NUM_EPOCHS']
    batch_size = t_config['BATCH_SIZE']
    
    source_code = 'Data configuration:\n{0}\nTraining configuration:\n{1}\n\nSource code:\n{2}'.format(
        pprint.pformat(d_config),
        pprint.pformat(t_config),
        get_source_code([main, model_cls, RAMASDataset, MELDDataset, CMUMOSEIDataset, feature_extractor_cls, data_preprocessor_cls, NetTrainer]))
    
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
            if aug:
                all_transforms[ds] = [
                    transforms.Compose([
                        RandomChoice([PolarityInversion(), WhiteNoise(), Gain()]),
                    ]),
                ]
            else:
                all_transforms[ds] = [
                    None
                ]
        else:
            all_transforms[ds] = None
        
    # Defining feature extractor and data preprocessor
    feature_extractor = feature_extractor_cls(**feature_extractor_args)
    data_preprocessor = data_preprocessor_cls(**data_preprocessor_args)
    
    # Defining metadata and data augmentations
    metadata_info = {'RAMAS': {}, 'MELD': {}, 'CMUMOSEI': {}}
    for corpus in ds_names:
        corpus_labels = pd.read_csv(os.path.join(d_config[corpus]['DATA_ROOT'], d_config[corpus]['LABELS_FILE']))

        for ds in ds_names[corpus]:
            audio_root = os.path.join(d_config[corpus]['DATA_ROOT'], d_config[corpus]['VOCALS_ROOT'])
            vad_metadata_filename = d_config[corpus]['VAD_FILE'].replace('.pickle', '_{0}.pickle'.format(ds_names[corpus][ds])) if 'MELD' in corpus else d_config[corpus]['VAD_FILE']

            with open(os.path.join(d_config[corpus]['DATA_ROOT'], vad_metadata_filename), 'rb') as handle:
                vad_metadata = pickle.load(handle)
            
            metadata_info[corpus][ds] = {
                'audio_root': os.path.join(audio_root, ds_names[corpus][ds]) if 'MELD' in corpus else audio_root,
                'labels': corpus_labels[corpus_labels['subset'] == ds_names[corpus][ds]],
                'dump_filepath': '{0}_{1}_{2}'.format(corpus, ds_names[corpus][ds].upper(), features_dump_file),
                'vad_metadata': vad_metadata
            }
    
    # Define RAMAS, MELD, CMUMOSEI datasets
    datasets_stats = {'RAMAS': {}, 'MELD': {}, 'CMUMOSEI': {}}
    datasets = {'RAMAS': {}, 'MELD': {}, 'CMUMOSEI': {}}
    datasets_classes = {'RAMAS': RAMASDataset, 'MELD': MELDDataset, 'CMUMOSEI': CMUMOSEIDataset}
    for corpus in ds_names:
        for ds in ds_names[corpus]:
            if 'train' in ds:
                datasets[corpus][ds] = torch.utils.data.ConcatDataset([
                    datasets_classes[corpus](audio_root=metadata_info[corpus][ds]['audio_root'],
                                             metadata=metadata_info[corpus][ds]['labels'], 
                                             dump_filepath=metadata_info[corpus][ds]['dump_filepath'],
                                             vad_metadata=metadata_info[corpus][ds]['vad_metadata'],
                                             include_neutral=include_neutral,
                                             sr=sr, win_max_length=win_max_length, win_shift=win_shift, win_min_length=win_min_length,
                                             feature_extractor=feature_extractor,
                                             transform=t, 
                                             data_preprocessor=data_preprocessor) for t in all_transforms[ds]
                ])

                datasets_stats[corpus][ds] = datasets[corpus][ds].datasets[0].info['stats']
            else:
                datasets[corpus][ds] = datasets_classes[corpus](audio_root=metadata_info[corpus][ds]['audio_root'],
                                                                metadata=metadata_info[corpus][ds]['labels'], 
                                                                dump_filepath=metadata_info[corpus][ds]['dump_filepath'],
                                                                vad_metadata=metadata_info[corpus][ds]['vad_metadata'], 
                                                                include_neutral=include_neutral,
                                                                sr=sr, win_max_length=win_max_length, win_shift=win_shift, win_min_length=win_min_length,
                                                                feature_extractor=feature_extractor,
                                                                transform=all_transforms[ds], 
                                                                data_preprocessor=data_preprocessor)

                datasets_stats[corpus][ds] = datasets[corpus][ds].info['stats']

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
    
    experiment_name = 'w{0}{1}-{2}'.format('a-' if aug else '-',
                                           model_cls.__name__.replace('-', '_').replace('/', '_'),
                                           datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
    
    # Defining NetTrainer 
    net_trainer = NetTrainer(log_root=logs_root,
                             experiment_name=experiment_name,
                             c_names=c_names,
                             measures=measures,
                             device=device,
                             label_type=LabelType.SINGLELABEL,                            
                             group_predicts_fn=singlecorpus_grouping,
                             source_code=source_code,
                             c_names_to_display=c_names_to_display)
    
    # Defining model
    model = model_cls.from_pretrained(**model_args)
    model.to(device)
    
    # Defining weighted loss
    class_sample_count_emo = [sum(x) for x in zip(*[datasets_stats[corpus]['train']['counts']['emo_7'] for corpus in ds_names])]
    class_sample_count_sen = [sum(x) for x in zip(*[datasets_stats[corpus]['train']['counts']['sen_3'] for corpus in ds_names])]
    loss = MTLoss(emotion_weights=torch.Tensor(class_sample_count_emo / sum(class_sample_count_emo)).to(device), emotion_alpha=1,
                  sentiment_weights=torch.Tensor(class_sample_count_sen / sum(class_sample_count_sen)).to(device), sentiment_alpha=1)
    
    # Defining optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)

    # Defining scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=10, T_mult=2)

    model, max_perf = net_trainer.run(model=model, loss=loss, optimizer=optimizer, scheduler=scheduler,
                                      num_epochs=num_epochs, dataloaders=dataloaders, datasets_stats=datasets_stats)

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

    m_clses = [
        AudioModelWT,
    ]

    logs_dir = {
        'OM': '/media/maxim/WesternDigital/RAMAS2024/multicorpus',
    }
    
    fe_clses = [BaseFeatureExtractor]
    win_params = [
        {'WIN_MAX_LENGTH': 4, 'WIN_SHIFT': 2, 'WIN_MIN_LENGTH': 2}
    ]
    
    for win_param in win_params:
        for fe_cls in fe_clses:            
            for m_cls in m_clses:
                t_config = deepcopy(tconf)
                t_config['LOGS_ROOT'] = logs_dir['OM']
                t_config['AUGMENTATION'] = False
                
                t_config['FEATURE_EXTRACTOR']['WIN_MAX_LENGTH'] = win_param['WIN_MAX_LENGTH']
                t_config['FEATURE_EXTRACTOR']['WIN_SHIFT'] = win_param['WIN_SHIFT']
                t_config['FEATURE_EXTRACTOR']['WIN_MIN_LENGTH'] = win_param['WIN_MIN_LENGTH']
                t_config['FEATURE_EXTRACTOR']['cls'] = fe_cls
                t_config['FEATURE_EXTRACTOR']['args'] = {}
                
                t_config['DATA_PREPROCESSOR']['cls'] = Wav2Vec2DataPreprocessor
                t_config['DATA_PREPROCESSOR']['args'] = {'preprocessor_name': 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'}

                t_config['MODEL']['cls'] = m_cls
                model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
                model_config = AutoConfig.from_pretrained(model_name)

                model_config.out_emo = len(d_config['RAMAS']['C_NAMES']['emo'])
                model_config.out_sen = len(d_config['RAMAS']['C_NAMES']['sen'])
                model_config.context_length = define_context_length(win_param['WIN_MAX_LENGTH'])

                model_args = AttrDict()
                model_args.pretrained_model_name_or_path = model_name
                model_args.config = model_config
                t_config['MODEL']['args'] = model_args
                
                main(d_config=d_config, t_config=t_config)

    
if __name__ == "__main__":
    run_expression_training()