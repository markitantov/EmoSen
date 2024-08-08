import sys

sys.path.append('src')

import os
import pprint
import pickle
import datetime
from copy import deepcopy

import pandas as pd

import torch
from torchvision import transforms

from audio.configs.singlecorpus_config import data_config as dconf
from audio.configs.singlecorpus_config import training_config as tconf

from audio.augmentation.wave_augmentation import RandomChoice, PolarityInversion, WhiteNoise, Gain

from audio.data.meld_dataset import MELDDataset
from audio.data.grouping import singlecorpus_grouping
from audio.data.common import define_context_length

from audio.features.feature_extractors import *
from audio.data.data_preprocessors import *

from audio.models.audio_transformers_models import *
from audio.models.audio_xlstm_models import *
from audio.models.audio_mamba_models import *

from audio.loss.loss import MTLoss

from audio.utils.accuracy import *

from audio.net_trainer.net_trainer import NetTrainer, LabelType

from audio.utils.common import get_source_code, define_seed, AttrDict
  

def main(d_config: dict, t_config: dict) -> None:
    """Trains with configuration in the following steps:
    - Defines datasets names
    - Defines data augmentations
    - Defines feature extractor and data preprocessor
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
    data_root = d_config['DATA_ROOT']
    audio_root = d_config['VOCALS_ROOT']
    labels_file = d_config['LABELS_FILE']
    vad_file = d_config['VAD_FILE']
        
    # Defining class names
    c_names = d_config['C_NAMES']
    include_neutral = d_config['INCLUDE_NEUTRAL']
    
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
        get_source_code([main, model_cls, MELDDataset, feature_extractor_cls, data_preprocessor_cls, NetTrainer]))
    
    # Defining datasets 
    ds_names = {
        'train': 'train', 
        'devel': 'dev',
        'test': 'test',
    }
    
    c_names_to_display = {}
    for task, class_names in c_names.items():
        c_names_to_display[task] = [cn.capitalize() for cn in class_names]
    
    # Defining metadata and data augmentations
    labels = pd.read_csv(os.path.join(data_root, labels_file))
    
    metadata_info = {}
    all_transforms = {}
    for ds in ds_names:
        with open(os.path.join(data_root, vad_file.replace('.pickle', '_{0}.pickle'.format(ds_names[ds]))), 'rb') as handle:
            vad_metadata = pickle.load(handle)
        
        metadata_info[ds] = {
            'audio_root': os.path.join(data_root, audio_root, ds_names[ds]),
            'labels': labels[labels['subset'] == ds_names[ds]],
            'dump_filepath': os.path.join(data_root, 'MELD_{0}_{1}'.format(ds_names[ds].upper(), features_dump_file)),
            'vad_metadata': vad_metadata
        }

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
    
    # Defining datasets
    datasets = {}
    datasets_stats = {'MELD': {}}
    for ds in ds_names:
        if 'train' in ds:
            datasets[ds] = torch.utils.data.ConcatDataset([
                MELDDataset(audio_root=metadata_info[ds]['audio_root'],
                            metadata=metadata_info[ds]['labels'], 
                            dump_filepath=metadata_info[ds]['dump_filepath'],
                            vad_metadata=metadata_info[ds]['vad_metadata'],
                            include_neutral=include_neutral,
                            sr=sr, win_max_length=win_max_length, win_shift=win_shift, win_min_length=win_min_length,
                            feature_extractor=feature_extractor,
                            transform=t, 
                            data_preprocessor=data_preprocessor) for t in all_transforms[ds]
                ]
            )

            datasets_stats['MELD'][ds] = datasets[ds].datasets[0].info['stats']
        else:
            datasets[ds] = MELDDataset(audio_root=metadata_info[ds]['audio_root'],
                                       metadata=metadata_info[ds]['labels'], 
                                       dump_filepath=metadata_info[ds]['dump_filepath'],
                                       vad_metadata=metadata_info[ds]['vad_metadata'],
                                       include_neutral=include_neutral,
                                       sr=sr, win_max_length=win_max_length, win_shift=win_shift, win_min_length=win_min_length,
                                       feature_extractor=feature_extractor,
                                       transform=all_transforms[ds], 
                                       data_preprocessor=data_preprocessor)

            datasets_stats['MELD'][ds] = datasets[ds].info['stats']

    # Defining dataloaders
    dataloaders = {}
    for ds in ds_names:
        dataloaders[ds] = torch.utils.data.DataLoader(
            datasets[ds],
            batch_size=batch_size,
            shuffle=('train' in ds)
        )
        
    # Defining measures, measure with 0 index is main measure
    measures = [
        EmoSenCombinedMeasure('emo_sen_combined'), # main measure
        WARMeasure('emo_A(WAR)'),
        UARMeasure('emo_UAR'),
        WeightedF1Measure('emo_WF1'),
        MacroF1Measure('emo_MacroF1'),
        WARMeasure('sen_A(WAR)'),
        UARMeasure('sen_UAR'),
        WeightedF1Measure('sen_WF1'),
        MacroF1Measure('sen_MacroF1')
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
    
    model = model_cls(**model_args)
    model.to(device)
    
    # Defining weighted loss
    class_sample_count = datasets_stats['MELD']['train']['counts']
    loss = MTLoss(emotion_weights=torch.Tensor(class_sample_count['emo_7'] / sum(class_sample_count['emo_7'])).to(device), emotion_alpha=1,
                  sentiment_weights=torch.Tensor(class_sample_count['sen_3'] / sum(class_sample_count['sen_3'])).to(device), sentiment_alpha=1)
    
    # Defining optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)

    # Defining scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=10, T_mult=2)

    model, max_perf = net_trainer.run(model=model, loss=loss, optimizer=optimizer, scheduler=scheduler,
                                      num_epochs=num_epochs, dataloaders=dataloaders, datasets_stats=datasets_stats)

    for phase in ds_names:
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
    d_config = dconf['MELD']
    
    m_clses = [
        AudioModelT1,  AudioModelT2, AudioModelT3, AudioModelT4, AudioModelT5, AudioModelT6,
        AudioModelM1,  AudioModelM2, AudioModelM3, AudioModelM4, AudioModelM5, AudioModelM6,
        AudioModelX1,  AudioModelX2, AudioModelX3, AudioModelX4, AudioModelX5, AudioModelX6,
    ]

    logs_dir = {
        'T': '/media/maxim/WesternDigital/RAMAS2024/sc_meldtest/transformers',
        'M': '/media/maxim/WesternDigital/RAMAS2024/sc_meldtest/mamba',
        'X': '/media/maxim/WesternDigital/RAMAS2024/sc_meldtest/xlstm'
    }
    
    fe_clses = [ExHuBERTFeatureExtractor]
    win_params = [
        {'WIN_MAX_LENGTH': 4, 'WIN_SHIFT': 2}
    ]
    
    for win_param in win_params:
        for fe_cls in fe_clses:            
            for m_cls in m_clses:
                t_config = deepcopy(tconf)
                t_config['LOGS_ROOT'] = logs_dir[str(m_cls)[-4]]
                t_config['AUGMENTATION'] = False
                
                t_config['FEATURE_EXTRACTOR']['WIN_MAX_LENGTH'] = win_param['WIN_MAX_LENGTH']
                t_config['FEATURE_EXTRACTOR']['WIN_SHIFT'] = win_param['WIN_SHIFT']
                t_config['FEATURE_EXTRACTOR']['cls'] = fe_cls
                t_config['FEATURE_EXTRACTOR']['args']['win_max_length'] = win_param['WIN_MAX_LENGTH']
                
                t_config['MODEL']['cls'] = m_cls
                model_args = AttrDict()
                model_args.out_emo = len(d_config['C_NAMES']['emo'])
                model_args.out_sen = len(d_config['C_NAMES']['sen'])
                model_args.context_length = define_context_length(win_param['WIN_MAX_LENGTH'])
                t_config['MODEL']['args'] = {'config': model_args}
                
                main(d_config=d_config, t_config=t_config)

    
if __name__ == "__main__":
    run_expression_training()