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

from transformers import AutoConfig

from audio.configs.singlecorpus_config import data_config as dconf
from audio.configs.singlecorpus_config import training_config as tconf

from audio.augmentation.wave_augmentation import RandomChoice, PolarityInversion, WhiteNoise, Gain

from audio.data.meld_dataset import MELDDataset
from audio.data.data_preprocessors import BaseDataPreprocessor
from audio.data.grouping import singlecorpus_grouping

from audio.features.feature_extractors import AudeeringFeatureExtractor

from audio.models.audio_models_v3 import *

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
    features_dump_file = d_config['FEATURES_DUMP_FILE']
    
    logs_root = t_config['LOGS_ROOT']
    model_cls = t_config['MODEL_PARAMS']['model_cls']
    model_name = t_config['MODEL_PARAMS']['args']['model_name']
    aug = t_config['AUGMENTATION']
    num_epochs = t_config['NUM_EPOCHS']
    batch_size = t_config['BATCH_SIZE']
    
    source_code = 'Data configuration:\n{0}\nTraining configuration:\n{1}\n\nSource code:\n{2}'.format(
        pprint.pformat(d_config),
        pprint.pformat(t_config),
        get_source_code([main, model_cls, MELDDataset, AudeeringFeatureExtractor, BaseDataPreprocessor, NetTrainer]))
    
    # Defining datasets 
    ds_names = {
        'train': 'train', 
        'devel': 'dev',
        'test': 'test',
    }
    
    # Defining class names
    c_names = {
        'emo': ['neutral', 'happy', 'sad', 'anger', 'surprise', 'disgust', 'fear'],
        'sen': ['negative', 'neutral', 'positive']
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
    feature_extractor = AudeeringFeatureExtractor(sr=16000, win_max_length=4)
    data_preprocessor = BaseDataPreprocessor()
    
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
                            include_neutral=True,
                            sr=16000, win_max_length=4, win_shift=2, win_min_length=0,
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
                                       include_neutral=True,
                                       sr=16000, win_max_length=4, win_shift=2, win_min_length=0,
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
    
    # Defining model
    model_cfg = AttrDict()
    model_cfg.out_emo = len(c_names['emo'])
    model_cfg.out_sen = len(c_names['sen'])
    model = model_cls(config=model_cfg)
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
    model_cls = [AudioModelV3, AudioModelV4, AudioModelV5]

    for augmentation in [False]:
        for m_cls in model_cls:
            t_config = deepcopy(tconf)
            t_config['LOGS_ROOT'] = '/media/maxim/WesternDigital/RAMAS2024/singlecorpus_meld/'
            t_config['AUGMENTATION'] = augmentation
            t_config['MODEL_PARAMS']['model_cls'] = m_cls
                
            main(d_config=d_config, t_config=t_config)

    
if __name__ == "__main__":
    run_expression_training()