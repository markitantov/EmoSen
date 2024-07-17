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

from audio.data.cmumosei_dataset import CMUMOSEIDataset
from audio.data.data_preprocessors import Wav2Vec2DataPreprocessor
from audio.data.grouping import singlecorpus_grouping

from audio.models.audio_models_v2 import *

from audio.loss.loss import MLMTLoss

from audio.utils.accuracy import *

from audio.net_trainer.net_trainer import NetTrainer, LabelType

from audio.utils.common import get_source_code, define_seed
  

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
        get_source_code([main, model_cls, CMUMOSEIDataset, Wav2Vec2DataPreprocessor, NetTrainer]))
    
    # Defining datasets 
    ds_names = {
        'train': 'train', 
        'devel': 'dev',
        'test': 'test',
    }
    
    # Defining class names
    c_names = {
        'emo': ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear'],
        'sen': ['negative', 'neutral', 'positive']
    }
    
    c_names_to_display = {}
    for task, class_names in c_names.items():
        c_names_to_display[task] = [cn.capitalize() for cn in class_names]
    
    # Defining metadata and data augmentations
    labels = pd.read_csv(os.path.join(data_root, labels_file))
    with open(os.path.join(data_root, vad_file), 'rb') as handle:
        vad_metadata = pickle.load(handle)
    
    metadata_info = {}
    all_transforms = {}
    for ds in ds_names:
        metadata_info[ds] = {
            'labels': labels[labels['subset'] == ds_names[ds]],
            'dump_filepath': os.path.join(data_root, 'CMUMOSEI_{0}_{1}.pickle'.format(ds_names[ds].upper(), features_dump_file)),
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
        
    # Defining data preprocessor
    data_preprocessor = Wav2Vec2DataPreprocessor(model_name)
    
    # Defining datasets
    datasets = {}
    datasets_stats = {'CMUMOSEI': {}}
    for ds in ds_names:
        if 'train' in ds:
            datasets[ds] = torch.utils.data.ConcatDataset([
                CMUMOSEIDataset(audio_root=os.path.join(data_root, audio_root),
                                metadata=metadata_info[ds]['labels'], 
                                dump_filepath=metadata_info[ds]['dump_filepath'],
                                vad_metadata=vad_metadata,
                                include_neutral=False,
                                sr=16000, win_max_length=4, win_shift=2, win_min_length=0,
                                transform=t, data_preprocessor=data_preprocessor) for t in all_transforms[ds]
                ]
            )

            datasets_stats['CMUMOSEI'][ds] = datasets[ds].datasets[0].info['stats']
        else:
            datasets[ds] = CMUMOSEIDataset(audio_root=os.path.join(data_root, audio_root),
                                           metadata=metadata_info[ds]['labels'], 
                                           dump_filepath=metadata_info[ds]['dump_filepath'],
                                           vad_metadata=vad_metadata, 
                                           include_neutral=False,
                                           sr=16000, win_max_length=4, win_shift=2, win_min_length=0,
                                           transform=all_transforms[ds], data_preprocessor=data_preprocessor)

            datasets_stats['CMUMOSEI'][ds] = datasets[ds].info['stats']

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
        MultilabelEmoSenCombinedMeasure('emo_sen_combined'), # main measure
        MeanWeightedAccuracyMeasure('emo_mWA'),
        MeanWeightedF1Measure('emo_mWF1'),
        MeanMacroF1Measure('emo_mMacroF1'),
        MeanUARMeasure('emo_mUAR'),
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
                             label_type=LabelType.MULTILABEL,                            
                             group_predicts_fn=singlecorpus_grouping,
                             source_code=source_code,
                             c_names_to_display=c_names_to_display)
    
    # Defining model
    model_cfg = AutoConfig.from_pretrained(model_name)
    model_cfg.out_emo = len(c_names['emo'])
    model_cfg.out_sen = len(c_names['sen'])
    model = model_cls.from_pretrained(model_name, config=model_cfg)
    model.to(device)
    
    # Defining weighted loss
    class_sample_count = datasets_stats['CMUMOSEI']['train']['counts']
    loss = MLMTLoss(emotion_weights=torch.Tensor(class_sample_count['emo_6'] / sum(class_sample_count['emo_6'])).to(device), emotion_alpha=1,
                    sentiment_weights=torch.Tensor(class_sample_count['sen_3'] / sum(class_sample_count['sen_3'])).to(device), sentiment_alpha=1)
    
    # Defining optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

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
    d_config = dconf['CMUMOSEI']
    model_cls = [AudioModelV3, AudioModelV4, AudioModelV5]

    for augmentation in [False]:
        for m_cls in model_cls:
            t_config = deepcopy(tconf)
            t_config['AUGMENTATION'] = augmentation
            t_config['MODEL_PARAMS']['model_cls'] = m_cls
                
            main(d_config=d_config, t_config=t_config)

    
if __name__ == "__main__":
    run_expression_training()