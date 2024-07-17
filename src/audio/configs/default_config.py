import sys

sys.path.append('src')

data_config: dict = {
    'CMUMOSEI': {
        'DATA_ROOT': '',
        'VIDEO_ROOT': '',
        'AUDIO_ROOT': '',
        'VOCALS_ROOT': '',
        'VAD_FILE': '',
        'LABELS_FILE': '',
    },
    'MELD': {
        'DATA_ROOT': '',
        'VIDEO_ROOT': '',
        'AUDIO_ROOT': '',
        'VOCALS_ROOT': '',
        'VAD_FILE': '',
        'LABELS_FILE': '',
    },
    'RAMAS': {
        'DATA_ROOT': '',
        'VIDEO_ROOT': '',
        'AUDIO_ROOT': '',
        'VOCALS_ROOT': '',
        'VAD_FILE': '',
        'LABELS_FILE': '',
    },  
}

training_config: dict = {
    'LOGS_ROOT': '',
    'MODEL_PARAMS': {
        'model_cls': None,
        'args': {
            None
        }
    },
    'AUGMENTATION': None,
    'NUM_EPOCHS': None,
    'BATCH_SIZE': None,
}