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
        'C_NAMES': {
            'emo': ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear'],
            'sen': ['negative', 'neutral', 'positive']
        },
        'INCLUDE_NEUTRAL': False,
    },
    'MELD': {
        'DATA_ROOT': '',
        'VIDEO_ROOT': '',
        'AUDIO_ROOT': '',
        'VOCALS_ROOT': '',
        'VAD_FILE': '',
        'LABELS_FILE': '',
        'C_NAMES': {
            'emo': ['neutral', 'happy', 'sad', 'anger', 'surprise', 'disgust', 'fear'],
            'sen': ['negative', 'neutral', 'positive']
        },
        'INCLUDE_NEUTRAL': True
    },
    'RAMAS': {
        'DATA_ROOT': '',
        'VIDEO_ROOT': '',
        'AUDIO_ROOT': '',
        'VOCALS_ROOT': '',
        'VAD_FILE': '',
        'LABELS_FILE': '',
        'C_NAMES': {
            'emo': ['neutral', 'happy', 'sad', 'anger', 'surprise', 'disgust', 'fear'],
            'sen': ['negative', 'neutral', 'positive']
        },
        'INCLUDE_NEUTRAL': True
    },
}

training_config: dict = {
    'LOGS_ROOT': '',
    'MODEL': {
        'cls': None,
        'args': {
        }
    },
    'FEATURE_EXTRACTOR': {
        'FEATURES_DUMP_FILE': '',
        'WIN_MAX_LENGTH': 4,
        'WIN_SHIFT': 2,
        'WIN_MIN_LENGTH': 0,
        'SR': 16000,
        'cls': None,
        'args': {
            'sr': 16000
        }
    },
    'DATA_PREPROCESSOR': {
        'cls': None,
        'args': {
        }
    },
    'AUGMENTATION': None,
    'NUM_EPOCHS': None,
    'BATCH_SIZE': None,
}