"""
This is the script for extracting audio from video with/without filtering speech.
"""

import os
import pickle

import torch
from tqdm import tqdm

from audio.configs.singlecorpus_config import data_config


model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)

(get_speech_timestamps, _, read_audio, _, _) = utils


def detect_speech(inp_path: str,
                  sampling_rate: int = 16000) -> list[dict]:
    """Finds speech segments using VAD

    Args:
        inp_path (str): Input file path
        sampling_rate (int, optional): Sampling rate of audio. Defaults to 16000.

    Returns:
        list[dict]: list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
    """
    
    wav = read_audio(inp_path, sampling_rate=sampling_rate)
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
    
    return speech_timestamps


def run_vad(data_config: dict,
            db: str) -> None:
    """Loops through the directory, and run VAD on DB.

    Args:
        data_config (dict): Dictonary info of database
        db (str): Database: can be 'CMUMOSEI' or 'MELD' or 'RAMAS'
    """
    # run on CPU
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ds_names = ['train', 'dev', 'test'] if db == 'MELD' else ['']
    
    for ds_name in ds_names:
        res = {} 
        for fn in tqdm(os.listdir(os.path.join(data_config['DATA_ROOT'], data_config['VOCALS_ROOT'], ds_name))):
            if fn.startswith('.'): # corrupted train dia125_utt3
                continue
            
            speech_timestamps = detect_speech(inp_path=os.path.join(data_config['DATA_ROOT'], data_config['VOCALS_ROOT'], ds_name, fn))
            res[fn] = speech_timestamps

        with open(os.path.join(data_config['DATA_ROOT'], 'vad_{0}.pickle'.format(ds_name) if ds_name else 'vad.pickle'), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    dbs = list(data_config.keys())
    
    for db in dbs:
        print('Starting VAD on {}'.format(db))
        run_vad(data_config=data_config[db], db=db)