"""
This is the script for extracting audio from video with/without filtering speech.
"""

import os
import wave
import shutil
import subprocess

import sox
from tqdm import tqdm


from audio.configs.singlecorpus_config import data_config


def convert_without_filtering(inp_path: str, 
                              out_path: str, 
                              checking: bool = True) -> None:
    """Convert video to audio using ffmpeg

    Args:
        inp_path (str): Input file path
        out_path (str): Output file path
        checking (bool, optional): Used for checking paths of the ffmpeg command. Defaults to True.
    """
    out_dirname = os.path.dirname(out_path)
    os.makedirs(out_dirname, exist_ok=True)

    # sample rate 16000
    command = f"ffmpeg -y -i {inp_path} -async 1 -vn -acodec pcm_s16le -ar 16000 -ac 1 {out_path}"
       
    if checking:
        print(command)
    else:
        _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)


def convert_with_filtering(inp_path: str, 
                           out_path: str, 
                           checking: bool = True) -> None:
    """Extract speech from the video file using Spleeter and ffmpeg

    Args:
        inp_path (str): Input file path
        out_path (str): Output file path
        checking (bool, optional): Used for checking paths of the spleeter/ffmpeg commands. Defaults to True.
    """
    out_dirname = os.path.dirname(out_path)
    os.makedirs(out_dirname, exist_ok=True)

    # 44100 for spleeter
    command = f"ffmpeg -y -i {inp_path} -async 1 -vn -acodec pcm_s16le -ar 44100 {out_path}"
       
    if checking:
        print(command)
    else:
        _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

    if not checking:
        inp_duration = sox.file_info.duration(out_path)

    # extract speech using spleeter
    command = f"spleeter separate -o {out_dirname} {out_path} -d 1620" # maximum length in seconds
    if checking:
        print(command)
    else:
        _ = subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT, env=os.environ.copy()
        )

    if not checking:
        spleeter_duration = sox.file_info.duration(out_path)

    # convert 44100 to 16000
    command = "ffmpeg -y -i {0} -async 1 -ar 16000 -ac 1 {1}".format(
        os.path.join(
            out_dirname, os.path.basename(out_path).split(".")[0], "vocals.wav"
        ),
        out_path,
    )
    
    if checking:
        print(command)
    else:
        _ = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        
        shutil.rmtree(
            os.path.join(out_dirname, os.path.basename(out_path).split(".")[0])
        )

    # check results for errors
    if not checking:
        final_duration = sox.file_info.duration(out_path)
        
        if (abs(inp_duration - spleeter_duration) < 1e-4) and (abs(inp_duration - final_duration) < 1e-4):
            pass
        else:
            print(f"Error {inp_path}")
            print(inp_duration, spleeter_duration, final_duration)
        

def convert_video_to_audio(data_config: dict, 
                           db: str,
                           filtering: bool = False,
                           checking: bool = True) -> None:
    """Loops through the directory, and extract speech from each video file using Spleeter and ffmpeg.

    Args:
        data_config (dict): Dictonary info of database
        db (str): Database: can be 'CMUMOSEI' or 'MELD' or 'RAMAS'
        filtering (bool, optional): Apply spleeter or not. Defaults to False.
        checking (bool, optional): Used for checking paths of the spleeter/ffmpeg commands. Defaults to True.
    """
    # run on CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    ds_names = ['train', 'dev', 'test'] if db == 'MELD' else ['']

    out_wavs_root = os.path.join(data_config['DATA_ROOT'], data_config['AUDIO_ROOT'])
    out_vocals_root = os.path.join(data_config['DATA_ROOT'], data_config['VOCALS_ROOT'])

    for ds_name in ds_names:        
        for fn in tqdm(os.listdir(os.path.join(data_config['VIDEO_ROOT'], ds_name))):
            if fn.startswith('.'): # corrupted train dia125_utt3
                continue
            
            convert_without_filtering(
                inp_path=os.path.join(data_config['VIDEO_ROOT'], ds_name, fn),
                out_path=os.path.join(out_wavs_root, ds_name, fn.replace("mp4", "wav").replace("mov", "wav").replace("avi", "wav")),
                checking=checking,
            )
            
            if filtering:
                convert_with_filtering(
                    inp_path=os.path.join(data_config['VIDEO_ROOT'], ds_name, fn),
                    out_path=os.path.join(out_vocals_root, ds_name, fn.replace("mp4", "wav").replace("mov", "wav").replace("avi", "wav")),
                    checking=checking,
                )


if __name__ == "__main__":
    dbs = list(data_config.keys())
    
    for db in dbs:
        print('Converting {}'.format(db))
        convert_video_to_audio(data_config=data_config[db], db=db, checking=False, filtering=True)