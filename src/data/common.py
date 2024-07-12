import os
import pickle

import numpy as np

import torch
import torchaudio


def read_audio(filepath: str, sample_rate: int) -> torch.Tensor:
    """Read audio using torchaudio

    Args:
        filepath (str): Path to audio file
        sample_rate (int): Sample rate of audio file

    Returns:
        torch.Tensor: Wave
    """
    full_wave, sr = torchaudio.load(filepath)

    if full_wave.size(0) > 1:
        full_wave = full_wave.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        full_wave = transform(full_wave)

    full_wave = full_wave.squeeze(0)
    
    return full_wave


def emo_to_label(value: np.ndarray[float], include_neutral: bool = False) -> np.ndarray[float]:
    """Convert emotion vector to labels
    [neutral, happy, sad, anger, surprise, disgust, fear]

    Args:
        value (np.ndarray[float]): Emotion vector
        include_neutral (bool, optional): Cut off Neutral emotion. Defaults to False.

    Returns:
        np.ndarray[float]: List of converted emotions
    """
    return value if include_neutral else value[1:]


def ohe_emotions(label: int) -> np.ndarray[float]:
    """Convert label to emotion vector
    [neutral, happy, sad, anger, surprise, disgust, fear]

    Args:
        label (int): Emotion label

    Returns:
        np.ndarray[float]: Converted emotion vector
    """
    res = np.zeros(7)
    res[label] = 1
    return res


def ohe_sentiment(value: int) -> np.ndarray[int]:
    """Applis OHE to sentiment value
    Negative, -1 -> 0 class -> [1, 0, 0]
    Neutral, 0 -> 1 class -> [0, 1, 0]
    Positive, 1 -> 2 class -> [0, 0, 1]

    Args:
        value (int): Sentiment value

    Returns:
        np.ndarray[int]: Converted OHE sentiment
    """
    return np.array([int(i == sen_to_label(value)) for i in range(3)])


def sen_to_label(value: int) -> int:
    """Convert sentiment value to label
    Negative, -1 -> 0 class
    Neutral, 0 -> 1 class
    Positive, 1 -> 2 class

    Args:
        value (int): Sentiment value

    Returns:
        int: Converted sentiment label
    """
    return {
        -1: 0,
        0: 1,
        1: 2
    }[value]


def label_to_sen(label: int) -> int:
    """Convert label to sentiment value
    0 class -> Negative, -1
    1 class -> Neutral, 0
    2 class -> Positive, 1

    Args:
        label (int): Sentiment label

    Returns:
        int: Converted sentiment value
    """
    return {
        0: -1,
        1: 0,
        2: 1
    }[label]


def save_data(data: any, filename: str) -> None:
    """Dumps data to pickle

    Args:
        data (any): Data
        filename (str): Filename
    """
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(filename: str) -> any:
    """Reads data from pickle

    Args:
        filename (str): Filename
    
    Returns:
        data (any): Data
    """
    data = None
    if os.path.exists(filename):
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)

    return data


def find_intersections(x: list[dict], y: list[dict]) -> list[dict]:
    """Find intersections of two lists of dicts with intervals

    Args:
        x (list[dict]): First list
        y (list[dict]): Second list

    Returns:
        list[dict]: Windows with VAD intersection
    """

    timings = []
    # `i` is pointer for `x`, `j` - for `y`
    i = 0
    j = 0

    while i < len(x) and j < len(y):
        # Left bound for intersecting segment
        l = max(x[i]['start'], y[j]['start'])
         
        # Right bound for intersecting segment
        r = min(x[i]['end'], y[j]['end'])

        if l <= r: # If segment is valid 
            timings.append({'start': l, 'end': r})
         
        # If i-th interval's right bound is 
        # smaller increment i else increment j
        if x[i]['end'] < y[j]['end']:
            i += 1
        else:
            j += 1

    return timings


def slice_audio(start_time: float, end_time: float, win_max_length: float, win_shift: float, win_min_length: float) -> list[dict]:
    """Slices audio on windows

    Args:
        start_time (float): Start time of audio
        end_time (float): End time of audio
        win_max_length (float): Window max length
        win_shift (float): Window shift
        win_min_length (float): Window min length

    Returns:
        list[dict]: List of dict with timings, f.e.: {'start': 0, 'end': 12}
    """    

    if end_time < start_time:
        return []
    elif (end_time - start_time) > win_max_length:
        timings = []
        while start_time < end_time:
            end_time_chunk = start_time + win_max_length
            if end_time_chunk < end_time:
                timings.append({'start': start_time, 'end': end_time_chunk})
            elif end_time_chunk == end_time: # if tail exact `win_max_length` seconds
                timings.append({'start': start_time, 'end': end_time_chunk})
                break
            else: # if tail less then `win_max_length` seconds
                if end_time - start_time < win_min_length: # if tail less then `win_min_length` seconds
                    break
                
                timings.append({'start': start_time, 'end': end_time})
                break

            start_time += win_shift
        return timings
    else:
        return [{'start': start_time, 'end': end_time}]


if __name__ == "__main__":
    print(slice_audio(12, 6.1, 4, 2, 2))