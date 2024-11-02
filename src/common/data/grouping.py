import sys

sys.path.append('src')

import numpy as np
import pandas as pd

from common.data.utils import label_to_sen, save_data


def singlecorpus_grouping(targets: dict[list], predicts: dict[list], samples_info: dict[tuple], current_phase: str = None, datasets_stats: dict[dict] = None) -> dict[tuple]:
    """Single corpus grouping
    1. Forms list of filenames and db_names
    2. Forms two pd.DataFrame: for emotion, and for sentiment with targets, predicts, filenames, db_names
    3. Processes emotions:
        3.1. In case of CMUMOSEI:
            3.1.1 In case of multicorpus (7 classes for CMUMOSEI):
                3.1.1.1. Converts 7 classes to 6 classes
            3.1.2 Else
                3.1.1.1 Converts each emotion vector according rule: 1 if emo_idx >= 0.5 else 0, example [0.4, 0, 0.5, 0.7, 1.0, 0.0] => [0, 0, 1, 1, 1, 0]
            3.1.2. Splits list of targets/predicts into separate columns, example [0, 0, 1, 1, 1, 0] => [0], [0], [1], [1], [1], [0]
            3.1.3. Drops original lists of target/predicts
        3.2. In case of other corpora (RAMAS or MELD):
            3.2.1 Applies argmax on targets/predicts, example [0.4, 0, 0.5, 0.7, 1.0, 0.0] => [4]
        
        3.3. Groups predicts using pandas

        3.4 In case of CMUMOSEI:
            3.4.1. Converts each column of targets\predicts with emotions to list of emotions [0], [0], [1], [1], [1], [0] => [0, 0, 1, 1, 1, 0]
    4. Processes sentiment:
        4.1. Applies argmax on targets/predicts, example [0.4, 0, 0.5, 0.7, 1.0, 0.0] => [4]
        4.2. Groups predicts using pandas
    5. [Optional] Appends missing files
        5.1. Forms set of file, detects missing files
        5.2. Processes emotions
            5.2.1. In case of CMUMOSEI
                5.2.1.1 For missing file applies OHE to majority class of TRAIN for predicts, picks targets from current phase
            5.2.2. In case of other corpora (RAMAS or MELD):
                5.2.2.1 For missing file picks majority class of TRAIN for predicts, picks targets from current phase
        5.3. Processes sentiment
            5.3.1 For missing file picks majority class of TRAIN for predicts, applies argmax on targets from current phase
    6. Forms resulted dict

    Args:
        targets (dict[list]): Dict of lists with targets for 'emo' and 'sen' (np.ndarray([0, 0, 1, 1, 0, 0]), np.ndarray([0, 0, 1]))
        predicts (dict[list]): Dict of lists with predicts for 'emo' and 'sen' (np.ndarray([0.0, 0.3, 0.5, 1.0, 0.0, 0.0]), np.ndarray([0.3, 0.3, 0.3]))
        samples_info (dict[tuple]): List of dicts with sample info:
                                    0: {'filename': ['-3g5yACwYnA_74.083_82.7645.wav', '-3g5yACwYnA_27.031_41.3.wav', '-3g5yACwYnA_13.6315_27.031.wav', 
                                                     '-3g5yACwYnA_13.6315_27.031.wav', '-3g5yACwYnA_27.031_41.3.wav', '-3g5yACwYnA_4.84_13.6315.wav', 
                                                     '-3g5yACwYnA_82.7645_100.555.wav', '-3g5yACwYnA_82.7645_100.555.wav', '-3g5yACwYnA_13.6315_27.031.wav', 
                                                     '-3g5yACwYnA_74.083_82.7645.wav', '-3g5yACwYnA_13.6315_27.031.wav', '-3g5yACwYnA_82.7645_100.555.wav],
                                        'start_t': [...],
                                        'end_t': [...],
                                        'db': ['CMUMOSEI', 'CMUMOSEI', 'CMUMOSEI', 
                                               'CMUMOSEI', 'CMUMOSEI', 'CMUMOSEI', 
                                               'CMUMOSEI', 'CMUMOSEI', 'CMUMOSEI', 
                                               'CMUMOSEI', 'CMUMOSEI', 'CMUMOSEI'],
                                        },
                                    1: {...}

        current_phase (str): Current phase, can be train/dev/test. Defaults to None.
        datasets_stats: Metadata statistics for all corpora. Defaults to None.

    Returns:
        dict[tuple]: For each db tuple: 
                        targets - dict with keys ['emo', 'sen']
                        predicts - dict with keys ['emo', 'sen']
                        list(filenames)
    """
    res = {}
    filenames = []
    db_names = []
    for s_info in samples_info:
        filenames.extend(s_info['filename'])
        db_names.extend(s_info['db'])
    
    d_emo = {}
    d_emo['predicts'] = predicts['emo']
    d_emo['targets'] = targets['emo']
    d_emo['filenames'] = filenames
    d_emo['db_names'] = db_names
    df_all_emo = pd.DataFrame(d_emo)

    d_sen = {}
    d_sen['predicts'] = predicts['sen']
    d_sen['targets'] = targets['sen']
    d_sen['filenames'] = filenames
    d_sen['db_names'] = db_names
    df_all_sen = pd.DataFrame(d_sen)

    ordered_db_names = sorted(list(set(db_names)))
    for db in ordered_db_names:
        df_emo = df_all_emo[df_all_emo['db_names'] == db].copy()
        emo_idx = [i for i in range(0, len(predicts['emo'][0]))]
        emo_idx = emo_idx[:-1] if 'CMUMOSEI' in db else emo_idx
        if 'CMUMOSEI' in db:
            if len(df_emo['predicts'].values[0]) > 6: # Case of multiclass with softlabels
                df_emo['targets'] = list(np.where(np.asarray(df_emo['targets'].tolist()) > 0, 1, 0)[:, 1:].astype(int))
                predicts_emo = np.asarray(df_emo['predicts'].tolist())
                neutral_emotion_treshold = 1 - 1 / 7 
                other_emotion_treshold = 1 / 7
                neutral_emotion_mask = predicts_emo[:, 0] >= neutral_emotion_treshold
                other_emotions_mask = predicts_emo[:, 1:] >= other_emotion_treshold
                new_predicts_emo = np.zeros_like(predicts_emo[:, 1:])
                other_emotions = other_emotions_mask.astype(int)
                new_predicts_emo[~neutral_emotion_mask] = other_emotions[~neutral_emotion_mask]
                new_predicts_emo = list(new_predicts_emo.astype(int))

                df_emo['predicts'] = new_predicts_emo                
            else:
                df_emo['predicts'] = df_emo['predicts'].apply(lambda x: np.where(x >= 0.5, 1, 0))
    
            df_emo[['emop_{0}'.format(i) for i in emo_idx]] = pd.DataFrame(df_emo['predicts'].tolist(), index=df_emo.index)
            df_emo[['emot_{0}'.format(i) for i in emo_idx]] = pd.DataFrame(df_emo['targets'].tolist(), index=df_emo.index)
            df_emo = df_emo.drop(columns=['targets', 'predicts'])
        else:
            df_emo['targets'] = df_emo['targets'].apply(lambda x: np.argmax(x))
            df_emo['predicts'] = df_emo['predicts'].apply(lambda x: np.argmax(x))

        df_emo = df_emo.groupby('filenames', as_index=False).agg(lambda x: pd.Series.mode(x)[0])
        if 'CMUMOSEI' in db:
            df_emo['predicts'] = df_emo[['emop_{0}'.format(i) for i in emo_idx]].values.tolist()
            df_emo['targets'] = df_emo[['emot_{0}'.format(i) for i in emo_idx]].values.tolist()

        emo_targets = df_emo['targets'].to_list()
        emo_predicts = df_emo['predicts'].to_list()
        
        df_sen = df_all_sen[df_all_sen['db_names'] == db].copy()
        df_sen['targets'] = df_sen['targets'].apply(lambda x: np.argmax(x))
        df_sen['predicts'] = df_sen['predicts'].apply(lambda x: np.argmax(x))
        df_sen = df_sen.groupby('filenames', as_index=False).agg(lambda x: pd.Series.mode(x)[0])
        sen_targets = df_sen['targets'].to_list()
        sen_predicts = df_sen['predicts'].to_list()

        filenames = df_sen['filenames'].to_list()
    
        if datasets_stats:
            emo_type = 'emo_{0}'.format(len(emo_idx))
            current_phase = current_phase.split('_')[0]
            missing_files = set(datasets_stats[db][current_phase]['fns'].keys()) - set(filenames)
            for missing_file in missing_files:
                if 'CMUMOSEI' in db:
                    emo_targets.append(datasets_stats[db][current_phase]['fns'][missing_file][emo_type])
                    emo_predicts.append((np.arange(len(emo_idx)) == datasets_stats[db]['train']['majority_class'][emo_type]).astype(float))
                else:
                    emo_targets.append(np.argmax(datasets_stats[db][current_phase]['fns'][missing_file][emo_type]))
                    emo_predicts.append(datasets_stats[db]['train']['majority_class'][emo_type])

                sen_targets.append(np.argmax(datasets_stats[db][current_phase]['fns'][missing_file]['sen_3']))
                sen_predicts.append(datasets_stats[db]['train']['majority_class']['sen_3']) # argmax-ready

                filenames.append(missing_file)

        res[db] = (
            {'emo': emo_targets, 'sen': sen_targets}, 
            {'emo': emo_predicts, 'sen': sen_predicts}, 
            filenames
        )
    
    return res


def multicorpus_grouping(targets: dict[list], predicts: dict[list], samples_info: dict[tuple], current_phase: str = None, datasets_stats: dict[dict] = None) -> dict[tuple]:
    if 'train' not in current_phase: # Avoid inconsistencies between 6 and 7 classes
        db_names = []
        for s_info in samples_info:
            db_names.extend(s_info['db'])

        db_name = set(db_names)
        assert len(db_name) == 1, 'There are spy agents in the dataloader'

        if 'CMUMOSEI' in db_name:
            new_targets_emo = list(np.where(np.asarray(targets['emo']) > 0, 1, 0)[:, 1:])

            predicts_emo = np.asarray(predicts['emo'])
            neutral_emotion_treshold = 1 - 1 / 7 
            other_emotion_treshold = 1 / 7
            neutral_emotion_mask = predicts_emo[:, 0] >= neutral_emotion_treshold
            other_emotions_mask = predicts_emo[:, 1:] >= other_emotion_treshold
            new_predicts_emo = np.zeros_like(predicts_emo[:, 1:])
            other_emotions = other_emotions_mask.astype(int)
            new_predicts_emo[~neutral_emotion_mask] = other_emotions[~neutral_emotion_mask]
            new_predicts_emo = list(new_predicts_emo)

            targets['emo'] = new_targets_emo
            predicts['emo'] = new_predicts_emo
        
    res = singlecorpus_grouping(targets=targets, predicts=predicts, samples_info=samples_info, current_phase=current_phase, datasets_stats=datasets_stats)
    return res


if __name__ == "__main__":      
    import pickle
    with open('test_grouping.pickle', 'rb') as handle:
        d = pickle.load(handle)

    # for idx, t in enumerate(d['targets']['emo']):
    #     d['targets']['emo'][idx] = np.where(t > 0, 1, 0)
            
    # singlecorpus_grouping(d['targets'], d['predicts'], d['samples_info'])

    multicorpus_grouping(*d)
