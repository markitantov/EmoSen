import os
import logging
from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from audio.visualization.visualize import conf_matrix, plot_conf_matrix
from audio.utils.common import create_logger


class LabelType(Enum):
    """Label type Enum
    Used in NetTrainer
    """
    SINGLELABEL: int = 1
    MULTILABEL: int = 2


class NetTrainer:
    """Trains the model
       - Performs logging:
            Logs general information (epoch number, phase, loss, performance) in file and console
            Creates tensorboard logs for each phase
            Saves source code in file
            Saves the best model and confusion matrix of this model
       - Runs train/test/devel phases
       - Calculates performance measures
       - Calculates confusion matrix
       - Saves models
       
        Args:
            log_root (str): Directory for logging
            experiment_name (str): Name of experiments for logging
            c_names (list[str]): Class names to calculate the confusion matrix 
            metrics (list[BaseMeasure]): List of performance measures based on the best results of which the model will be saved. 
                                         The first measure (0) in the list will be used for this, the others provide 
                                         additional information
            device (torch.device): Device where the model will be trained
            label_type (LabelType, optional): Label type. Defaults to LabelType.SINGLELABEL.
            group_predicts_fn (callable, optional): Function for grouping predicts, f.e. file-wise or windows-wise. 
                                                    It can be used to calculate performance metrics on train/devel/test sets. 
                                                    Defaults to None.
            source_code (str, optional): Source code and configuration for logging. Defaults to None.
            c_names_to_display (list[str], optional): Class names to visualize confuson matrix. Defaults to None.
        """
    def __init__(self, 
                 log_root: str, 
                 experiment_name: str,
                 c_names: list[str], 
                 measures: list[object], 
                 device: torch.device, 
                 label_type: LabelType = LabelType.SINGLELABEL,
                 group_predicts_fn: callable = None, 
                 source_code: str = None, 
                 c_names_to_display: list[str] = None) -> None:
        self.device = device

        self.model = None
        self.loss = None
        self.optimizer = None
        self.scheduler = None

        self.datasets_stats = None

        self.log_root = log_root
        self.exp_folder_name = experiment_name

        self.measures = measures
        self.c_names = c_names
        self.c_names_to_display = c_names_to_display
        self.label_type = label_type

        if source_code:
            os.makedirs(os.path.join(self.log_root, self.exp_folder_name, 'logs'), exist_ok=True)
            with open(os.path.join(self.log_root, self.exp_folder_name, 'logs', 'source.log'), 'w') as f:
                f.write(source_code)

        self.group_predicts_fn = group_predicts_fn

        self.logging_paths = None
        self.logger = None

    def create_loggers(self, fold_num: int = None) -> None:
        """Creates folders for logging experiments:
        - general logs (log_path)
        - models folder (model_path)
        - tensorboard logs (tb_log_path)

        Args:
            fold_num (int, optional): Used for cross-validation to specify fold number. Defaults to None.
        """
        fold_name = '' if fold_num is None else 'fold_{0}'.format(fold_num)
        self.logging_paths = {
            'log_path': os.path.join(self.log_root, self.exp_folder_name, 'logs'),
            'model_path': os.path.join(self.log_root, self.exp_folder_name, 'models', fold_name),
            'tb_log_path': os.path.join(self.log_root, self.exp_folder_name, 'tb', fold_name),
        }

        for log_path in self.logging_paths:
            if log_path == 'tb_log_path':
                continue

            os.makedirs(self.logging_paths[log_path], exist_ok=True)

        self.logger = create_logger(os.path.join(self.log_root, self.exp_folder_name, 'logs',
                                                 '{0}.log'.format(fold_name if fold_name else 'logs')),
                                    console_level=logging.NOTSET,
                                    file_level=logging.NOTSET)
    
    def run(self, 
            model: torch.nn.Module, 
            loss: torch.nn.modules.loss, 
            optimizer: torch.optim, 
            scheduler: torch.optim.lr_scheduler, 
            num_epochs: int,
            dataloaders: dict[torch.utils.data.dataloader.DataLoader],
            datasets_stats: dict[dict] = None, 
            log_epochs: list[int] = [], 
            fold_num: int = None, 
            verbose: bool = True) -> None:
        """Iterates over epochs including the following steps:
        - Iterates over phases (train/devel/test phase):
            - Calls `iterate_model` function (as main loop for training/validation/testing)
            - Calculates performance measures (or metrics) using `calc_metrics` function
            - Performs logging
            - Compares performance with previous epochs on phase
            - Calculates confusion matrix
            - Saves better model and confusion matrix
            - Saves epoch/phase/loss/performance statistics in csv file

        Args:
            model (torch.nn.Module): Model instance
            loss (torch.nn.modules.loss): Loss function
            optimizer (torch.optim): Optimizer
            scheduler (torch.optim.lr_scheduler): Scheduler for dynamicly change LR
            num_epochs (int): Number of training epochs
            dataloaders (dict[torch.utils.data.dataloader.DataLoader]): Dataloader
            datasets_stats (dict[dict]): Statistics of dataset for corresponding Dataloader. Defaults to None.
            log_epochs (list[int], optional): Exact epoch number for logging. Defaults to [].
            fold_num (int, optional): Used for cross-validation to specify fold number. Defaults to None.
            verbose (bool, optional): Detailed output including tqdm. Defaults to True.
        """
        phases = list(dataloaders.keys())
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.datasets_stats = datasets_stats
        
        self.create_loggers(fold_num)
        d_global_stats = []
        
        summary = {} 
        max_perf = {}
        for phase in phases:
            os.makedirs(os.path.join(self.logging_paths['tb_log_path'], phase), exist_ok=True)
            summary[phase] = SummaryWriter(os.path.join(self.logging_paths['tb_log_path'], phase))
            
            main_measure_name = str(self.measures[0])
            max_perf[phase] = {
                'epoch': 0,
                'performance': {
                    main_measure_name: 0,
                },
            }
                
        self.logger.info(self.exp_folder_name)
        for epoch in range(1, num_epochs + 1):
            self.logger.info('Epoch {}/{}:'.format(epoch, num_epochs))
            d_epoch_stats = {'epoch': epoch}
            
            for phase, dataloader in dataloaders.items():
                if 'test' in phase and dataloader is None:
                    continue
                
                joint_info, epoch_loss = self.iterate_model(phase=phase,
                                                            dataloader=dataloader,
                                                            epoch=epoch,
                                                            verbose=verbose)
                
                for db, vs in joint_info.items():
                    targets, predicts, sample_info = vs                     
                    
                    performance = self.calc_metrics(targets, predicts, sample_info)
                                
                    d_epoch_stats['{}_loss'.format(phase)] = epoch_loss
                    summary[phase].add_scalar('loss', epoch_loss, epoch)
                
                    epoch_score = performance[main_measure_name]
                    for measure in performance:
                        summary[phase].add_scalar(measure, performance[measure], epoch)                    
                        d_epoch_stats['{}_{}'.format(phase, measure)] = performance[measure]
                
                    is_max_performance = (
                            ((('test' in phase) or ('devel' in phase)) and (epoch_score > max_perf[phase]['performance'][main_measure_name])) or
                            ((('test' in phase) or ('devel' in phase)) and (epoch in log_epochs)))
                
                    if is_max_performance:
                        if epoch_score > max_perf[phase]['performance'][main_measure_name]:
                            max_perf[phase]['performance'] = performance
                            max_perf[phase]['epoch'] = epoch
                    
                        self.draw_confusion_matrix(targets=targets, predicts=predicts, 
                                                   epoch=epoch, phase=phase, 
                                                   main_measure_name=main_measure_name, epoch_score=epoch_score)
                    
                        model.cpu()
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.state_dict(),
                        }, os.path.join(self.logging_paths['model_path'], 'epoch_{0}.pth'.format(epoch)))
                    
                        model.to(self.device)
                
                    if ('devel' in phase) and (epoch == max_perf['devel']['epoch']) and ('test' in phase):
                        self.draw_confusion_matrix(targets=targets, predicts=predicts, 
                                                   epoch=epoch, phase=phase, 
                                                   main_measure_name=main_measure_name, epoch_score=epoch_score)
                
            d_global_stats.append(d_epoch_stats)
            pd_global_stats = pd.DataFrame(d_global_stats)
            pd_global_stats.to_csv(os.path.join(self.log_root, 
                                                self.exp_folder_name, 
                                                'logs', 
                                                'stats.csv' if fold_num is None else 'fold_{0}.csv'.format(fold_num)),
                                   sep=';', index=False)
            
            self.logger.info('')
            
        for phase in phases[1:]:
            self.logger.info(phase.capitalize())
            self.logger.info('Epoch: {}, Max performance:'.format(max_perf[phase]['epoch']))
            self.logger.info([measure for measure in max_perf[phase]['performance']])
            self.logger.info([max_perf[phase]['performance'][measure] for measure in max_perf[phase]['performance']])
            self.logger.info('')

        for phase in phases:
            summary[phase].close()
            
        return model, max_perf
    
    def iterate_model(self, 
                      phase: str, 
                      dataloader: torch.utils.data.dataloader.DataLoader, 
                      epoch: int = None, 
                      verbose: bool = True) -> tuple[dict[tuple[list[np.ndarray], list[np.ndarray], list]], float]:
        """Main training/validation/testing loop:
        ! Note ! This loop needs to be changed if you change scheduler. Default scheduler is CosineAnnealingWarmRestarts
        - Applies sigmoid function on emotion predicts and softmax function on sentiment predicts

        Args:
            phase (str): Name of phase: could be train, devel(valid), test
            dataloader (torch.utils.data.dataloader.DataLoader): Dataloader of phase
            epoch (int, optional): Epoch number. Defaults to None.
            verbose (bool, optional): Detailed output with tqdm. Defaults to True.

        Returns:
            tuple[dict[tuple[list[np.ndarray], list[np.ndarray], list]], float]: Dictionary with tuple of targets, 
                                                                                 predicts, 
                                                                                 sample_info after grouping predicts/targets, and
                                                                                 epoch_loss
        """
        
        targets = {'emo': [], 'sen': []}
        predicts = {'emo': [], 'sen': []}
        sample_info = []
        
        if 'train' in phase:
            self.model.train()
        else:
            self.model.eval()
        
        loss_values = {
            'total': 0.,
            'emotion': 0.,
            'sentiment': 0,
        }
        
        iters = len(dataloader)

        # Iterate over data.
        for idx, data in enumerate(tqdm(dataloader, disable=not verbose)):
            inps, labs, s_info = data
            
            inps = inps.to(self.device)
            labs = {k: v.to(self.device) for k, v in labs.items()}
            
            self.optimizer.zero_grad()

            # forward and backward
            preds = None
            with torch.set_grad_enabled('train' in phase):
                preds = self.model(inps)
                if self.loss:
                    total_loss_value = self.loss(preds, labs)

                # backward + optimize only if in training phase
                if 'train' in phase and self.loss:
                    total_loss_value.backward()
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step(epoch + idx / iters)

            # statistics
            if self.loss:
                loss_values['total'] += total_loss_value.item() * len(s_info)
                loss_values['emotion'] += self.loss.emotion_loss_value.item() * len(s_info)
                loss_values['sentiment'] += self.loss.sentiment_loss_value.item() * len(s_info)
            
            if self.label_type == LabelType.SINGLELABEL:                            
                preds = {'emo': F.softmax(preds['emo'], dim=-1), 'sen': F.softmax(preds['sen'], dim=-1)}
            else:
                preds = {'emo': F.sigmoid(preds['emo']), 'sen': F.softmax(preds['sen'], dim=-1)}
            
            for task in ['emo', 'sen']:
                targets[task].extend(labs[task].cpu().numpy())
                predicts[task].extend(preds[task].cpu().detach().numpy())

            sample_info.extend(s_info)

        for loss_type in loss_values:
            loss_values[loss_type] = loss_values[loss_type] / iters if self.loss else 0
        
        if verbose:
            self.logger.info('Epoch: {}. {}. {}Performance:'.format(epoch, phase.capitalize(), 
                                                                      ''.join(['Loss {0}: {1:.4f}, '.format(loss_type, loss_value) for loss_type, loss_value in loss_values.items()])))

        if self.group_predicts_fn:
            joint_info = self.group_predicts_fn(targets, predicts, sample_info, phase, self.datasets_stats)
        else:
            joint_info = {sample_info[0]['db']: (targets, predicts, sample_info)}

        return joint_info, loss_values['total']
    
    def draw_confusion_matrix(self, 
                              targets: list[np.ndarray], 
                              predicts: list[np.ndarray], 
                              epoch: int, 
                              phase: str, 
                              main_measure_name: str, 
                              epoch_score: float):
        """Draw confusion matrix

        Args:
            targets (list[np.ndarray]): List of targets
            predicts (list[np.ndarray]): List of predicts
            epoch (int): Number of epoch
            phase (str): Name of phase: could be train, devel(valid), test
            main_measure_name (str): Name of measure
            epoch_score (float): Score of performance measure
        """
        for task in ['emo', 'sen']:
            if self.label_type == LabelType.MULTILABEL and 'emo' in task:
                continue
                        
        cm = conf_matrix(targets[task], predicts[task], [i for i in range(len(self.c_names[task]))])
        res_name = 'epoch_{0}_{1}_{2}_{3:.3f}'.format(epoch, task, phase, epoch_score)
        confusion_matrix_title = '{0}. {1}. {2} = {3:.3f}%'.format(task.replace('emo', 'Emotion').replace('sen', 'Sentiment'), 
                                                                   phase.capitalize(), 
                                                                   str(main_measure_name),
                                                                   epoch_score)
        plot_conf_matrix(cm, 
                         labels=self.c_names_to_display[task] if self.c_names_to_display[task] else self.c_names[task],
                         xticks_rotation=45,
                         title=confusion_matrix_title,
                         save_path=os.path.join(self.logging_paths['model_path'], '{0}.svg'.format(res_name)))


    def calc_metrics(self, 
                     targets: list[np.ndarray], 
                     predicts: list[np.ndarray], 
                     filenames: list[str],
                     verbose: bool = True) -> dict[float]:
        """Calculates each performance measure from `self.measures`

        Args:
            targets (list[np.ndarray]): List of targets
            predicts (list[np.ndarray]): List of predicts
            filenames (list[str]): List of filenames
            verbose (bool, optional): Detailed output of each performance measure. Defaults to True.

        Returns:
            dict[float]: Return dictionary [str(measure)] = value
        """
        performance = {}
        for measure in self.measures:
            if '_m' in str(measure):
                # mWA, mWF1, mMacroF1, mUAR
                performance[str(measure)] = measure.calc(targets['emo'], predicts['emo'])
                if verbose:
                    self.logger.info('{0} for emotions:'.format(str(measure)))
                    self.logger.info(self.c_names['emo'])
                    self.logger.info(['{0:.3f}'.format(emo_score) for emo_score in measure.get_scores()])
            elif 'combined' in str(measure): # emo_sen_combined
                performance[str(measure)] = measure.calc(targets, predicts)
            elif 'emo' in str(measure):
                # Emotions A(WAR), UAR, WF1 и MacroF1
                performance[str(measure)] = measure.calc(targets['emo'], predicts['emo'])
            else:
                # Sentiment A(WAR), UAR, WF1 и MacroF1
                performance[str(measure)] = measure.calc(targets['sen'], predicts['sen'])
        
        if verbose:
            self.logger.info([measure for measure in performance])
            self.logger.info(['{0:.3f}'.format(performance[measure]) for measure in performance])

        return performance