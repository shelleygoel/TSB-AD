# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import math
import random
import sys
import tempfile

import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import random_split

from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.models.tspulse.utils.helpers import PatchMaskingDatasetWrapper
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import TimeSeriesAnomalyDetectionPipeline

from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.dataset import TSPulseFinetuneDataset

import random, argparse, time, os, logging
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

def attach_timestamp_column(
    df: pd.DataFrame, time_col: str = "timestamp", freq: str = "5s", start_date: str = "2002-01-01"
):
    n = df.shape[0]
    if time_col not in df:
        df[time_col] = pd.date_range(start_date, freq=freq, periods=n)
    return df


def get_dataset_name(filename: str):
    file_basename = os.path.basename(filename)
    return file_basename.split("_")[1]


def select_best_mode_by_dataset(filename: str, 
                                target_col: str = "file_name",
                                metric_col: str = "VUS-PR",
                                mode_col: str = "MODE",
                                greater_is_better: bool = True,
                                ):
    df = pd.read_csv(filename, sep=',', header='infer', index_col=None)
    performance = {}
    counter = {}
    for target_file, metric, mode in zip(df[target_col], df[metric_col], df[mode_col]):
        dset_name = get_dataset_name(target_file)
        if dset_name not in performance:
            performance[dset_name] = {}
            counter[dset_name] = {}
        
        if mode not in performance[dset_name]:
            performance[dset_name][mode] = 0
            counter[dset_name][mode] = 0
        
        performance[dset_name][mode] += float(metric)
        counter[dset_name][mode] += 1
    
    best_metric = {}
    for dset_name in performance:
        mode_names = []
        mode_performance = []
        for mode in performance[dset_name]:
            mode_names.append(mode)
            mode_performance.append(performance[dset_name][mode] / counter[dset_name][mode])
        if greater_is_better:
            index = np.argmax(mode_performance)
        else:
            index = np.argmin(mode_performance)
        best_metric[dset_name] = mode_names[index]
    return best_metric


class TSPulsePipeline(BaseDetector):
    def __init__(
        self,
        model_path: str | None = None,
        batch_size: int = 256,
        aggr_win_size: int = 96,
        num_input_channels: int = 1,
        smoothing_window: int = 8,
        prediction_mode: str = "time",
        finetune_epochs: int = 20,
        finetune_validation: float = 0.2,
        finetune_lr: float = 1e-4,
        finetune_seed: int = 42,
        finetune_freeze_backbone: bool = False,
        finetune_decoder_mode: str = "common_channel",
        **kwargs,
    ):
        self._batch_size = batch_size
        self._headers = [f"x{i + 1}" for i in range(num_input_channels)]

        if model_path is None:
            model_path = "ibm-granite/granite-timeseries-tspulse-r1"

        if num_input_channels == 1:
            finetune_decoder_mode = "common_channel"

        if finetune_decoder_mode is None:
            if num_input_channels > 1:
                self.decoder_mode = "mix_channel"
            else:
                self.decoder_mode = "common_channel"
        else:
            self.decoder_mode = finetune_decoder_mode
        # Setting random seed
        random.seed(finetune_seed)
        np.random.seed(finetune_seed)
        # Loading the model to memory
        self._model = TSPulseForReconstruction.from_pretrained(
            model_path,
            num_input_channels=num_input_channels,
            decoder_mode=self.decoder_mode,
            scaling="revin",
            mask_type="user",
        )
        # Reading patch length from the loaded model instance
        p_length = self._model.config.patch_length
        if (aggr_win_size < p_length) or (aggr_win_size % p_length != 0):
            raise ValueError(f"Error: aggregation window must be greater than and multiple of patch_length={p_length}")
        prediction_mode_array = [s_.strip() for s_ in str(prediction_mode).split("+")]
        
        # Storing pipeline configuration parameters
        self._pipeline_config = {
            "timestamp_column": "timestamp",
            "target_columns": self._headers.copy(),
            "prediction_mode": prediction_mode_array.copy(),
            "aggregation_length": aggr_win_size,
            "smoothing_window": smoothing_window,
            "least_significant_scale": 0.0,
            "least_significant_score": 1.0,
        }
        # Storing model finetuning parameters
        self._finetune_params = {
            "finetune_epochs": finetune_epochs,
            "finetune_validation": finetune_validation,
            "finetune_lr": finetune_lr,
            "finetune_seed": finetune_seed,
            "finetune_freeze_backbone": finetune_freeze_backbone,
        }
        self._scorer = TimeSeriesAnomalyDetectionPipeline(
            self._model,
            target_columns=self._pipeline_config.get("target_columns"),
            prediction_mode=prediction_mode_array,
            aggregation_length=aggr_win_size,
            smoothing_window=self._pipeline_config.get("smoothing_window"),
            least_significant_scale=self._pipeline_config.get("least_significant_scale"),
            least_significant_score=self._pipeline_config.get("least_significant_score"),
        )

    def zero_shot(self, x, label=None):
        self.decision_scores_ = self.decision_function(x)

    def fit(self, X, y=None):
        try:
            print("Fine-tuning TSPulse.")
            validation_size = float(self._finetune_params.get("finetune_validation", 0.2))
            create_valid = True
            if X.shape[0] < 3000:  # 20% of this should be > context_len
                print("Data too small to create a validation set.")
                create_valid = False
                validation_size = 0.0

            if X.shape[0] < self._model.config.context_length:
                print("Skipping fine-tuning due to very short length")
                return

            tsTrain = X[: int((1 - validation_size) * len(X))]
            context_length = self._model.config.context_length
            train_dataset = PatchMaskingDatasetWrapper(
                TSPulseFinetuneDataset(tsTrain, window_size=context_length, return_dict=True),
                window_length=self._pipeline_config.get("aggregation_length"),
                patch_length=self._model.config.patch_length,
                window_position="last",
            )
            if len(train_dataset) < 100:
                print("Skipping fine-tuning due to very few training samples")
                return

            if create_valid:
                tsValid = X[int((1 - validation_size) * len(X)) :]
                valid_dataset = PatchMaskingDatasetWrapper(
                    TSPulseFinetuneDataset(tsValid, window_size=context_length, return_dict=True),
                    window_length=self._pipeline_config.get("aggregation_length"),
                    patch_length=self._model.config.patch_length,
                    window_position="last",
                )
            else:
                valid_dataset = train_dataset

            max_finetune_samples = 100_000
            if len(train_dataset) > max_finetune_samples:
                use_fraction = max_finetune_samples / len(train_dataset)
                # Randomly select use_fraction samples to make finetuning faster
                train_dataset, _ = random_split(train_dataset, [use_fraction, 1 - use_fraction])
                valid_dataset, _ = random_split(valid_dataset, [use_fraction, 1 - use_fraction])
                print(
                    f"Training samples are > max_finetune_samples ({max_finetune_samples}), using {round(use_fraction * 100)}% for faster fine-tuning."
                )

            freeze_backbone = self._finetune_params.get("finetune_freeze_backbone")
            # Freeze the backbone
            if freeze_backbone:
                # Freeze the backbone of the model
                for param in self._model.backbone.parameters():
                    param.requires_grad = False

            temp_dir = tempfile.mkdtemp()

            suggested_lr = self._finetune_params.get("finetune_lr", 1e-4)
            finetune_num_epochs: int = int(self._finetune_params.get("finetune_epochs", 20))
            if not create_valid:
                finetune_num_epochs = min(5, finetune_num_epochs)

            finetune_batch_size = self._batch_size
            if len(train_dataset) < 500:
                finetune_batch_size = 8
            num_workers = 4
            num_gpus = 1

            print(f"Fine-tune: Train samples = {len(train_dataset)}, Valid Samples = {len(valid_dataset)}")

            finetune_args = TrainingArguments(
                output_dir=temp_dir,
                overwrite_output_dir=True,
                learning_rate=suggested_lr,
                num_train_epochs=finetune_num_epochs,
                do_eval=True,
                eval_strategy="epoch",
                per_device_train_batch_size=finetune_batch_size,
                per_device_eval_batch_size=finetune_batch_size * 10,
                dataloader_num_workers=num_workers,
                report_to="tensorboard",
                save_strategy="epoch",
                logging_strategy="epoch",
                save_total_limit=1,
                logging_dir=temp_dir,  # Make sure to specify a logging directory
                load_best_model_at_end=True,  # Load the best model when training ends
                metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
                greater_is_better=False,  # For loss
            )

            # Create the early stopping callback
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=5,  # Number of epochs with no improvement after which to stop
                early_stopping_threshold=1e-5,  # Minimum improvement required to consider as improvement
            )

            # Optimizer and scheduler
            optimizer = AdamW(self._model.parameters(), lr=suggested_lr)
            scheduler = OneCycleLR(
                optimizer,
                suggested_lr,
                epochs=finetune_num_epochs,
                steps_per_epoch=math.ceil(len(train_dataset) / (finetune_batch_size * num_gpus)),
            )

            finetune_trainer = Trainer(
                model=self._model,
                args=finetune_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                callbacks=[early_stopping_callback],
                optimizers=(optimizer, scheduler),
            )

            # Fine tune
            finetune_trainer.train()

        except Exception as e:
            print("Error occured in finetune. Error =", e)
            sys.exit(-1)

    def decision_function(self, X):
        """
        Not used, present for API consistency by convention.
        """
        data = attach_timestamp_column(pd.DataFrame(X, columns=self._headers))
        score = self._scorer(data, batch_size=self._batch_size)
        if not isinstance(score, pd.DataFrame) or ("anomaly_score" not in score):
            raise ValueError("Error: expect anomaly_score column in the output!")

        score = score["anomaly_score"].values.ravel()
        norm_value = np.nanmax(np.asarray(score), axis=0, keepdims=True) + 1e-5
        anomaly_score = score / norm_value
        return anomaly_score


def run_TSPulse_Unsupervised(data, **HP):
    num_input_channels = data.shape[1]
    clf = TSPulsePipeline(
            model_path="ibm-granite/granite-timeseries-tspulse-r1",
            num_input_channels=num_input_channels,
            batch_size=256,
            aggr_win_size=HP.get('win_size', 96),
            smoothing_window=8,
            prediction_mode=HP.get('prediction_mode', 'time'),
        )
    score = clf.decision_function(data)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score


def run_TSPulse_Semisupervised(data_train, data_test, **HP):
    num_input_channels = data_train.shape[1]

    clf = TSPulsePipeline(
            model_path="ibm-granite/granite-timeseries-tspulse-r1",
            num_input_channels=num_input_channels,
            batch_size=256,
            aggr_win_size=HP.get('win_size', 96),
            smoothing_window=8,
            prediction_mode=HP.get('prediction_mode', 'time'),
            finetune_decoder_mode='common_channel',
            finetune_validation=0.2,
            finetune_freeze_backbone=False,
            finetune_epochs=20,
            finetune_lr=1e-4,
        )
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score



if __name__ == '__main__':
    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running TSPulse')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='../Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='TSPulse')
    
    # Optional argument to enable mode selection for TSPulse
    parser.add_argument('--tuning_results', 
                        type=str, 
                        default=None, 
                        help="Provide the resource file with tuning data experiment results. "
                             "Required by TSPulse algorithm for optimal data dependent mode selection.")
    parser.add_argument('--prediction_mode', 
                        type=str, 
                        default='#', 
                        help="Mode specification for TSPulse algorithm.")
    
    args = parser.parse_args()

    Custom_AD_HP = {
        'win_size': 96, 
        'prediction_mode': 'time'
    }
    
    if args.prediction_mode != "#":
        Custom_AD_HP['prediction_mode'] = args.prediction_mode
    elif (args.tuning_results is not None) and os.path.isfile(args.tuning_results):
        lookup = select_best_mode_by_dataset(args.tuning_results)
        dset_name = get_dataset_name(args.filename)
        Custom_AD_HP['prediction_mode'] = lookup.get(dset_name, 'time')

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data, rank=1)
    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]

    start_time = time.time()

    # output = run_TSPulse_Semisupervised(data_train, data, **Custom_AD_HP)
    output = run_TSPulse_Unsupervised(data, **Custom_AD_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output)+3*np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)

