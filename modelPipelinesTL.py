import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_score, recall_score, f1_score
# REGRESSION: added for R3a
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tqdm
import os
import datetime
import yaml
import json
import tensorflow as tf

logger = logging.getLogger(__name__)

from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Conv1D, MaxPooling1D, GlobalAveragePooling1D, RepeatVector
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from abc import ABC, abstractmethod
import pyarrow.parquet as pq
import gc
from tensorflow.keras import backend as K

class ModelPipeline(ABC):
    def __init__(self, config):
        self.config = config
        self.output_parent_dir = os.path.abspath(os.path.expanduser(self.config.get('output_parent_dir', '.output')))
        os.makedirs(self.output_parent_dir, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.labels_train = None
        self.labels_test = None
        self.y_test = None # For autoencoder

    def loadData(self, inputPath):
        """
        Loads data from a Parquet file with large array columns memory-efficiently.
        It reads metadata columns first, then loads large signal columns one by one.
        """
        logger.info(f"Loading data from {inputPath}...")

        try:
            parquet_file = pq.ParquetFile(inputPath)

            # Get all column names from the file's schema
            all_columns = parquet_file.schema.names

            # Identify metadata columns (all columns that are NOT signals)
            signal_cols = self.config.get('signalColumns', [])
            metadata_cols = [col for col in all_columns if col not in signal_cols]

            # 1. Load only the small metadata columns first
            if metadata_cols:
                dataDF = parquet_file.read(columns=metadata_cols).to_pandas()
            else:
                # If no metadata, create an empty DataFrame with the right number of rows
                num_rows = parquet_file.metadata.num_rows
                dataDF= pd.DataFrame(index=range(num_rows))

            # 2. Load each large signal column individually and add it to the DataFrame
            for col in signal_cols:
                signal_series = parquet_file.read(columns=[col]).to_pandas()[col]
                dataDF[col] = signal_series

            logger.info(f"Successfully loaded {len(dataDF)} rows from {inputPath}")

        except Exception as e:
            logger.error(f"An error occurred during data loading: {e}")
            dataDF = None
        
        return dataDF

    def setWearTH(self, data_DF, wearTH=0.45, wearColumnName='wear_norm'):
        data_DF_new = data_DF.copy()
        data_DF_new['wear_class'] = np.where(data_DF_new[wearColumnName] > wearTH, 1, 0)
        return data_DF_new

    def reformatData(self, data_DF, signalColumns=['vibSpindle', 'vibTable'], labelColumn='wear_class', extract_labels=True):
        sequence_data = [
            np.stack([row[col] for col in signalColumns], axis=-1)
            for _, row in tqdm.tqdm(data_DF.iterrows(), total=len(data_DF), desc='Extracting sequences')
        ]
        
        labels = None
        if extract_labels:
            if labelColumn not in data_DF.columns:
                raise KeyError(f"Label column '{labelColumn}' not found in DataFrame. Cannot extract labels.")
            labels = data_DF[labelColumn].to_numpy()

        logger.info(f"Using signals: {signalColumns}, {len(sequence_data)} sequences extracted")
            
        return np.array(sequence_data, dtype=object), labels

    def rectangularSequenceData(self, sequence_data, signal_length=9000, pooling_type='mean'):
        num_samples = len(sequence_data)
        if num_samples == 0:
            return np.array([])
        num_signals = sequence_data[0].shape[1]
        rectangular = np.zeros((num_samples, signal_length, num_signals), dtype=sequence_data[0].dtype)
        for i, seq in enumerate(tqdm.tqdm(sequence_data, desc='Processing sequences')):
            seq_len = seq.shape[0]
            if seq_len < signal_length:
                padded = np.zeros((signal_length, num_signals), dtype=seq.dtype)
                padded[:seq_len, :] = seq
                rectangular[i] = padded
            else:
                window_size = seq_len // signal_length
                if window_size < 1:
                    rectangular[i] = seq[:signal_length, :]
                else:
                    trimmed_len = window_size * signal_length
                    trimmed_seq = seq[:trimmed_len, :]
                    reshaped = trimmed_seq.reshape(signal_length, window_size, num_signals)
                    if pooling_type == 'mean':
                        pooled = reshaped.mean(axis=1)
                    elif pooling_type == 'max':
                        pooled = reshaped.max(axis=1)
                    else:
                        raise ValueError("Unsupported pooling_type: choose 'mean' or 'max'")
                    rectangular[i] = pooled
        return rectangular

    def normalize(self, sequence_train, sequence_test):
        # NOTE: For inference, sequence_train can be a dummy or empty array, as the scaler is fitted on training data only.
        # For simplicity in this implementation, we assume that during inference, the training set from the original training is used to fit the scaler.
        # A more robust implementation would save the scaler object from training and load it for inference.
        X_train = np.array(sequence_train, dtype=np.float32)
        X_test = np.array(sequence_test, dtype=np.float32)
        
        # Only fit the scaler on training data if it's available
        if X_train.shape[0] > 0:
            nsamples, nx, ny = X_train.shape
            X_train_2d = X_train.reshape((nsamples * nx, ny))

            self.scaler = StandardScaler() ## NEW: Store scaler as an instance attribute
            self.scaler.fit(X_train_2d)

            X_train_scaled_2d = self.scaler.transform(X_train_2d)
            self.X_train = X_train_scaled_2d.reshape(nsamples, nx, ny)
        else: # If no training data (e.g., pure inference mode), just prepare the test set
             self.X_train = X_train

        # Always transform the test data
        if X_test.shape[0] > 0:
            nsamples_test, nx_test, ny_test = X_test.shape
            X_test_2d = X_test.reshape((nsamples_test * nx_test, ny_test))
            X_test_scaled_2d = self.scaler.transform(X_test_2d) # Use the fitted scaler
            self.X_test = X_test_scaled_2d.reshape(nsamples_test, nx_test, ny_test)
        else:
            self.X_test = X_test

        logger.info("Data has been scaled.")


    def prepare_data(self, train_case_ids=None, test_case_ids=None, scaler_data_path=None, test_data_path=None):
        # REGRESSION: added for R3a
        task_type = self.config.get('task_type', 'classification')

        if scaler_data_path is not None:
            # --- MODE 1: INFERENCE ---
            logger.info("Preparing data in INFERENCE mode.")
            scaler_full_DF = self.loadData(scaler_data_path)
            train_DF, _ = self.splitDataByColumn(scaler_full_DF, 'CaseID', train_case_ids, None)
            if not train_DF.empty:
                sequence_train_raw, _ = self.reformatData(train_DF, signalColumns=self.config['signalColumns'], extract_labels=False)
                sequence_train = self.rectangularSequenceData(sequence_train_raw, signal_length=self.config['signal_length'], pooling_type=self.config['pooling_type'])
            else:
                sequence_train = np.array([])
            self.labels_train = np.array([])

            test_full_DF = self.loadData(test_data_path)
            # REGRESSION: skip binarization for regression
            if task_type == 'classification':
                test_full_DF = self.setWearTH(test_full_DF, wearTH=self.config['wearTH'], wearColumnName=self.config['wearColumnName'])
            else:
                label_col = self.config.get('labelColumn', 'wear_norm')
                before = len(test_full_DF)
                test_full_DF = test_full_DF.dropna(subset=[label_col])
                dropped = before - len(test_full_DF)
                if dropped:
                    logger.info(f"Regression: dropped {dropped} rows with NaN in '{label_col}' (inference)")
            _, test_DF = self.splitDataByColumn(test_full_DF, 'CaseID', None, test_case_ids)
            if not test_DF.empty:
                sequence_test_raw, self.labels_test = self.reformatData(test_DF, signalColumns=self.config['signalColumns'], labelColumn=self.config['labelColumn'])
                if self.labels_test is not None and self.labels_test.size > 0:
                    # REGRESSION: skip to_categorical for regression
                    if task_type == 'classification':
                        self.labels_test = to_categorical(self.labels_test)
                    else:
                        self.labels_test = self.labels_test.astype(np.float32)
                sequence_test = self.rectangularSequenceData(sequence_test_raw, signal_length=self.config['signal_length'], pooling_type=self.config['pooling_type'])
            else:
                sequence_test, self.labels_test = np.array([]), np.array([])
        else:
            # --- MODE 2: TRAINING or TRANSFER LEARNING ---
            logger.info("Preparing data in TRAINING/TRANSFER LEARNING mode.")
            data_path = test_data_path if test_data_path is not None else self.config['inputPath']
            full_DF = self.loadData(data_path)
            # REGRESSION: skip binarization for regression
            if task_type == 'classification':
                full_DF = self.setWearTH(full_DF, wearTH=self.config['wearTH'], wearColumnName=self.config['wearColumnName'])
            else:
                # Drop rows with NaN in the label column (classification handles these via np.where; regression cannot)
                label_col = self.config.get('labelColumn', 'wear_norm')
                before = len(full_DF)
                full_DF = full_DF.dropna(subset=[label_col])
                dropped = before - len(full_DF)
                if dropped:
                    logger.info(f"Regression: dropped {dropped} rows with NaN in '{label_col}'")
            train_DF, test_DF = self.splitDataByColumn(full_DF, 'CaseID', train_case_ids, test_case_ids)

            if not train_DF.empty:
                sequence_train_raw, self.labels_train = self.reformatData(train_DF, signalColumns=self.config['signalColumns'], labelColumn=self.config['labelColumn'])
                if self.labels_train is not None and self.labels_train.size > 0:
                    # REGRESSION: skip to_categorical for regression
                    if task_type == 'classification':
                        self.labels_train = to_categorical(self.labels_train)
                    else:
                        self.labels_train = self.labels_train.astype(np.float32)
                sequence_train = self.rectangularSequenceData(sequence_train_raw, signal_length=self.config['signal_length'], pooling_type=self.config['pooling_type'])
            else:
                sequence_train, self.labels_train = np.array([]), np.array([])

            if not test_DF.empty:
                sequence_test_raw, self.labels_test = self.reformatData(test_DF, signalColumns=self.config['signalColumns'], labelColumn=self.config['labelColumn'])
                if self.labels_test is not None and self.labels_test.size > 0:
                    # REGRESSION: skip to_categorical for regression
                    if task_type == 'classification':
                        self.labels_test = to_categorical(self.labels_test)
                    else:
                        self.labels_test = self.labels_test.astype(np.float32)
                sequence_test = self.rectangularSequenceData(sequence_test_raw, signal_length=self.config['signal_length'], pooling_type=self.config['pooling_type'])
            else:
                sequence_test, self.labels_test = np.array([]), np.array([])
        
        # Finally, normalize the data. For transfer learning, the scaler is fit on the new training data.
        self.normalize(sequence_train, sequence_test)

    def splitDataByColumn(self, data_DF, column, train_values, test_values):
        train_DF = pd.DataFrame()
        test_DF = pd.DataFrame()
        if train_values:
            train_DF = data_DF[data_DF[column].isin(train_values)].copy()
        if test_values:
            test_DF = data_DF[data_DF[column].isin(test_values)].copy()
        
        logger.info(f"Train set: {len(train_DF)} samples, Test set: {len(test_DF)} samples")
        return train_DF, test_DF

    @abstractmethod
    def modelSetup(self):
        pass

    def trainModel(self, is_transfer=False): ## NEW: Added flag for transfer learning
        log_dir_suffix = "transfer" if is_transfer else "initial"
        log_dir = os.path.join(self.output_parent_dir, 'logs', 'tensorboard', f"{self.config['model_type']}_{log_dir_suffix}")
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        self.history = self.model.fit(
            self.X_train,
            self.labels_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=self.config.get('validation_split', 0.2),
            verbose=self.config['verbose'],
            callbacks=[tensorboard_callback]
        )
        
        output_dir = os.path.join(self.output_parent_dir, 'models')
        os.makedirs(output_dir, exist_ok=True)
        model_name_suffix = "transfer" if is_transfer else "model"
        model_path = os.path.join(output_dir, f"{self.config['model_type']}_{model_name_suffix}_{self.timestamp}")
        self.model.save(model_path+'.keras')
        self.model.export(model_path)
        logger.info(f"Model saved to {model_path}")

    def evalModel(self, history=None):
        if history:
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            # Save the plot to a file in the logs directory
            log_dir = os.path.join(self.config['output_parent_dir'], 'logs')
            os.makedirs(log_dir, exist_ok=True)
            plot_path = os.path.join(log_dir, f"loss_plot_{self.config.get('model_type', 'model')}_{self.timestamp}.png")
            plt.savefig(plot_path)
            plt.close()  # Close the plot to avoid displaying it in the notebook
            logger.info(f"Loss plot saved to {plot_path}")
        else:
            logger.info("No history object provided, skipping loss plot.")

        # Predict on the test data
        reconstructions = self.model.predict(self.X_test)
        # Calculate reconstruction error (e.g., Mean Absolute Error)
        recon_error = np.mean(np.abs(reconstructions - self.X_test), axis=(1, 2))
        
        # Create a DataFrame with true labels and reconstruction errors
        results_df = pd.DataFrame({
            'y_true': self.y_test_classes,
            'reconstruction_error': recon_error
        })
        
        # Save the results to a CSV file
        log_dir = os.path.join(self.config['output_parent_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f"autoencoder_evaluation_{self.timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Autoencoder evaluation results saved to {csv_path}")

        return results_df

    def save_config(self):
        config_dir = os.path.join(self.output_parent_dir, 'configs')
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, f"{self.config['model_type']}_config_{self.timestamp}.yaml")
        
        serializable_config = {}
        for key, value in self.config.items():
            if isinstance(value, (np.ndarray, np.generic)):
                serializable_config[key] = value.tolist()
            else:
                serializable_config[key] = value

        with open(config_path, 'w') as f:
            yaml.dump(serializable_config, f, default_flow_style=False)
        logger.info(f"Config saved to {config_path}")

    # REGRESSION: added for R3a — regression evaluation method
    def evalModelRegression(self, mode="regression_eval"):
        y_pred = self.model.predict(self.X_test).flatten()
        y_true = self.labels_test.flatten()

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        results_data = {
            'mode': [mode],
            'Train Cases': [self.config.get('train_caseIDs', [])],
            'Test Cases': [self.config.get('test_caseIDs', [])],
            'MAE': [mae],
            'RMSE': [rmse],
            'R2': [r2],
        }
        results_df = pd.DataFrame(results_data)

        output_dir = os.path.join(self.output_parent_dir, 'logs')
        os.makedirs(output_dir, exist_ok=True)
        csv_filename = f"evaluation_regression_{mode}_{self.config['model_type']}_{self.timestamp}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)
        results_df.to_csv(csv_filepath, index=False)
        logger.info(f"Regression results saved to {csv_filepath}")
        logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # In modelPipelines.py, inside the ModelPipeline class

    def run(self):
        """
        Executes the pipeline for a single task based on the 'mode' in the config.
        This is designed to be called by run.py in a separate process for each task.
        """
        mode = self.config.get('mode', 'train')
        task_type = self.config.get('task_type', 'classification')  # REGRESSION: added for R3a

        if mode == 'train':
            logger.info("Running in TRAINING mode.")
            self.prepare_data(self.config['train_caseIDs'], self.config['test_caseIDs'])
            self.modelSetup()
            self.save_config()
            self.trainModel()
            # REGRESSION: added for R3a
            if task_type == 'regression':
                self.evalModelRegression(mode='train')
            else:
                self.evalModel()
            
        elif mode == 'inference':
            logger.info("Running in INFERENCE mode.")
            # The config will only have one task because run_all_tasks.py creates a temporary config for each process
            if not self.config.get('inference_tasks'):
                 raise ValueError("'inference_tasks' list is empty or missing in the config.")
            task = self.config['inference_tasks'][0] 
            self.run_inference(task)
            
        elif mode == 'transfer_learn':
            logger.info("Running in TRANSFER LEARNING mode.")
            if not self.config.get('transfer_learning_tasks'):
                 raise ValueError("'transfer_learning_tasks' list is empty or missing in the config.")
            task = self.config['transfer_learning_tasks'][0]
            self.run_transfer_learning(task)
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'train', 'inference', 'transfer_learn'.")

    ## ----------------------------------------------------------------
    ## NEW: Inference Method
    ## ----------------------------------------------------------------
    def run_inference(self, inference_task):
        logger.info(f"Starting inference on {inference_task['inputPath']}")

        # 1. Load the pre-trained model
        logger.info(f"Loading model from: {inference_task['model_path']}")
        self.model = tf.keras.models.load_model(inference_task['model_path']+'.keras')
        self.model.summary()

        # --- SOLUTION: Clearly define the two separate data sources ---
        
        # Source 1: Data for fitting the scaler (from top-level config)
        scaler_data_path = self.config['inputPath']
        scaler_cases = self.config['train_caseIDs']
        
        # Source 2: Data for testing/inference (from the specific task)
        test_data_path = inference_task['inputPath']
        test_cases = inference_task['test_caseIDs']
        
        # 2. Prepare data using the two sources
        self.prepare_data(
            train_case_ids=scaler_cases,
            test_case_ids=test_cases,
            scaler_data_path=scaler_data_path,
            test_data_path=test_data_path
        )

        # 3. Evaluate the model on the new data
        self.evalModel(mode=f"inference_on_{os.path.basename(inference_task['inputPath']).split('.')[0]}")

        logger.info("Inference finished.")

    ## ----------------------------------------------------------------
    ## NEW: Transfer Learning Method
    ## ----------------------------------------------------------------
    def run_transfer_learning(self, tl_task):
        logger.info(f"Starting transfer learning for {tl_task['inputPath']}")

        # We only provide the path to the new dataset.
        self.prepare_data(
            train_case_ids=tl_task['train_caseIDs'],
            test_case_ids=tl_task['test_caseIDs'],
            test_data_path=tl_task['inputPath'] # This is the only path needed.
        )
        
        logger.info(f"Loading base model from: {tl_task['model_path']}")
        source_model = tf.keras.models.load_model(tl_task['model_path']+'.keras')
        
        try:
            base_model = source_model.get_layer('feature_extractor_base')
        except ValueError:
            # ... (generic base model reconstruction logic) ...
            last_base_layer = None
            for layer in reversed(source_model.layers):
                if not isinstance(layer, (Dense, Dropout)):
                    last_base_layer = layer
                    break
            base_model = Model(inputs=source_model.input, outputs=last_base_layer.output)

        base_model.trainable = False

        # This will now work because self.labels_train is correctly shaped
        transfer_head = self.create_transfer_head(base_model.output)
        self.model = Model(inputs=base_model.input, outputs=transfer_head)

        # REGRESSION: added for R3a — use MSE for regression tasks
        task_type = self.config.get('task_type', 'classification')
        if task_type == 'regression':
            self.model.compile(optimizer=Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])
        else:
            self.model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

        self.trainModel(is_transfer=True)
        # REGRESSION: added for R3a
        transfer_label = f"transfer_on_{os.path.basename(tl_task['inputPath']).split('.')[0]}"
        if task_type == 'regression':
            self.evalModelRegression(mode=transfer_label)
        else:
            self.evalModel(mode=transfer_label)
        
        logger.info("Transfer learning finished.")


    ## ----------------------------------------------------------------
    ## NEW: Helper to create a transfer head
    ## ----------------------------------------------------------------
    def create_transfer_head(self, base_output):
        """Creates a new head for transfer learning (classification or regression)."""
        # REGRESSION: added for R3a — branch on task_type
        task_type = self.config.get('task_type', 'classification')
        x = Dropout(0.5)(base_output)
        x = Dense(32, activation='relu', name='transfer_dense_1')(x)
        if task_type == 'regression':
            outputs = Dense(1, activation='linear', name='transfer_output')(x)
        else:
            n_classes = self.labels_train.shape[1] if self.labels_train.ndim > 1 and self.labels_train.shape[1] > 0 else 2
            outputs = Dense(n_classes, activation='softmax', name='transfer_output')(x)
        return outputs



## ====================================================================
## MODIFIED MODEL CLASSES
## ====================================================================

class LSTMPipeline(ModelPipeline):
    def evalModel(self, mode='train'):
        loss, accuracy = self.model.evaluate(self.X_test, self.labels_test, verbose=0)
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.labels_test, axis=1)
        precision = precision_score(y_true_classes, y_pred_classes, zero_division=0)
        recall = recall_score(y_true_classes, y_pred_classes, zero_division=0)
        f1 = f1_score(y_true_classes, y_pred_classes, zero_division=0)
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        results_df = pd.DataFrame({
            'mode': [mode],
            'Train Cases': [self.config.get('train_caseIDs', [])],
            'Test Cases': [self.config.get('test_caseIDs', [])],
            'Accuracy': [accuracy], 'Loss': [loss],
            'Precision': [precision], 'Recall': [recall], 'F1-Score': [f1],
            'confusion_matrix': [cm.tolist()],
        })
        output_dir = os.path.join(self.output_parent_dir, 'logs')
        os.makedirs(output_dir, exist_ok=True)
        csv_filepath = os.path.join(output_dir, f"evaluation_results_{mode}_{self.config['model_type']}_{self.timestamp}.csv")
        results_df.to_csv(csv_filepath, index=False)
        logger.info(f"Evaluation results saved to {csv_filepath}")
        logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    def modelSetup(self):
        # REGRESSION: added for R3a — branch on task_type
        task_type = self.config.get('task_type', 'classification')
        inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))

        # --- Base Model (Feature Extractor) ---
        base_layers = LSTM(64, return_sequences=True)(inputs)
        base_output = LSTM(32, return_sequences=False)(base_layers)
        self.base_model = Model(inputs=inputs, outputs=base_output, name="feature_extractor_base")
        self.base_model.trainable = True

        # --- Head ---
        head_layers = Dense(16, activation='tanh')(self.base_model.output)
        if task_type == 'regression':
            outputs = Dense(1, activation='linear')(head_layers)
            self.model = Model(inputs=self.base_model.input, outputs=outputs)
            self.model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
        else:
            outputs = Dense(self.labels_train.shape[1], activation='softmax')(head_layers)
            self.model = Model(inputs=self.base_model.input, outputs=outputs)
            self.model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

class Conv1DPipeline(ModelPipeline):
    def evalModel(self, mode='train'):
        loss, accuracy = self.model.evaluate(self.X_test, self.labels_test, verbose=0)
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.labels_test, axis=1)
        precision = precision_score(y_true_classes, y_pred_classes, zero_division=0)
        recall = recall_score(y_true_classes, y_pred_classes, zero_division=0)
        f1 = f1_score(y_true_classes, y_pred_classes, zero_division=0)
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        results_df = pd.DataFrame({
            'mode': [mode],
            'Train Cases': [self.config.get('train_caseIDs', [])],
            'Test Cases': [self.config.get('test_caseIDs', [])],
            'Accuracy': [accuracy], 'Loss': [loss],
            'Precision': [precision], 'Recall': [recall], 'F1-Score': [f1],
            'confusion_matrix': [cm.tolist()],
        })
        output_dir = os.path.join(self.output_parent_dir, 'logs')
        os.makedirs(output_dir, exist_ok=True)
        csv_filepath = os.path.join(output_dir, f"evaluation_results_{mode}_{self.config['model_type']}_{self.timestamp}.csv")
        results_df.to_csv(csv_filepath, index=False)
        logger.info(f"Evaluation results saved to {csv_filepath}")
        logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    def modelSetup(self):
        # REGRESSION: added for R3a — branch on task_type
        task_type = self.config.get('task_type', 'classification')
        inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))

        # --- Base Model (Feature Extractor) ---
        base_layers = Conv1D(filters=32, kernel_size=5, activation='relu')(inputs)
        base_layers = MaxPooling1D(pool_size=2)(base_layers)
        base_layers = Conv1D(filters=64, kernel_size=5, activation='relu')(base_layers)
        base_output = GlobalAveragePooling1D()(base_layers)
        self.base_model = Model(inputs=inputs, outputs=base_output, name="feature_extractor_base")
        self.base_model.trainable = True

        # --- Head ---
        head_layers = Dropout(0.5)(self.base_model.output)
        head_layers = Dense(32, activation='relu')(head_layers)
        if task_type == 'regression':
            outputs = Dense(1, activation='linear')(head_layers)
            self.model = Model(inputs=self.base_model.input, outputs=outputs)
            self.model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
        else:
            outputs = Dense(self.labels_train.shape[1], activation='softmax')(head_layers)
            self.model = Model(inputs=self.base_model.input, outputs=outputs)
            self.model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        
# The AutoencoderPipeline remains unchanged as the request was for supervised models
class AutoencoderPipeline(ModelPipeline):
    def prepare_data(self):
        data_DF = self.loadData(self.config['inputPath'])
        data_DF = self.setWearTH(data_DF, wearTH=self.config['wearTH'], wearColumnName=self.config['wearColumnName'])
        
        sequence_data_raw, labels = self.reformatData(data_DF, signalColumns=self.config['signalColumns'], labelColumn=self.config['labelColumn'])
        
        sequence_data = self.rectangularSequenceData(sequence_data_raw, signal_length=self.config['signal_length'], pooling_type=self.config['pooling_type'])

        healthy_sequences = sequence_data[labels == 0]
        unhealthy_sequences = sequence_data[labels == 1]

        train_size = int(self.config['train_split_ratio'] * len(healthy_sequences))
        self.X_train = healthy_sequences[:train_size]
        X_test_healthy = healthy_sequences[train_size:]
        
        self.X_test = np.concatenate((X_test_healthy, unhealthy_sequences), axis=0)
        self.y_test = np.concatenate([np.zeros(len(X_test_healthy)), np.ones(len(unhealthy_sequences))])

        # Downsample, clip and scale
        self.X_train = self.X_train[:, ::self.config['downsample_rate'], :]
        self.X_test = self.X_test[:, ::self.config['downsample_rate'], :]
        self.X_train = np.clip(self.X_train, self.config['clip_range'][0], self.config['clip_range'][1])
        self.X_test = np.clip(self.X_test, self.config['clip_range'][0], self.config['clip_range'][1])
        
        # We need a scaler for the autoencoder as well.
        # Fit on healthy training data, transform both train and test.
        self.scaler = StandardScaler()
        nsamples, nx, ny = self.X_train.shape
        self.X_train = self.scaler.fit_transform(self.X_train.reshape(-1, ny)).reshape(nsamples, nx, ny)

        nsamples_test, nx_test, ny_test = self.X_test.shape
        self.X_test = self.scaler.transform(self.X_test.reshape(-1, ny_test)).reshape(nsamples_test, nx_test, ny_test)

    def modelSetup(self):
        timesteps = self.X_train.shape[1]
        input_dim = self.X_train.shape[2]

        self.model = Sequential(
            [
                LSTM(64, activation='tanh', return_sequences=True, input_shape=(timesteps, input_dim)),
                Dropout(0.2),
                LSTM(32, activation='tanh', return_sequences=False),
                Dense(16, activation='relu'), # Latent space
                RepeatVector(timesteps),
                LSTM(32, activation='tanh', return_sequences=True),
                Dropout(0.2),
                LSTM(64, activation='tanh', return_sequences=True),
                Dense(input_dim, activation='linear')
            ]
        )
        self.model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
        self.model.summary()

    def trainModel(self):
        log_dir = os.path.join(self.output_parent_dir, 'logs', 'tensorboard', self.config['model_type'])
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

        self.history = self.model.fit(
            self.X_train, self.X_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=self.config['validation_split'],
            callbacks=[tensorboard_callback, lr_scheduler],
            verbose=self.config['verbose']
        )
        output_dir = os.path.join(self.output_parent_dir, 'models')
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f'autoencoder_model_{self.timestamp}')
        self.model.save(model_path+'.keras')
        self.model.export(model_path)
        logger.info(f"Model saved to {model_path}")

    def evalModel(self, history=None):
        if history:
            plt.figure(figsize=(15, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            
            output_dir = os.path.join(self.output_parent_dir, 'logs')
            os.makedirs(output_dir, exist_ok=True)
            filename = f"autoencoder_loss_{self.timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            logger.info(f"Training loss plot saved to {filepath}")
        else:
            logger.info("No history object provided, skipping loss plot.")

        reconstruction = self.model.predict(self.X_test)
        reconstruction_error = np.mean(np.square(self.X_test - reconstruction), axis=(1, 2))

        # Save reconstruction errors to CSV
        log_dir = os.path.join(self.config['output_parent_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        results_df = pd.DataFrame({'reconstruction_error': reconstruction_error, 'y_true': self.y_test})
        csv_path = os.path.join(log_dir, f"autoencoder_evaluation_{self.timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Autoencoder evaluation results saved to {csv_path}")

        fpr, tpr, thresholds = roc_curve(self.y_test, reconstruction_error)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")

        plt.subplot(1, 2, 2)
        plt.hist(reconstruction_error[self.y_test==0], bins=50, label='Healthy', alpha=0.7)
        plt.hist(reconstruction_error[self.y_test==1], bins=50, label='Worn', alpha=0.7)
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Reconstruction Error Distribution')
        plt.legend()
        
        plt.tight_layout()
        output_dir = os.path.join(self.output_parent_dir, 'logs') # Save plot in logs dir
        filename_roc = f"autoencoder_eval_{self.timestamp}.png"
        filepath_roc = os.path.join(output_dir, filename_roc)
        plt.savefig(filepath_roc)
        plt.close()
        logger.info(f"Evaluation plot saved to {filepath_roc}")
        logger.info(f"AUC: {roc_auc:.4f}")