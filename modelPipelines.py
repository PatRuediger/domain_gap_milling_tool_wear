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

logger = logging.getLogger(__name__)
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, Input, LSTM, Conv1D, MaxPooling1D, Dropout, RepeatVector
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
from abc import ABC, abstractmethod
import pyarrow.parquet as pq

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

    def reformatData(self, data_DF, signalColumns=['vibSpindle', 'vibTable'], labelColumn='wear_class'):
        sequence_data = [
            np.stack([row[col] for col in signalColumns], axis=-1)
            for _, row in tqdm.tqdm(data_DF.iterrows(), total=len(data_DF), desc='Extracting sequences')
        ]
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
        X_train = np.array(sequence_train, dtype=np.float32)
        X_test = np.array(sequence_test, dtype=np.float32)
        
        nsamples, nx, ny = X_train.shape
        X_train_2d = X_train.reshape((nsamples * nx, ny))

        scaler = StandardScaler()
        scaler.fit(X_train_2d)

        X_train_scaled_2d = scaler.transform(X_train_2d)
        self.X_train = X_train_scaled_2d.reshape(nsamples, nx, ny)

        nsamples_test, nx_test, ny_test = X_test.shape
        X_test_2d = X_test.reshape((nsamples_test * nx_test, ny_test))
        X_test_scaled_2d = scaler.transform(X_test_2d)
        self.X_test = X_test_scaled_2d.reshape(nsamples_test, nx_test, ny_test)

        logger.info("Data has been scaled using a scaler fit on the training set.")

    def prepare_data(self):
        data_DF = self.loadData(self.config['inputPath'])
        # REGRESSION: added for R3a — skip binarization for regression task
        task_type = self.config.get('task_type', 'classification')
        if task_type == 'classification':
            data_DF = self.setWearTH(data_DF, wearTH=self.config['wearTH'], wearColumnName=self.config['wearColumnName'])

        train_DF, test_DF = self.splitDataByColumn(data_DF, 'CaseID', self.config['train_caseIDs'], self.config['test_caseIDs'])

        sequence_train_raw, self.labels_train = self.reformatData(train_DF, signalColumns=self.config['signalColumns'], labelColumn=self.config['labelColumn'])
        sequence_test_raw, self.labels_test = self.reformatData(test_DF, signalColumns=self.config['signalColumns'], labelColumn=self.config['labelColumn'])

        # REGRESSION: added for R3a — skip to_categorical for regression task
        if task_type == 'classification':
            self.labels_train = to_categorical(self.labels_train)
            self.labels_test = to_categorical(self.labels_test)
        else:
            self.labels_train = self.labels_train.astype(np.float32)
            self.labels_test = self.labels_test.astype(np.float32)

        sequence_train = self.rectangularSequenceData(sequence_train_raw, signal_length=self.config['signal_length'], pooling_type=self.config['pooling_type'])
        sequence_test = self.rectangularSequenceData(sequence_test_raw, signal_length=self.config['signal_length'], pooling_type=self.config['pooling_type'])

        self.normalize(sequence_train, sequence_test)

    def splitDataByColumn(self, data_DF, column, train_values, test_values):
        train_DF = data_DF[data_DF[column].isin(train_values)].copy()
        test_DF = data_DF[data_DF[column].isin(test_values)].copy()
        logger.info(f"Train set: {len(train_DF)} samples, Test set: {len(test_DF)} samples")
        return train_DF, test_DF

    @abstractmethod
    def modelSetup(self):
        pass

    def trainModel(self):
        log_dir = os.path.join(self.output_parent_dir, 'logs', 'tensorboard', self.config['model_type'])
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        self.history = self.model.fit(
            self.X_train,
            self.labels_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=self.config['validation_split'],
            verbose=self.config['verbose'],
            callbacks=[tensorboard_callback]
        )
        
        output_dir = os.path.join(self.output_parent_dir, 'models')
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{self.config['model_type']}_model_{self.timestamp}")
        self.model.save(model_path+'.keras')
        self.model.export(model_path)
        logger.info(f"Model saved to {model_path}")

    def evalModel(self):
        loss, accuracy = self.model.evaluate(self.X_test, self.labels_test)

        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.labels_test, axis=1)

        # Calculate metrics
        precision = precision_score(y_true_classes, y_pred_classes, zero_division=0)
        recall = recall_score(y_true_classes, y_pred_classes, zero_division=0)
        f1 = f1_score(y_true_classes, y_pred_classes, zero_division=0)
        cm = confusion_matrix(y_true_classes, y_pred_classes)

        # Create results DataFrame
        results_data = {
            'Train Cases': [self.config.get('train_caseIDs', [])],
            'Test Cases': [self.config.get('test_caseIDs', [])],
            'Accuracy': [accuracy],
            'Loss': [loss],
            'Precision': [precision],
            'Recall': [recall],
            'F1-Score': [f1],
            'confusion_matrix': [cm.tolist()],  # Convert numpy array to list for serialization
            'y_true_classes': [y_true_classes.tolist()],
            'y_pred_classes': [y_pred_classes.tolist()]
        }
        results_df = pd.DataFrame(results_data)

        # Save results to CSV
        output_dir = os.path.join(self.output_parent_dir, 'logs')
        os.makedirs(output_dir, exist_ok=True)
        csv_filename = f"evaluation_results_{self.config['model_type']}_{self.timestamp}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)
        results_df.to_csv(csv_filepath, index=False)
        logger.info(f"Evaluation results saved to {csv_filepath}")

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues, ax=ax1)
        ax1.set_title('Confusion Matrix')

        ax2.plot(self.history.history['loss'], label='train loss')
        ax2.plot(self.history.history['val_loss'], label='validation loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss')
        ax2.legend()

        plt.tight_layout()
        
        plot_filename = f"training_loss_{self.config['model_type']}_{self.timestamp}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath)
        plt.close()
        logger.info(f"Training loss plot saved to {plot_filepath}")
        logger.info(f"Test Accuracy: {accuracy:.2f}, Test Loss: {loss:.2f}")

    def save_config(self):
        config_dir = os.path.join(self.output_parent_dir, 'configs')
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, f"{self.config['model_type']}_config_{self.timestamp}.yaml")
        
        # Create a serializable copy of the config
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
    def evalModelRegression(self):
        y_pred = self.model.predict(self.X_test).flatten()
        y_true = self.labels_test.flatten()

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        results_data = {
            'Train Cases': [self.config.get('train_caseIDs', [])],
            'Test Cases': [self.config.get('test_caseIDs', [])],
            'MAE': [mae],
            'RMSE': [rmse],
            'R2': [r2],
        }
        results_df = pd.DataFrame(results_data)

        output_dir = os.path.join(self.output_parent_dir, 'logs')
        os.makedirs(output_dir, exist_ok=True)
        csv_filename = f"evaluation_regression_{self.config['model_type']}_{self.timestamp}.csv"
        csv_filepath = os.path.join(output_dir, csv_filename)
        results_df.to_csv(csv_filepath, index=False)
        logger.info(f"Regression evaluation results saved to {csv_filepath}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        ax1.set_xlabel('True wear_norm')
        ax1.set_ylabel('Predicted wear_norm')
        ax1.set_title(f'Regression: R²={r2:.3f}, MAE={mae:.3f}')

        ax2.plot(self.history.history['loss'], label='train loss')
        ax2.plot(self.history.history['val_loss'], label='val loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('MSE Loss')
        ax2.set_title('Training Loss')
        ax2.legend()

        plt.tight_layout()
        plot_filepath = os.path.join(output_dir, f"regression_eval_{self.config['model_type']}_{self.timestamp}.png")
        plt.savefig(plot_filepath)
        plt.close()
        logger.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    def run(self):
        self.prepare_data()
        self.modelSetup()
        self.save_config()
        self.trainModel()
        # REGRESSION: added for R3a
        if self.config.get('task_type', 'classification') == 'regression':
            self.evalModelRegression()
        else:
            self.evalModel()

class LSTMPipeline(ModelPipeline):
    def modelSetup(self):
        # REGRESSION: added for R3a — branch on task_type
        task_type = self.config.get('task_type', 'classification')
        inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
        base_layers = LSTM(64, return_sequences=True)(inputs)
        base_output = LSTM(32, return_sequences=False)(base_layers)
        base_model = Model(inputs=inputs, outputs=base_output, name="feature_extractor_base")
        base_model.trainable = True
        head_layers = Dense(16, activation='tanh')(base_model.output)
        if task_type == 'regression':
            outputs = Dense(1, activation='linear')(head_layers)
            self.model = Model(inputs=base_model.input, outputs=outputs)
            self.model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
        else:
            outputs = Dense(self.labels_train.shape[1], activation='softmax')(head_layers)
            self.model = Model(inputs=base_model.input, outputs=outputs)
            self.model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

class Conv1DPipeline(ModelPipeline):
    def modelSetup(self):
        # REGRESSION: added for R3a — branch on task_type
        task_type = self.config.get('task_type', 'classification')
        inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
        base_layers = Conv1D(filters=32, kernel_size=5, activation='relu')(inputs)
        base_layers = MaxPooling1D(pool_size=2)(base_layers)
        base_layers = Conv1D(filters=64, kernel_size=5, activation='relu')(base_layers)
        base_output = GlobalAveragePooling1D()(base_layers)
        base_model = Model(inputs=inputs, outputs=base_output, name="feature_extractor_base")
        base_model.trainable = True
        head_layers = Dropout(0.5)(base_model.output)
        head_layers = Dense(32, activation='relu')(head_layers)
        if task_type == 'regression':
            outputs = Dense(1, activation='linear')(head_layers)
            self.model = Model(inputs=base_model.input, outputs=outputs)
            self.model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
        else:
            outputs = Dense(self.labels_train.shape[1], activation='softmax')(head_layers)
            self.model = Model(inputs=base_model.input, outputs=outputs)
            self.model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

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

        self.normalize(self.X_train, self.X_test)

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

    def evalModel(self):
        plt.figure(figsize=(15, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
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

        reconstruction = self.model.predict(self.X_test)
        reconstruction_error = np.mean(np.square(self.X_test - reconstruction), axis=(1, 2))

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
        filename_roc = f"autoencoder_eval_{self.timestamp}.png"
        filepath_roc = os.path.join(output_dir, filename_roc)
        plt.savefig(filepath_roc)
        plt.close()
        logger.info(f"Evaluation plot saved to {filepath_roc}")
        logger.info(f"AUC: {roc_auc:.4f}")

