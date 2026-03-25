
import yaml
import sys
import numpy as np
import tensorflow as tf
from modelPipelines import LSTMPipeline, Conv1DPipeline, AutoencoderPipeline

def main(config):
    np.random.seed(42)
    tf.random.set_seed(42)
    model_type = config.get('model_type', 'lstm').lower()
    if model_type == 'lstm':
        pipeline = LSTMPipeline(config)
    elif model_type == '1d_conv':
        pipeline = Conv1DPipeline(config)
    elif model_type == 'autoencoder':
        pipeline = AutoencoderPipeline(config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    pipeline.run()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <config.yaml>")
        sys.exit(1)
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    main(config)
