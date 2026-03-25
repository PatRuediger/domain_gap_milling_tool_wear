import yaml
import sys
import subprocess
import os
import numpy as np
import tensorflow as tf

def main(config_path):
    np.random.seed(42)
    tf.random.set_seed(42)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    process_id = os.getpid()
    
    mode = config.get('mode', 'train')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_new_path = os.path.join(script_dir, "run_new.py")

    if mode == 'train':
        subprocess.run([sys.executable, run_new_path, config_path])

    elif mode == 'inference':
        tasks = config.get('inference_tasks', [])
        if not tasks:
            raise ValueError("'inference_tasks' not found in config for inference mode.")

        for i, task_config in enumerate(tasks):
            single_task_config = config.copy()
            single_task_config['inference_tasks'] = [task_config]

            temp_config_path = f"temp_config_pid{process_id}_task_{i}.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(single_task_config, f)

            subprocess.run([sys.executable, run_new_path, temp_config_path])
            os.remove(temp_config_path)

    elif mode == 'transfer_learn':
        tasks = config.get('transfer_learning_tasks', [])
        if not tasks:
            raise ValueError("'transfer_learning_tasks' not found.")

        for i, task_config in enumerate(tasks):
            single_task_config = config.copy()
            single_task_config['transfer_learning_tasks'] = [task_config]

            temp_config_path = f"temp_config_pid{process_id}_task_{i}.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(single_task_config, f)

            subprocess.run([sys.executable, run_new_path, temp_config_path])
            os.remove(temp_config_path)

    else:
        raise ValueError(f"Unknown mode: {mode}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_all_tasks.py <config.yaml>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)