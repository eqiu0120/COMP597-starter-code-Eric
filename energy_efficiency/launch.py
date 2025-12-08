from typing import Any, Dict, Optional, Tuple
import argparse
import gc
import src.data as data
import src.models as models
import src.trainer as trainer
import src.config as config

def process_conf(conf : config.Config) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    dataset = data.load_data(conf)
    print(f"Dataset loaded with {len(dataset)} samples.")

    return models.model_factory(conf, dataset)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(config.ConfigArgs.MODEL.to_arg(), type=str, help="Which model to train.", choices=[models.get_available_models()])
    parser.add_argument(config.ConfigArgs.TRAINER.to_arg(), type=str, help="How to train the model", default="simple", choices=["simple"])
    parser.add_argument(config.ConfigArgs.DATASET.to_arg(), type=str, help="Which dataset to use.", default="allenai/c4",  choices=["allenai/c4"])
    # TODO Remove default, it makes no sense.
    parser.add_argument(config.ConfigArgs.DATASET_TRAIN_FILES.to_arg(), type=str, help="Which files to use for the dataset.", default="/mnt/nobackup/omicha1/c4/downloaded/multilingual/c4-en.tfrecord-00000-of-*.json.gz")
    # TODO Again, default makes no sense. Should be removed.
    parser.add_argument(config.ConfigArgs.DATASET_SPLIT.to_arg(), type=str, help="How to split the dataset (ex: train[:100])", default="train[:100]")
    parser.add_argument(config.ConfigArgs.DATASET_LOAD_NUM_PROC.to_arg(), type=int, help="Number of threads used to load the dataset.", default=20)
    parser.add_argument(config.ConfigArgs.TOKENIZE_NUM_PROCESS.to_arg(), type=int, help="Number of threads used to tokenize the dataset.", default=20)
    parser.add_argument(config.ConfigArgs.BATCH_SIZE.to_arg(), type=int, help="Size of batches", default=4)
    parser.add_argument(config.ConfigArgs.TRAIN_STATS.to_arg(), type=str, help="Type of statistics to gather. By default it is set to no-op, which ignores everything.", default="no-op", choices=["no-op", "no-op-sync", "simple", "torch-profiler", "codecarbon", "averaged-energy"])
    # the following arguments are used for codecarbon tracking
    parser.add_argument(config.ConfigArgs.RUN_NUM.to_arg(), type=int, help="The run number used for codecarbon file tracking.", default=0)
    parser.add_argument(config.ConfigArgs.PROJECT_NAME.to_arg(), type=str, help="The name of the project used for codecarbon file tracking.", default="energy-efficiency")
    
    # TODO: The default is set to 1e-6 which is a good default rate for training Qwen models. 1e-7 is a good default rate for Switch-Transformers. Make sure to adjust it for different models
    parser.add_argument(config.ConfigArgs.LEARNING_RATE.to_arg(), type=float, help="The learning rate for training. It is used by the optimizer for all models.", default=1e-6)
    
    args, _ = parser.parse_known_args()
    return args

def main():
    args = get_args()
    conf = config.Config(args)
    print(f"available models: {models.get_available_models()}")
    model_trainer, model_kwargs = process_conf(conf)

    model_trainer.train(model_kwargs)

    # This forces garbage collection at process exit. It ensure proper closing of resources.
    del conf
    del model_kwargs
    del model_trainer

if __name__ == "__main__":
    main()
    gc.collect()

