import argparse 
import importlib

DATA_CLASS_MODULE = "dataset"
MODEL_CLASS_MODULE = "models"
TRAIN_DATA_DIR = "../input/data/train/images"

def import_class(module_and_class_name):
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def setup_data_and_model_from_args(args: argparse.Namespace, data_class_module = DATA_CLASS_MODULE, model_class_module = MODEL_CLASS_MODULE):
    data_class = import_class(f"{data_class_module}.{args.data_class}")
    model_class = import_class(f"{model_class_module}.{args.model_class}")

    dataset = data_class(TRAIN_DATA_DIR)

    #Fine-tuning 결과에 따라 달라질 것임
    model = model_class(dataset.num_classes)
    return dataset, model

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')