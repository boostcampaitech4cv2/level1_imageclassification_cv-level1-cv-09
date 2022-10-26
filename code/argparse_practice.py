from dataset import MaskBaseDataset, BaseAugmentation
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

def setup_data_and_model_from_args(args: argparse.Namespace):
    data_class = import_class(f"{DATA_CLASS_MODULE}.{args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{args.model_class}")

    dataset = data_class(TRAIN_DATA_DIR)
    model = model_class(dataset.num_classes)
    return dataset, model

parser = argparse.ArgumentParser()

# Data and model checkpoints directories
parser.add_argument('--data_class', type=str, default="MaskBaseDataset", help='Which dataset will you use?')
parser.add_argument("--model_class", type = str, default = "ResNet34", help = "Which model will you use?")
args = parser.parse_args()

dataset, model = setup_data_and_model_from_args(args)

print("dataset type:", type(dataset))
print("model type:", type(model))
print(model)