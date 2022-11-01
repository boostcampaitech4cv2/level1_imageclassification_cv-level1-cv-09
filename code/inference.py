import argparse
import multiprocessing
import os
from importlib import import_module

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils import str_to_bool
from dataset import TestDataset, MaskBaseDataset

import warnings
warnings.filterwarnings('ignore')

FOLD_NUM = 5

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("models"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    SINGLE_TRAIN = str_to_bool(args.single)
    print("SINGLE TRAIN???:", SINGLE_TRAIN)
    
    print("APPLY TTA??????:", args.tta)
    if SINGLE_TRAIN:
        num_classes = MaskBaseDataset.num_classes  # 18
        model = load_model(model_dir, num_classes, device).to(device)
        model.eval()

        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
        dataset = TestDataset(img_paths, args.resize, tta= args.tta)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        print("Calculating inference results..")
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred = model(images)
                pred = pred.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())

        info['ans'] = preds
        save_path = os.path.join(output_dir, args.output_name)
        info.to_csv(save_path, index=False)
        print(f"Inference Done! Inference result saved at {save_path}")
    
    else:
        #You choose to do three-way branch training
        mask_dir = args.mask_dir
        gender_dir = args.gender_dir
        age_dir = args.age_dir

        model_dirs = [mask_dir, gender_dir, age_dir]
        number_classes = [3, 2, 3]

        total_predictions = []

        print("Calculating inference results..")
        for order, (each_model_dir, num_classes) in enumerate(list(zip(model_dirs, number_classes))):
            model = load_model(each_model_dir, num_classes, device).to(device)
            model.eval()

            img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
            dataset = TestDataset(img_paths, args.resize, tta= args.tta)
            
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=multiprocessing.cpu_count() // 2,
                shuffle=False,
                pin_memory=use_cuda,
                drop_last=False,
            )

            preds = []
            with torch.no_grad():
                for idx, images in enumerate(loader):
                    images = images.to(device)
                    pred = model(images)
                    pred = pred.argmax(dim=-1)
                    preds.extend(pred.cpu().numpy())

            if order == 0:
                print("Mask model done. Multiply it by 6")
                total_predictions.append(preds)
            elif order ==1:
                print("gender model done. Multiply it by 3")
                total_predictions.append(preds)
            else:
                print("Age model done. multiply it by 1")
                total_predictions.append(preds)
        final_predictions = []

        #Unpacking
        mask_preds, gender_preds, age_preds = total_predictions

        for mask, gender, age in zip(mask_preds, gender_preds, age_preds):
            final_label = 6 * mask + 3 * gender + age
            final_predictions.append(final_label)
            
        info['ans'] = final_predictions
        save_path = os.path.join(output_dir, args.output_name)
        info.to_csv(save_path, index=False)

        tempfile = pd.DataFrame()
        tempfile["id"] = info["ImageID"]
        tempfile["age"] = age_preds
        tempfile["mask"] =  mask_preds
        tempfile["gender"] = gender_preds

        save_other_path = os.path.join(output_dir, "pseudo_label.csv")
        tempfile.to_csv(save_other_path,index= False)
        print(f"Inference Done! Inference result saved at {save_path}")
        print(f"Other file is also done. Saved at {save_other_path}")


"""
SM : 마찬가지로, kfold inference도 구분하지 마세요!
"""

@torch.no_grad()
def kfold_inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    SINGLE_TRAIN = str_to_bool(args.single)
    print("SINGLE TRAIN???:", SINGLE_TRAIN)

    if SINGLE_TRAIN:
        num_classes = MaskBaseDataset.num_classes  # 18
        model = load_model(model_dir, num_classes, device).to(device)
        model.eval()

        img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
        dataset = TestDataset(img_paths, args.resize)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        print("Calculating inference results..")
        preds = []
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred = model(images)
                pred = pred.argmax(dim=-1)
                preds.extend(pred.cpu().numpy())

        info['ans'] = preds
        save_path = os.path.join(output_dir, args.output_name)
        info.to_csv(save_path, index=False)
        print(f"Inference Done! Inference result saved at {save_path}")
    
    else:
        #You choose to do three-way branch training
        mask_dir = args.mask_dir
        gender_dir = args.gender_dir
        age_dir = args.age_dir

        model_dirs = [mask_dir, gender_dir, age_dir]
        number_classes = [3, 2, 3]

        total_predictions = []

        print("Calculating inference results..")
        for order, (each_model_dir, num_classes) in enumerate(list(zip(model_dirs, number_classes))):
            total_fold_preds = []
            for fold in range(5):
                print("Fold:", fold)
                model_dir = os.path.join(each_model_dir, f"fold{fold}")
                model = load_model(model_dir, num_classes, device).to(device)
                model.eval()

                img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
                dataset = TestDataset(img_paths, args.resize)
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    num_workers=multiprocessing.cpu_count() // 2,
                    shuffle=False,
                    pin_memory=use_cuda,
                    drop_last=False,
                )

                #여기에 2D array들이 담길 것
                each_fold_probs = []

                with torch.no_grad():
                    for idx, images in enumerate(loader):
                        images = images.to(device)
                        each_fold_pred = model(images)
                        each_fold_probs.extend(each_fold_pred.cpu().numpy())

                fold_prediction = np.array(each_fold_probs)
                print("shape of fold_prediction:", fold_prediction.shape)  #( Batch x prediction 확률이어야함)
                total_fold_preds.append(each_fold_probs)
                        
            sum_pred_arr = np.zeros_like(total_fold_preds[0])
            for each_pred in total_fold_preds:
                sum_pred_arr += each_pred
            
            sum_pred_arr /= FOLD_NUM

            preds = sum_pred_arr.argmax(axis=-1)

            if order == 0:
                print("Mask model done. Multiply it by 6")
                total_predictions.append(preds)
            elif order ==1:
                print("gender model done. Multiply it by 3")
                total_predictions.append(preds)
            else:
                print("Age model done. multiply it by 1")
                total_predictions.append(preds)
        final_predictions = []

        #Unpacking
        mask_preds, gender_preds, age_preds = total_predictions

        for mask, gender, age in zip(mask_preds, gender_preds, age_preds):
            final_label = 6 * mask + 3 * gender + age
            final_predictions.append(final_label)
            
        info['ans'] = final_predictions
        save_path = os.path.join(output_dir, args.output_name)
        info.to_csv(save_path, index=False)

        tempfile = pd.DataFrame()
        tempfile["id"] = info["ImageID"]
        tempfile["age"] = age_preds
        tempfile["mask"] =  mask_preds
        tempfile["gender"] = gender_preds

        save_other_path = os.path.join(output_dir, "pseudo_label.csv")
        tempfile.to_csv(save_other_path,index= False)
        print(f"Inference Done! Inference result saved at {save_path}")
        print(f"Other file is also done. Saved at {save_other_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #Joint training, or single training?
    parser.add_argument('--single', type=str_to_bool, nargs='?', const=True, default=False)

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    
    # parser.add_argument('--resize', type=tuple, help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--resize', type=tuple, default=(380, 380), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--output_name', type=str, default='output.csv')

    #For Three-way branch training
    #To activate three-way training, set --single = False in CLI environment
    parser.add_argument("--age_dir", type=str, default = "/saved_models/joint_exp/age")
    parser.add_argument("--gender_dir", type=str, default = "/saved_models/joint_exp/gender")
    parser.add_argument("--mask_dir", type=str, default = "/saved_models/joint_exp/mask")

    #Will you do tta???
    parser.add_argument('--tta', type=str_to_bool, nargs='?', const=True, default=False)

    args = parser.parse_args()


    assert args.resize is not None, "Resize를 꼭 지정해주시고 반드시 반드시반드시 반드시 Train과 동일하게 지정해주세요!!!!!!!!!!!!!!!!!!!!!!!!!!"

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("잠깐! 혹시 inference의 resize가 train과 맞는지 확인하셨나요?")
    #Change if you want to do kfold, or just use inference
    inference(data_dir, model_dir, output_dir, args)
    #kfold_inference(data_dir, model_dir, output_dir, args)
    print("됐습니다! 혹시 inference의 resize가 train과 맞는지 확인하셨나요?")