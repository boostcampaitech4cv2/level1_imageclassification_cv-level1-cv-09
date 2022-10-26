# 간단한 사용법

## 모델을 저장할 directory를 만들기

-  제 경우엔 saved_dir을 팠고
=> 단일 실험은 saved_dir 아래에서 exp를 계속 늘려갔고
=> 다중 실험은 saved_dir 아래에 joint_exp를 판 후, age & mask & gender를 각각 늘려갔습니다.

## batch script에 대한 설명

- 쉘 스크립트의 가동은 sh (파일명.sh)로 가능합니다.

### three_way_training.sh

```
SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --data_class AgeOnlyDataset  --name "age"
SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --data_class GenderOnlyDataset  --name "gender"
SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --data_class MaskOnlyDataset  --name "mask"
```

아, 단일 학습은 고려하지 않았습니다. 그건 그냥 README에 있는거 그대로 학습시켜도 충분합니다.

3중 학습의 경우...
이거 각 줄이, agedataset에 대한 Model을 학습 & genderdataset에 대한 학습 & maskdataset에 대한 학습하는 코드를 의미합니다.
새롭게 학습하고 싶지 않다면, 주석처리하면 됩니다.

### three_way_inference.sh
```
#!/bin/sh
SM_CHANNEL_EVAL="/opt/ml/input/data/eval" SM_CHANNEL_MODEL="saved_models/exp6" SM_OUTPUT_DATA_DIR="inference_output"  \
python inference.py --model ResNet34 \
--single False \
--age_dir "saved_models/joint_exp/age3" \
--gender_dir "saved_models/joint_exp/gender3" \
--mask_dir "saved_models/joint_exp/mask3"
```
단일 학습을 시키고 싶다면, --single 옵션을 True로 바꾼 후 CHANNEL_MODEL에 최신 단일 모델을 입력하세요.
다중 학습을 시킬 것이라면,
각 age_dir, gender_dir, mask_dir에 각각의 최신 모델디렉토리를 변경시키면 충분합니다.


# What-to-do??

할 게 제법 남았습니다.

- Data Augmentation : CutMix
- Use Pseudo-label : Mislabel fix
- Find best single model : 분명 더 좋은 모델이 있겠죠?
- Custom Loss : Loss를 바꾸거나, Scheduler, OPtimizer를 바꾸거나.

# Pseudo_label.csv??
- Inference 시에, age, mask,gender에 대해 prediction을 내뱉도록 했습니다.
- 이를 train_label과 함께 이용하는 방법도 있겠죠?(실제 Test시에.) => 치팅에 가깝겠지만, 극후반부에 고려해볼법한 것 같아요.
- Single model을 train dataset에 대해 적용시킨다면, 그걸 통해 data correction을 할 수 있을 것 같습니다.


