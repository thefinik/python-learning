# Image Classification with Transfer Learning (MobileNetV2)

Tiny demo project: 3-class image classifier using Transfer Learning.

## Model

- Base model: `MobileNetV2` pre-trained on ImageNet
- Base model is frozen (`base_model.trainable = False`)
- Custom classification head:
  - `GlobalAveragePooling2D`
  - `Dense(128, relu)`
  - `Dense(3, softmax)`

## Dataset structure

```text
dataset_3class/
    train/
        class1/
        class2/
        class3/
    test/
        class1/
        class2/
        class3/

### Model file

The trained model (`model_3class.h5`) is not included in this repo because of size and local TensorFlow issues on macOS.  
To train it yourself, run:

```bash
python3 train.py