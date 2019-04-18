This is a demo for purning resnet18 on mini-imagenet.
- This folder contains dataset and codes, you can just run `get_small_model.py` and it would show `top1 acc` and `top5 acc`. Then it would save the model after pruning.
- `pytorch==0.4.0` and `torchvision==0.2.1` are tested.
- Also, python3 is needed because of I use pickle to save the dataset, python2 may face problems when loading dataset.

- Dataset: **miniImagenet**

  The miniImageNet dataset is a subset of ImageNet and is composed of 60,000 images in 100 cate-
  gories. In each category, we take 500 images as training set and other 100 as testing set.

  You can download the dataset `test_set.pkl` from [here](https://drive.google.com/open?id=16Zi-LtW91Fd2S7KCWhOAPMsinbtEbzYI).