## Best compressed models using (GL-P)L-P & (GS-P)S-P

### 1. LeNet-5 on MNIST

Here we provide a well-trained model of LeNet-5 on MNIST with LASSO-penalty & Split LBI penalty, respectively. The related files are included in folder `./plain-best/`,[`./lasso-best/`](https://github.com/zhangshun97/SLBI_research_best_models/tree/master/LeNet-5%20on%20MNIST/lasso-best) and [`./slbi-best/`](https://github.com/zhangshun97/SLBI_research_best_models/tree/master/LeNet-5%20on%20MNIST/slbi-best). In each folder, you will find the following elements:

- folder `checkpoint` contains the well-trained model (without pruning)
- python file `model.py` contains the needed LeNet-5 structure
- Image `Train_hist_iter.png` shows the whole training curve, from which you can obtain the convergence speed of our algorithm
- python file `get_small_model.py` enables you to test the full model and the pruned model, which also saves the pruned model for you

Just run `get_small_model.py` and it will automatically download the required dataset in folder `./data/` and prints out essential arguments as follow example and, finally, you can find a pruned model in folder `./small_model_[lasso/slbi]/`

- example output:

  ```
  ==> Preparing data..
  Files already downloaded and verified
  ==> Building model..
  ==> Resuming from checkpoint..
  0 10 Loss: 0.336 | Acc: 90.400% (904/1000)
  1 10 Loss: 0.373 | Acc: 89.800% (1796/2000)
  2 10 Loss: 0.374 | Acc: 90.067% (2702/3000)
  3 10 Loss: 0.368 | Acc: 90.150% (3606/4000)
  4 10 Loss: 0.366 | Acc: 90.120% (4506/5000)
  5 10 Loss: 0.364 | Acc: 90.083% (5405/6000)
  6 10 Loss: 0.357 | Acc: 90.200% (6314/7000)
  7 10 Loss: 0.351 | Acc: 90.338% (7227/8000)
  8 10 Loss: 0.348 | Acc: 90.333% (8130/9000)
  9 10 Loss: 0.342 | Acc: 90.540% (9054/10000)
  Accuracy before pruning: 90.54
  Pruning rate: 98.4375%
  Start pruning layer: layer3.0.conv1.weight
  Start pruning layer: layer3.0.conv2.weight
  Start pruning layer: layer3.1.conv1.weight
  Start pruning layer: layer3.1.conv2.weight
  Start pruning layer: layer4.0.conv1.weight
  Start pruning layer: layer4.0.conv2.weight
  Start pruning layer: layer4.1.conv1.weight
  Start pruning layer: layer4.1.conv2.weight
  Pruning ended.
  Total compression rate (#remain / #total): 9.07 %
  0 10 Loss: 0.342 | Acc: 88.700% (887/1000)
  1 10 Loss: 0.357 | Acc: 87.850% (1757/2000)
  2 10 Loss: 0.364 | Acc: 87.800% (2634/3000)
  3 10 Loss: 0.359 | Acc: 87.750% (3510/4000)
  4 10 Loss: 0.357 | Acc: 88.000% (4400/5000)
  5 10 Loss: 0.358 | Acc: 87.717% (5263/6000)
  6 10 Loss: 0.356 | Acc: 87.786% (6145/7000)
  7 10 Loss: 0.352 | Acc: 88.013% (7041/8000)
  8 10 Loss: 0.354 | Acc: 87.911% (7912/9000)
  9 10 Loss: 0.350 | Acc: 88.070% (8807/10000)
  Accuracy after pruning: 88.07
  Saving..
  ```


### 2. ResNet-18 on CIFAR-10

Here we also provide a well-trained model of ResNet-18 on CIFAR-10 with LASSO-penalty & Split LBI, respectively. The related files are included in folder [`./lasso-best/`](https://github.com/zhangshun97/SLBI_research_best_models/tree/master/ResNet-18%20on%20CIFAR-10/lasso-best) and [`./slbi-best/`](https://github.com/zhangshun97/SLBI_research_best_models/tree/master/ResNet-18%20on%20CIFAR-10/slbi-best). Similarly, you can get the pruning information and the pruned model by running `get_small_model.py`.

- The best LASSO model can be downloaded [here](https://drive.google.com/open?id=1JlBseAqJBNdmVJz_L8vTGO75OW5dnab_) and should be moved to `./lasso-best/checkpoint/`
- The best SLBI model can be downloaded [here](https://drive.google.com/open?id=1XASwuBaKaK_ftkRL6ZcCzm8mJclwYIPh) and should be moved to `./slbi-best/checkpoint/`
- The best plain model can be downloaded [here](https://drive.google.com/open?id=1S3M4ompxV6yzgEc_OyHz5PO-HBvdN0_u) and should be moved to `./plain-best/checkpoint/`

### 3. ResNet-18 on miniImagenet

Here we also provide a well-trained model of ResNet-18 on CIFAR-10 with Split LBI. The related files are included in folder [`./slbi-best/`](https://github.com/zhangshun97/SLBI_research_best_models/tree/master/ResNet-18%20on%20miniImagenet/lbi-best). Similarly, you can get the pruning information and the pruned model by running `get_small_model.py`.

- Dataset: **miniImagenet**

  The miniImageNet dataset is a subset of ImageNet and is composed of 60,000 images in 100 cate-
  gories. In each category, we take 500 images as training set and other 100 as testing set.

  You can download the dataset `test_set.pkl` from [here](https://drive.google.com/open?id=16Zi-LtW91Fd2S7KCWhOAPMsinbtEbzYI).

- The best SLBI model can be downloaded [here](https://drive.google.com/open?id=1C9RqLjfsQf6rouBAD2z7ZsmwOY6YDFAY) and should be moved to `./slbi-best/checkpoint/`

- The plain training model can be download [here](https://drive.google.com/open?id=1CUl8gx-tjHRNlcCqBWmWqh4OY1Kxzi3c) and should be moved to `./plain-best/checkpoint/.`

---

**Note that**:

- These codes should work well with `PyTorch 0.4.0` & `torchvision 0.2.1`.
- The statistics of the pruned model, such as accuracy and compression rate, may be slightly different to those in our paper. This may caused by our 'saving procedure', which is that in order to be more efficient, we only save the model when its test accuracy (without pruning) is higher during the training. That is to say, our final results of accuracy compression rate may **not** refer to our finally saved model. However, they are very close!

