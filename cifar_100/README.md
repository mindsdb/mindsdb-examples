## Summary

I've chose CIFAR 100 over imagenet because it's quciker to run, and also the dual nature of the labels helps study the mutliple target variable prediction capabilities of mindsdb.

Mindsdb gets accuracy similar to a state of the art model in late 2015 to early 2016. Which is not bad, consider we don't optimize mindsdb for image labling and the training times and resulting model size are rather small (more on that bellow).


## Accuracy of other models

It should be noted that computer visions is a fast advancing field, thus, there will probably be better models than the one found that simply weren't applied to CIFRAR-100.

It should also be noted that the best computer vision models, when trained on AWS or GCP, can cost over 100k $ to train (once). Mindsdb, in it's current version, aims for performance over perfect accuracy, especially on image & text data. The cost of training the model used here would be somewhere under 50$. (This is in large part because we used pre-trained weights for the encoders).

### Best model known thus far

Paper: https://arxiv.org/pdf/1905.11946v2.pdf
Accuracy of: 91.7% On *I belive* classes (the paper isn't explicit about this)

You can see a ranking here: https://paperswithcode.com/sota/image-classification-on-cifar-100

### Other interesting models

Resnet56 with batchnorm, accuracy: ~62%.

A model which is closer in trianing time to mindsdb would be a resnet implemented in pytorch: https://zhenye-na.github.io/2018/10/07/pytorch-resnet-cifar100.html
This one has 56 layers and yields an accuracy of ~62% on the classes, which is bellow what mindsdb yield. The original Resnet paper doesn't have benchmarks on CIFAR, but I assume the accuracy it would yield would be similar or lower, since this model uses Batch normalization, whilst the original residual network paper was written before batchnorm was introduced.


## Mindsdb accuracy
Running on NVIDIA GeForce RTX 2070 (Laptop)
Training: ~ 2 hours
Testing: ~ 20 minutes

With resnext50-small encoder (balanced):

Accuracy for image classess: 70.09%
Accuracy for subperclasses: 82.00%


With resnet18 encoder (fast):

Accuracy for image classes: 67.06%
Accuracy for image superclasses: 80.02%

### Lightwood
Lightwood was able to run the dataset and yielded the above accuracies.

### Ludwig
Ludwig was unable to run this dataset due to size issues.

## References
* https://paperswithcode.com/sota/image-classification-on-cifar-100
* http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d313030
* https://benchmarks.ai/cifar-100
