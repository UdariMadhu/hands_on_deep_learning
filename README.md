# Hands-On Deep Learning with PyTorch: A short mini-course

This deep learning mini-course comprises a sequence of 10 assignments, aiming to provide first-hand experience in training deep neural networks. It assumes a basic understanding of both PyTorch and deep neural networks, and furnishes the boilerplate code required for neural network training. Each assignment is devoted to a single concept, including fine-tuning, architectural design, and distributed training. The completion of this course is expected to strongly boost your confidence in working with deep neural networks.

To get started, we have a basic neural network training script (`train.py`) that can train the network on common datasets (like mnist, cifar10, imagenet). By default we assume access to a small gpu-cluster, though it should be straightforward to also try it on [colab](https://colab.research.google.com).
```
pip install -r requirements.txt

CUDA_VISIBLE_DEVICES=0 python train.py --configs ./assets/configs.yml --override_args dataset.name=cifar10
```

## Assignment-1
We currently use the `CNNSimple` network, which does a good job but far from perfect. Can you write a better network architecture. It only achieves 98.8 test accuracy on mnist after 25 epochs. On mnist its very easy to achieves >99.2% accuracy. You can read about basic neural network architectures here:
- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 
- Deisgn of CNN network: https://cs231n.github.io/
- VGG network, a very simple CNN (not much relevant today): https://blog.paperspace.com/vgg-from-scratch-pytorch/
- Make a note of accuracy achieved post 25 epochs on mnist and cifar10.
- You can learn basic of neural network optimization principles in early slides of [this lecture](https://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture05-rnnlm.pdf).

## Assignment-2
Current we've deliberately skipped saving the trained network. 
- Can you save the network after training?
- At some point, we want to add the ability of resuming from a previous checkpoint (very crucial in large experiments as training runs often fails). But lets skip it for now as we won't be doing much large scale training and it would make the code cumbersome.

## Assignment-3
In assigment-1 you may have tried using some of the advanced architectures (though they weren't required) such as resnets and it's variants. From now onwards, we will completely switch to using some of these standard architectures
- Modern neural network architectures (such as ConvNexts) can works with varying imags sizes (e.g., 224x224, 300x300). But the architecture of neural network has to be changed slightly when working with super small resultions (e.g, 32x32). Let's say that a architecture has size pooling layers. 32x32 is so small that after 5 pooling you will have a 1x1 image. So generally we slightly modify standard neural networks to make them work with smaller resolution. 
- Setup two common architecutres for cifar10. You can find some smaller architectures for cifar10 resolution here: https://github.com/kuangliu/pytorch-cifar
- [`Timm`](https://github.com/huggingface/pytorch-image-models) is an excellent package to obtain neural networks architectures (it also has weight trained on ImageNet). Setup models initialization such that we can use any network from timm. Use assertions to ensure that timm models are only used for higher resolution datasets.
- Now try training one of the modern models on cifar10. See how better it does compared to the best model you designed in assignment-1. Designing a powerful network is one of the crucial task in deep learning.
- Let's try training a training a model on the ImageNet-100 dataset. This dataset is a 1/10 of ImageNet-1K dataset (which have 1.2M images), but training on it is still going to be dreadfully slow. For the rest of the assignments, our objective would be speed up training on this dataset both with engineering and deep learning tricks.

## Assignment-4
Let's learn the value of finetuning. Finetuning refers to training a network pretrained on generic large datasets (e.g., ImageNet or LAION-2B models for vision, gpt for language, etc.) and training them a little bit on your datasets. This paradigm is really powerful and is the cornerstone of modern deep leanring. In the last step of prev assignment, you have seen how slow training from scratch is.
- Lets use some of pretrained models from timm and finetuned them (you can do so by setting `pretrained=True`). Since ImageNet-100 is part of ImageNet, the pretrained network is already seen this dataset. So this will get finetuned crazy fast. Try to use network that are trained on some other datasets (e.g., laion-400m or laion-2B).
- Would you need to change some optimization parameters (e.g., batch-size, lr, etc) when finetuning compared to training from scratch.
- Finetuning is great but let's avoid using it in these assignments as the objective to train powerful neural network in absence of any pretrained models. Aferall someone has to train the pretrained model on big datasets.

## Assignment-5
Currntly we only use one gpu. You may have noticed that when running the code we set one gpu with `CUDA_VISIBLE_DEVICES=gpu_id`. Now configure your code to take advantage of multiple-gpus (`CUDA_VISIBLE_DEVICES=0,1,2,3`).
- You can read about it here: [Tutorial-1](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html) or [Tutorial-2](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html?highlight=dataparallel)
- Eventually we want to use DistributedDataParallel ([tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)) but that requires significant changes in current code. We would come back to it later. For reference, we have also provided a solution for this assignment (in `train_ddp.py`).
- How would you handle the optimization parameter when using multiple gpus, i.e., changing batch-size, lr, etc? Would multiple-gpus always make convergence faster?

## Assignment-6
Lets setup a proper multi-gpu training pipelines. 
- So far we used the `DataPrallel (DP)` approach from PyTroch, which is super easy to use but slow. Now we'll transition to `DistributedDataParallel (DDP)` which is the proper way of doing distributed training. Why? Because there are no hacks in DDP, as done throughout all distributed computation (no just ML), it would launch multiple jobs (1 per gpus in our case), and let each job handle its data, models and other artifacts separately. After each batch, it would ask the jobs to synchornize (in our case share their gradients, update the models parameters). It requires more changes to code than DP, but not too much. By default, you should support DDP in your projects, s.t., you can scale them to multiple gpus whenever needed.
- Once implemented, you'll see that your code can work with mutli-node multi-gpu setup and scale to hundreds of gpus. You can try it on multi-node setup on della, i.e., work with 2-nodes with 1 gpus from each.

## Assignment-7
Neural network training has certain instabilities. Some of them are taken care of by carefully designing the network. E.g., normalization layers brings stability as they help avoiding gradient vanishing or blowup. Similarly residual connections makes the training faster and stable. From engineering perspective, the most common trick is to warmup the training for a few epochs.
- Let's say we want to use `lr=0.1` and decay it over epochs. Training with high learning rate at start generally makes optimizer unstable and gradients might blowup. So warmup simply starts with a very small learning rate (say 1e-3) and slowly increase it upto desired level ([illustration](shorturl.at/tNU24))
- You task is to support warmup training for few epochs (let user control the number of warmup epochs).

## Assignment-8
Before we move further, lets add a powerful logging support. We need a way to log all training data (loss, acc, etc) somewhere. 
- A simple way would be to print it to a file. This can be done either by using a logger from python or by directing the stdout (that mean whatever printed on terminal) to a file (`ptyhon main.py | tee -a log.txt`)
- Another would be to use `weights & biases` which provides support to log and visualize all training logs. It is a powerful alternative to tensorboard.

## Assignment-9
Now you have a network that is very stable but super slow to train (even with 4 gpus). Can you speed it up further. 
- How about we use half-precision compute. Pytorch by default uses float32 (4-bytes for each float) datatype. If we use float-16, we can save roughly 2x on computational cost.
- You can read about half-precision training here ([Tutorial-1](https://pytorch.org/docs/stable/notes/amp_examples.html), [tutorial-2](https://huggingface.co/docs/transformers/v4.15.0/performance)). 
- You will quickly notice that everyones does a mixed-precision training, i.e., some operations are still kept in float32 format. This is because training with half-precision is rigged with issues ([here](https://discuss.pytorch.org/t/training-with-half-precision/11815)), e.g., in batch-normalization layers divides by variances, which is very small can make the output unstable. 
- Mixed precision training is quite evolved nowdays, so pytorch likely automatically would handle most things for you. Though note that mixed-precision training often fails because gradients (which tend to be in order of 1e-3-1e-6) vanishes and loss goes to inf/nan, so even it mixed-precision training is stable at the start it may fails at any random point in training (so always see mixed precision through all epochs)

## Assignment-10
When using multiple-gpus (let's say 256 with 32X8nodes) we get lot of compute power. At that point we have to ensure that gpus gets data very fast too, otherwise data loading becomes the bottleneck. Don't underestimate the times it takes to move GBs of dataset from disk to GPUs (even worse if you are reading data from hard disks or SSD). So you'll notice that most powerful gpu cluster nodes (e.g., on della) would have ~300GBs of RAM, such that you can load a vast amount of data in memory (from where its very fast to move to gpus)
- [FFCV](https://github.com/libffcv/ffcv) is a new library claiming to optimize the data loading pipelines and bringing almost magical improvements in speed our of nowhere (Frankly I'm not sure how well it would work out-of-the-box). You assigment is to incorporate ffcv in this codebase (it's pretty strightforward)
- They also have training [code](https://github.com/libffcv/ffcv-imagenet) for ImageNet. You can borrow some tricks from there, if needed. 

Additional resources:
- PyTorch official tutorials: https://pytorch.org/tutorials
- Understanding deep learning book: https://udlbook.github.io/udlbook
- CS231 course at Stanford: http://cs231n.stanford.edu

