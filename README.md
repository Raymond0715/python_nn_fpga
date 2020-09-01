# Deep Quantized Neural Networks on Classification tasks using `TensorFlow 2.3`

This is an example for quantized neural networks using "straight-through estimator (ste)" and "natural gradient (ng)" introduced in "XXXXXXX". 



Supported datasets: Cifar 10/100

Supported Model: resnet 20 and vgg 16



## Dependencies

- `TensorFlow 2.3.0`
- `numpy 1.19.0`



## Run command

```sh
python main.py \
	--model <model/name> \ # Must be one of resnet20 and vgg16
	--pretrain_path <pretrain/path, default = None> \
	--class_num <class/number> \
	--dataset <dataset> \ # Must be one of cifar10 and cifar100
	--quantilize <Choose quantization method> \ # Must be one of full, ste and ng
	--quantilize_w <weights/bits/width, e.g. 32> \ # Weights bits width for quantilize model
	--quantilize_x <activation/bits/width, e.g. 32> \ # Activation bits width for quantilize model
	--weight_decay <weight/decay, e.g. 0.0005> \
	--batch_size <batch/size, e.g. 128> \
	--num_epochs <epoch/number, e.g. 250> \
	--learning_rate <learning/rate, e.g. 0.1> \
	--log_dir <log/dir, e.g. log_dir> \
	--log_file <log/file, e.g. log_file.txt> \
	--ckpt_dir <ckpt/dir, e.g. ckpt> \
	--ckpt_file <ckpt/file, e.g. model> \
	--device <GPUs/index, e.g. 0,1,2,3>
```

example: 

```sh
python main.py \
	--model resnet20 \
	--pretrain_path resnet20_cifar10/ste.h5 \
	--class_num 10 \
	--dataset cifar10 \
	--quantilize ste \
	--quantilize_w 1 \
	--quantilize_x 1 \
	--weight_decay 0.0005 \
	--log_dir resnet20_cifar10 \
	--log_file ste.csv \
	--ckpt_dir resnet20_cifar10 \
	--ckpt_file ste.h5 \
	--device 0,1,2,3

python main.py \
	--model vgg16 \
	--pretrain_path vgg16_cifar10/ng_alpha_0.5c.h5 \
	--class_num 10 \
	--dataset cifar10 \
	--quantilize ste \
	--quantilize_w 1 \
	--quantilize_x 1 \
	--weight_decay 0.0005 \
	--log_dir vgg16_cifar10 \
	--log_file ste.csv \
	--ckpt_dir vgg16_cifar10 \
	--ckpt_file ste.h5 \
	--device 0,1,2,3

```

Besides, you can run `python main.py -h` for help. 



## Tools

In 'tools' directory, you can run following command to plot loss and accuracy record in log file.

```sh
python curve.py
```

You need to change log file path and y limits in `curve.py` for different log file. 