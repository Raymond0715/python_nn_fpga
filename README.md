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
	--model <model/name> \ ## resnet20 or vgg16
	--class_num <class/number> \
	--dataset <dataset> \
	--quantilize <Choose quantization method> \ # full, ste or ng
	--quantilize_w <weights/bits/width, e.g. 32> \
	--quantilize_x <activation/bits/width, e.g. 32> \
	--weight_decay <weight/decay, e.g. 0.0005> \
	--batch_size <batch/size, e.g. 128> \
	--num_epochs <epoch/number, e.g. 250> \
	--learning_rate <learning/rate, e.g. 0.1> \
	--log_dir <log/dir, e.g. log_dir> \
	--log_file <log/file, e.g. log_file.txt> \
	--device <GPUs/index, e.g. 0,1,2,3>
```

example: 

```sh
python main.py \
	--model resnet20 \
	--class_num 10 \
	--dataset cifar10 \
	--quantilize ste \
	--quantilize_w 1 \
	--quantilize_x 1 \
	--weight_decay 0.0005 \
	--log_dir resnet20_cifar10 \
	--log_file ste.csv \
	--device 0,1,2,3

python main.py \
	--model resnet20 \
	--class_num 10 \
	--dataset cifar10 \
	--quantilize ste \
	--quantilize_w 1 \
	--quantilize_x 1 \
	--weight_decay 0.0005 \
	--log_dir resnet20_cifar10 \
	--log_file ste.csv \
	--device 0,1,2,3
```

Besides, you can run `python main.py -h` for help. 



## Tools

In 'tools' directory, you can run following command to plot loss and accuracy record in log file.

```sh
python curve.py
```

You need to change log file path and y limits in `curve.py` for different log file. 