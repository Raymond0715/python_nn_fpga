**重要声明: 该测试数据生成工程是我在进行 FPGA 开发时自用的, 并不是用户友善的工程. 部分脚本之间功能重复或者有 参数/定义 依赖等情况. 欢迎并鼓励在实际使用时, 根据用户习惯与喜好自行修改.**

量化函数整数部分改为命令行参数.

# 1 烂笔头

## 1.1 常用命令

记录常见用于生成测试数据的命令以及相关代码的修改.

- `inference.py`

- `test_postprocess.py`

  - **乘法和移位需要修改的代码**:

    - `quantization.py` 中, 函数 `QuantilizeWeight` 中的量化函数( 注: Quantilize 拼错了, 并没有这个单词).

    - `quantization.py` 中, 函数 `QuantilizeActivation` 整数部分位宽为3.

  - 生成新的数据时, 包括乘法移位操作符变化, 位宽改变, 都需要保存新的偏置数据, 即在运行程序时, 添加参数 `--ckpt_bias None --ckpt_bias_store bias_56_256.bin`.

  - 修改网络结构时, 需要修改模型定义中的参数.

  - $56 \times 56 \times 256$; 乘法; w12a12; 激活值和权重整数位宽为4.
    ```sh
    python test_postprocess.py
    ```

  - $56 \times 56 \times 256$; 移位; w4a8; 激活值整数位宽为3; 下述设置中, `quantize_w_integer` 和 `quantize_w` 为移位操作等价为乘法操作后的值.
    ```sh
    python test_postprocess.py \
    --ckpt_directory post_process_bias_shift \
    --quantize_w_integer 4 \
    --quantize_w 8 \
    --quantize_x_integer 3 \
    --quantize_x 8 \
    --output_directory post_process_shift \
    --output_conv out_56_256_conv.dat \
    --output_bias out_56_256_bias.dat \
    --output_relu out_56_256_leakyrelu.dat
    ```

  - $56 \times 56 \times 256$; 移位; w4a8; 激活值整数位宽为3; 下述设置中, `quantize_w_integer` 和 `quantize_w` 为移位操作等价为乘法操作后的值; 第二层.
    ```sh
    python test_postprocess.py \
    --img_dat img_56_256_layer2_shift.dat \
    --ckpt_directory post_process_bias_shift \
    --quantize_w_integer 4 \
    --quantize_w 8 \
    --quantize_x_integer 3 \
    --quantize_x 8 \
    --output_directory post_process_shift \
    --output_conv out_56_256_conv_layer2.dat \
    --output_bias out_56_256_bias_layer2.dat \
    --output_relu out_56_256_leakyrelu_layer2.dat
    ```

- `convert_act_structure.py`

  - $56 \times 56 \times 256$; 乘法; w4a12; 激活值整数位宽为4.
    ```sh
    python convert_act_structure.py
     ```

  - $56 \times 56 \times 256$; 移位; w3a8; 激活值整数位宽为3; 二进制上板.
    ```sh
    python convert_act_structure.py \
    --input img_56_256.bin \
    --img_size 56 \
    --img_channels 256 \
    --quantize_x_integer 3 \
    --quantize_x 8 \
    --output img_56_256_process_shift.dat \
    --bin
    ```

  - $56 \times 56 \times 256$; 移位; w3a8; 激活值整数位宽为3; 文本文件仿真.
    ```sh
    python convert_act_structure.py \
    --input img_56_256.bin \
    --img_size 56 \
    --img_channels 256 \
    --quantize_x_integer 3 \
    --quantize_x 8 \
    --output img_56_256_process_shift_sim.txt \
    --txt
    ```

- `convert_h52txt.py`

  - 乘法和移位需要修改函数 `Store4DBinConvert` 中有关量化函数的部分.

  - PC 侧模拟移位和 FPGA 侧移位所用的权重数值不一样, FPGA 侧将输入映射到高8位, 并将PC中的左右移统一转化为右移, 所以在将数据转换为 FPGA 平台需要的数据时, 需要将 PC 侧的权重数据除8, 才能得到一致的计算结果.

  - 用于上板测试的数据, 以 16 位整数的格式保存为二进制文件, 相邻两个数颠倒以满足硬件内存排布的需求.

  - 用于上板测试的数据, 4位移位, 以 16 位整数的格式保存为二进制文件.
    ```sh
    python convert_h52txt.py \
    --img_w 56 \
    --img_ch 256 \
    --quantize shift \
    --quantize_w 4 \
    --input_file weight_56_256.h5 \
    --output_file weight_56_256_shift_process_16bit.dat \
    --bin
    ```

  - 用于上板测试的数据, 8位乘法, 以 16 位整数的格式保存为二进制文件.
    ```sh
    python convert_h52txt.py \
    --img_w 56 \
    --img_ch 256 \
    --quantize mul \
    --quantize_w_integer 4 \
    --quantize_w 12 \
    --input_file weight_56_256.h5 \
    --output_file weight_56_256_mul_process_16bit.dat \
    --bin
    ```

  - 用于仿真的数据, 4位移位, 以 16 位整数的格式保存为文本文件.
    ```sh
    python convert_h52txt.py \
    --img_w 56 \
    --img_ch 256 \
    --quantize shift \
    --quantize_w 4 \
    --input_file weight_56_256.h5 \
    --output_file weight_56_256_shift_process_16bit_sim.txt \
    --txt
    ```

- `convert_bias_bin2txt.py`

  - 将存 bias 的二进制文件转换为文本文件

  - $56 \times 56 \times 256$; 移位; a16; 整数位宽为7.
    ```sh
    python convert_bias_bin2txt.py \
    --num_data 256 \
    --package_size 4 \
    --directory post_process_bias_shift \
    --input_file bias_56_256.bin \
    --output_file bias_56_256_sim.txt
    ```

- `convert_out_structure.py`

  - 数据位宽变化时, 需要修改 `--paral_out` 参数.

  - $56 \times 56 \times 256$; 激活计算结果; 乘法; a12; 激活值位宽为4.
    ```sh
    python convert_out_structure.py
    ```

  - $56 \times 56 \times 256$; 巻积计算结果; 移位; a16; 激活值整数位宽为7.
    ```sh
    python convert_out_structure.py \
    --directory post_process_shift \
    --input out_56_256_conv.dat \
    --output out_56_256_conv_process.dat \
    --quantize_x_integer 7 \
    --quantize_x 16
    ```

  - $56 \times 56 \times 256$; 激活计算结果; 移位; a8; 激活值整数位宽为3.
    ```sh
    python convert_out_structure.py \
    --directory post_process_shift \
    --paral_out 8 \
    --input out_56_256_leakyrelu.dat \
    --output out_56_256_leakyrelu_process.dat \
    --quantize_x_integer 3 \
    --quantize_x 8
    ```


## 1.2 常用数字

$56 \times 56 \times 256 = 0x8318\_8000$


# 2 数据说明

  - 按照 $(batch, channels, rows, columns)$ 排布的文件有:

    - `create_img.py` 生成的图像数据

    - `test_*.py` 保存的计算结果

  - 按照 $(batch, rows, columns, channels)$ 排布的文件有: `tensorflow` 计算过程中，按照该数据排布顺序排布

  - 按照硬件8通道并行顺序排布, 具体内容敬请期待

  - 移位映射: shift code: `lx` 左移 x 位, `rx` 右移 x 位. 最高位符号位, 0 为正, 1为负.
    | `l3`| `l2`| `l1`|  `0`| `r1`| `r2`| `r3`| `r4`|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |000|001|010|011|100|101|110|111|


# 3 脚本简介

简要介绍测试数据生成工程中各个脚本的功能

## 3.1 `create_img.py`

  - 生成不同尺寸的随机数输入数据，尺寸数据在脚本中修改

  - 参数说明:

    - `img_w`: 数据数据宽

    - `img_h`: 输入数据高

    - `img_ch`: 输入数据通道数

    - `dat_path`: 生成数据文件地址

  - 运行如下命令生成测试输入数据:

    ```sh
    python create_img.py
    ```

## 3.2 `test_conv.py`

  - 生成卷积计算测试输出数据

  - 参数说明:

    - 运行如下命令查看参数定义:
      ```sh
      python test_conv.py -h
      ```

    - (友情提醒)`quantize_w` 和 `quantize_x` 代表输入特征层数据和权重数据的数据位宽

    - (友情提醒)`image` 无效, 在脚本中修改 `image_bin_path` 参数定义输入图像数据地址

    - (友情提醒)`ckpt` 是加载权重文件时权重文件名, `output_ckpt`是保存权重文件时权重文件名

  - 模型定义在脚本 22~56 行, 可根据实际需要修改模型参数和结构

  - 运行如下命令生成测试数据: (友情提醒) 参数有默认值

    ```sh
    python test_conv.py --quantize ste --quantize_w 12 --quantize_x 12 \
    --ckpt weight_56_256.h5 --dat out_56_256.dat --output_ckpt None \
    --img_size 56 --img_channels 256
    ```

## 3.3 `test_postprocess.py`

  - 生成后处理计算测试数据数据

  - 功能说明: 包含卷积计算和后处理计算, 其中卷积计算部分与 `test_conv.py` 中基本一致, 后处理计算包括加偏置和激活

  - 代码风格和结构与 `test_conv.py` 极其相似

## 3.4 `convert_act_structure.py`

  - 将输入数据转换为适配FPGA计算的数据排布格式

  - 参数说明:

    - `img_w`: 数据数据宽

    - `img_h`: 输入数据高

    - `img_ch`: 输入数据通道数

    - `dat_raw_path`: 原始输入数据文件地址

    - `dat_path`: 输出数据文件地址

  - 运行命令:

    ```sh
    python convert_act_structure.py
    ```

## 3.5 `convert_out_structure.py`

  - 将输出数据转换为适配FPGA计算的数据排布格式，代码风格与结构和
    `convert_act_structure.py` 极其相似

## 3.6 `convert_h52txt.py`

  - 将权重数据转换为适配FPGA计算的数据排布格式

  - 参数说明:

    - `--merge_bn` 无效

    - `--bin` 代表数据为文本文件或二进制文件, **必须**选择二进制文件

  - 运行命令:

    ```sh
    python convert_h52txt.py --bin
    ```

  - (友情提醒)网络结构依赖 `test_conv.py` 中的定义

## 3.7 `calculate_config.py`

  - 计算控制信号, 控制信号定义详见 `interface.txt`


# 4 Architecture

## 4.1 AlexNet Architecture

| Layer | Input layer                      | Output layer                     | Operation                                        |
| ----- | -------------------------------- | -------------------------------- | ------------------------------------------------ |
| 1     | $3 \times 224 \times 224, 150528$ | $64 \times 54 \times 54,186624$  | $conv: 64 \times 3 \times 11 \times 11, step: 4$ |
| 2     | $64 \times 54 \times 54,186624$  | $64 \times 26 \times 26,43264$   | $mp:3,step:2$                                    |
| 3     | $64 \times 26 \times 26,43264$   | $192 \times 26 \times 26,129792$ | $conv:192 \times 64 \times 5 \times 5$           |
| 4     | $192 \times 26 \times 26,129792$ | $192 \times 12 \times 12,27648$  | $mp:3,step:2$                                    |
| 5     | $192 \times 12 \times 12,27648$  | $384 \times 12 \times 12,55296$  | $conv:384 \times 192 \times 3 \times 3$          |
| 6     | $384 \times 12 \times 12,55296$  | $384 \times 12 \times 12,55296$  | $conv:384 \times 384 \times 3 \times 3$          |
| 7     | $384 \times 12 \times 12,55296$  | $256 \times 12 \times 12,36864$  | $conv:256 \times 384 \times 3 \times 3$          |
| 8     | $256 \times 12 \times 12,36864$  | $256 \times 5 \times 5,6400$     | $mp:3,step:2$                                    |
| 9     | $256 \times 5 \times 5,6400$     | $4096 \times 1 \times 1,4096$    | $conv:4096 \times 256 \times 5 \times 5$         |
| 10    | $4096 \times 1 \times 1, 4096$   | $4096 \times 1 \times 1, 4096$   | $conv:4096 \times 4096 \times 1 \times 1$        |
| 11    | $4096 \times 1 \times 1,4096$    | $1000 \times 1 \times 1,1000$    | $conv:1000 \times 4096 \times 1 \times 1$        |


## 4.2 VGG-16 Architecture

- Convolution part

| Layer | Input Layer(3.5M/8bits)              | Output Layer(3.5M/8bits)             | Operation (7M/4bits)                |
| ----- | ------------------------------------ | ------------------------------------ | ---------------------------------------- |
| 1     | $3 \times 224 \times 224, 150528$     | $64 \times 224 \times 224, 3211264$   | $conv: 64 \times 3 \times 3 \times 3$    |
| 2     | $64 \times 224 \times 224, 3211264$   | $64 \times 224 \times 224, 3211264$   | $conv: 64 \times 64 \times 3 \times 3$   |
| 3     | $64 \times 224 \times 224, 3211264$   | $64 \times 112 \times 112, 802816$   | $mp:2,step:2$                            |
| 4     | $64 \times 112 \times 112, 802816$   | $128 \times 112 \times 112, 1605632$ | $conv: 128 \times 64 \times 3 \times 3$  |
| 5     | $128 \times 112 \times 112, 1605632$ | $128 \times 112 \times 112, 1605632$ | $conv: 128 \times 128 \times 3 \times 3$ |
| 6     | $128 \times 112 \times 112, 1605632$ | $128 \times 56 \times 56, 401408$    | $mp:2,step:2$                            |
| 7     | $128 \times 56 \times 56, 401408$    | $256 \times 56 \times 56, 802816$    | $conv: 256 \times 128 \times 3 \times 3$ |
| 8     | $256 \times 56 \times 56, 802816$    | $256 \times 56 \times 56, 802816$    | $conv: 256 \times 256 \times 3 \times 3$ |
| 9     | $256 \times 56 \times 56, 802816$    | $256 \times 56 \times 56, 802816$    | $conv: 256 \times 256 \times 3 \times 3$ |
| 10    | $256 \times 56 \times 56, 802816$    | $256 \times 28 \times 28, 200704$    | $mp:2,step:2$                            |
| 11    | $256 \times 28 \times 28, 200704$    | $512 \times 28 \times 28, 401408$    | $conv: 512 \times 256 \times 3 \times 3$ |
| 12    | $512 \times 28 \times 28, 401408$    | $512 \times 28 \times 28, 401408$    | $conv: 512 \times 512 \times 3 \times 3$ |
| 13    | $512 \times 28 \times 28, 401408$    | $512 \times 28 \times 28, 401408$    | $conv: 512 \times 512 \times 3 \times 3$ |
| 14    | $512 \times 28 \times 28, 401408$    | $512 \times 14 \times 14, 100352$    | $mp:2,step:2$                            |
| 15    | $512 \times 14 \times 14, 100352$    | $512 \times 14 \times 14, 100352$    | $conv: 512 \times 512 \times 3 \times 3$ |
| 16    | $512 \times 14 \times 14, 100352$    | $512 \times 14 \times 14, 100352$    | $conv: 512 \times 512 \times 3 \times 3$ |
| 17    | $512 \times 14 \times 14, 100352$    | $512 \times 14 \times 14, 100352$    | $conv: 512 \times 512 \times 3 \times 3$ |
| 18 | $512 \times 14 \times 14, 100352$ | $512 \times 7 \times 7, 25088$ | $mp:2,step:2$ |
| 19 | $512 \times 7 \times 7, 25088$ | $4096 \times 1 \times 1, 4096$ | $conv: 4096 \times 512 \times 7 \times 7$ |
| 20 | $4096 \times 1 \times 1, 4096$ | $4096 \times 1 \times 1, 4096$ | $conv: 4096 \times 4096 \times 1 \times 1$ |
| 21 | $4096 \times 1 \times 1, 4096$ | $1000 \times 1 \times 1, 1000$ | $conv: 1000 \times 4096 \times 1 \times 1$ |

- Address:
  - Activation offset `0x81c0ed80`

  - Activation output first address: 

    $Activation\_offset + ACT\_SIZE + 56 \times 256 \times sizeof(real\_act)$

  - Last write address `0x823b6d60`

  - First word of last line `0x823afd80`, `0x823b6a00`


## 4.3 YOLOv3 tiny

| Layer   | Input Layer                                                | Output Layer                       | Operation                                        | ram cost           |
| ------- | ---------------------------------------------------------- | ---------------------------------- | ------------------------------------------------ | ------------------ |
| 1       | $3 \times 416 \times 416, 519168$                          | $16 \times 416 \times 416,2768896$ | $conv:3 \times 16 \times 3 \times 3,432$         | 519168, **432**    |
| 2       | $16 \times 416 \times 416,2768896$                         | $16 \times 208 \times 208,692224$  | $mp:2,step:2$                                    |                    |
| 3       | $16 \times 208 \times 208,692224$                          | $32 \times 208 \times 208,1384448$ | $conv:16 \times 32 \times 3 \times 3,4608$       | 692224, **4608**   |
| 4       | $32 \times 208 \times 208,1384448$                         | $32 \times 104 \times 104,346112$  | $mp:2,step:2$                                    |                    |
| 5       | $32 \times 104 \times 104,346112$                          | $64 \times 104 \times 104,692224$  | $conv:32 \times 64 \times 3 \times 3,18432$      | 3461112, **18432** |
| 6       | $64 \times 104 \times 104,692224$                          | $64 \times 52 \times 52,173056$    | $mp:2,step:2$                                    |                    |
| 7       | $64 \times 52 \times 52,173056$                            | $128 \times 52 \times 52,346112$   | $conv:64 \times 128 \times 3 \times 3,73728$     | 173056, **73728**  |
| 8       | $128 \times 52 \times 52,346112$                           | $128 \times 26 \times 26,86528$    | $mp:2,step:2$                                    |                    |
| 9       | $128 \times 26 \times 26,86528$                            | $256 \times 26 \times 26,173056$   | $conv:128 \times 256 \times 3 \times 3,294912$   | **86528**, 294912  |
| bench 1 |                                                            |                                    |                                                  |                    |
| 10      | $256 \times 26 \times 26,173056$                           | $256 \times 13 \times 13,43264$    | $mp:2,step:2$                                    |                    |
| 11      | $256 \times 13 \times 13,43264$                            | $512 \times 13 \times 13,86528$    | $conv:256 \times 512 \times 3 \times 3,1179648$  | **43264**, 1179648 |
| 12      | $512 \times 13 \times 13,86528$                            | $512 \times 13 \times 13,86528$    | $mp:2,step:1$                                    |                    |
| 13      | $512 \times 13 \times 13,86528$                            | $1024 \times 13 \times 13,173056$  | $conv:512 \times 1024 \times 3 \times 3,4718592$ | **86528**, 4718592 |
| 14      | $1024 \times 13 \times 13,173056$                          | $256 \times 13 \times 13,43264$    | $conv:1024 \times 256 \times 1 \times 1,262400$  | **173056**, 262400 |
| 15      | $256 \times 13 \times 13,43264$                            | $512 \times 13 \times 13,86528$    | $conv:256 \times 512 \times 3 \times 3,1179648$  | **43264**, 1179648 |
| 16      | $512 \times 13 \times 13,86528$                            | $255 \times 13 \times 13,43095$    | $conv:512 \times 255 \times 1 \times 1,130560$   | **86528**, 130560  |
| bench 2 |                                                            |                                    |                                                  |                    |
| 15      | $256 \times 13 \times 13,43264$                            | $128 \times 13 \times 13,21632$    | $conv:256 \times 128 \times 1 \times 1,32768$    | 43264, **32768**   |
| 16      | $128 \times 13 \times 13,21632$                            | $128 \times 26 \times 26,86528$    | $upsample$                                       |                    |
| 17      | $128 \times 26 \times 26 + 256 \times 26 \times 26,259584$ | $384 \times 26 \times 26,259584$   | $concat$                                         |                    |
| 18      | $384 \times 26 \times 26,259584$                           | $256 \times 26 \times 26,173056$   | $conv:384 \times 256 \times 3 \times 3,884736$   | **259584**, 884736 |
| 19      | $256 \times 26 \times 26,173056$                           | $255 \times 26 \times 26,172380$   | $conv:256 \times 255 \times 1 \times 1,65280$    | 173056, **65280**  |

Total weight number:
$8845744=432+4608+18432+73728+294912+1179648+4718592+262400+1179648+130560+32768+884736+65280$

Number of data:

| time    | DDR                                                                                                                           |
| ----    | ------------------------------------------------------------------------------------------------------------------------------|
| 1       | 0 ~ 519167 (in),       692224 ~ 3461119 (out)                                                                                 |
| 2       | 0 ~ 692223 (out),      692224 ~ 3461119 (in)                                                                                  |
| 3       | 0 ~ 692223 (in),       692224 ~ 2076671 (out)                                                                                 |
| 4       | 0 ~ 346111 (out),      692224 ~ 2076671 (in)                                                                                  |
| 5       | 0 ~ 346111 (in),       692224 ~ 1384447 (out)                                                                                 |
| 6       | 0 ~ 173055 (out),      692224 ~ 1384447 (in)                                                                                  |
| 7       | 0 ~ 173055 (in),       692224 ~ 1038335 (out)                                                                                 |
| 8       | 0 ~ 86527  (out),      692224 ~ 1038335 (in)                                                                                  |
| 9       | 0 ~ 86527  (in),       692224 ~ 865279  (out)                                                                                 |
| bench 1 |                                                                                                                               |
| 10      | 0 ~ 43263  (out),      692224 ~ 865279  (in)                                                                                  |
| 11      | 0 ~ 43263  (in),       173056 ~ 259583  (out), 692224 ~ 865279 (preserve)                                                     |
| 12      | 0 ~ 86527  (out),      173056 ~ 259583  (in),  692224 ~ 865279 (preserve)                                                     |
| 13      | 0 ~ 86527  (in),       173056 ~ 346111  (out), 692224 ~ 865279 (preserve)                                                     |
| 14      | 0 ~ 43263  (out),      173056 ~ 346111  (in),  692224 ~ 865279 (preserve)                                                     |
| 15      | 0 ~ 43263  (in),       173056 ~ 259583  (out), 692224 ~ 865279 (preserve)                                                     |
| 16      | 0 ~ 43263  (preserve), 173056 ~ 259583  (in),  692224 ~ 865279 (preserve), 3461120 ~ 3504214 (fout)                           |
| bench 2 |                                                                                                                               |
| 17      | 0 ~ 43263  (in),       43264 ~ 64895    (out), 692224 ~ 865279 (preserve), 3461120 ~ 3504214 (fout)                           |
| 18      | 43264 ~ 64895 (in),    605696 ~ 692223  (out), 692224 ~ 865279 (preserve), 3461120 ~ 3504214 (fout)                           |
| 19      | 0 ~ 173055 (out),      605696 ~ 865279  (in),                              3461120 ~ 3504214 (fout)                           |
| 20      | 0 ~ 173055 (in),                                                           3461120 ~ 3504214 (fout), 3504215 ~ 3676594 (fout) |
