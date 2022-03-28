# SOFTWARE
**重要声明: 该测试数据生成工程是我在进行 FPGA 开发时自用的, 并不是用户友善的工程. 部分脚本之间功能重复或者有 参数/定义 依赖等情况. 欢迎并鼓励在实际使用时, 根据用户习惯与喜好自行修改.**

量化函数整数部分改为命令行参数.

# 1 烂笔头

## 1.1 常用命令

记录常见用于生成测试数据的命令以及相关代码的修改.

转换流程:

1. 运行 `create_img.py` 创建输入

2. 运行 `inference.py` 或 `test_postprocess.py` 得到计算结果, **注意**, 如果是计算新的数据, 要创建新的, 随机初始化的权重数据并保存, 之后运行时加载保存的权重数据

3. 运行 `convert_act_structure.py` 创建 DDR layout 的激活值数据. 注意, 目前需要手动修改 `quantization.py` 中的量化函数.

4. 运行 `convert_h52txt.py` 创建 DDR layout 的权重数据. 注意, 目前需要手动修改 `quantization.py` 中的量化函数.

5. 运行 `convert_out_structure.py` 创建 DDR layout 的计算结果.

### 1.1.1 `inference.py`

  - 测试没有上采样的 YOLO 的一支:
    ```sh
    python inference.py \
    --img img_416_8.bin \
    --img_size 416 \
    --img_channels 8 \
    --ckpt yolo_tiny_bench1.h5 \
    --output out_yolo_bench1.dat
    ```

  - 测试 $56 \times 56 \times 256$ 数据, 测试代码正确
    ```sh
    python inference.py \
    --img img_56_256.bin \
    --img_size 56 \
    --img_channels 256 \
    --ckpt weight_56_256.h5 \
    --output test4yolo_out_56_256.dat
    ```

  - 测试 $208 \times 208 \times 16$ 移位, 量化参数和 `test_postprocess.py` 相同
    ```sh
    python inference.py \
    --img img_208_16.bin \
    --img_size 208 \
    --img_channels 16 \
    --ckpt weight_208_16.h5 \
    --output out_208_16.dat
    ```

  - 测试 $104 \times 104 \times 32$ 移位, 量化参数和 `test_postprocess.py` 相同
    ```sh
    python inference.py \
    --img img_104_32.bin \
    --img_size 104 \
    --img_channels 32 \
    --ckpt weight_104_32.h5 \
    --output out_104_32.dat
    ```


### 1.1.2 `test_postprocess.py`

  - **乘法和移位需要修改的代码**:

    - `quantization.py` 中, 函数 `QuantilizeWeight` 中的量化函数( 注: Quantilize 拼错了, 并没有这个单词).

    - `quantization.py` 中, 函数 `QuantilizeActivation` 整数部分位宽为3.

  - 生成新的数据时, 包括乘法移位操作符变化, 位宽改变, 都需要保存新的偏置数据, 即在运行程序时, 添加参数 `--ckpt_bias None --ckpt_bias_store bias_56_256.bin`.

  - 修改网络结构时, 需要修改模型定义中的参数.

  - $56 \times 56 \times 256$; 乘法; w12a12; 激活值和权重整数位宽为4.
    ```sh
    python test_postprocess.py
    ```

  - $56 \times 56 \times 256$; 移位; w4a8; 输入整数位宽为3; 下述设置中, `quantize_w_integer` 不生效, 输入值为16位, 整数位宽为7.
    ```sh
    python test_postprocess.py \
    --ckpt_directory post_process_bias_shift \
    --quantize shift \
    --quantize_w_integer 4 \
    --quantize_w 4 \
    --quantize_x_integer 3 \
    --quantize_x 8 \
    --quantize_o_integer 7 \
    --quantize_o 16 \
    --output_directory post_process_shift \
    --output_conv out_56_256_conv.dat \
    --output_bias out_56_256_bias.dat \
    --output_relu out_56_256_leakyrelu.dat
    ```

  - $28 \times 28 \times 512$; 移位; w4a8; 输入整数位宽为3; 下述设置中, `quantize_w_integer` 不生效, 输入值为16位, 整数位宽为7.
    ```sh
    python test_postprocess.py \
    --img_dat img_28_512.dat \
    --img_size 28 \
    --img_channels 512 \
    --ckpt weight_28_512.h5 \
    --ckpt_filter_num 512 \
    --ckpt_bias bias_28_512.dat \
    --ckpt_directory post_process_bias_shift \
    --quantize shift \
    --quantize_w_integer 4 \
    --quantize_w 4 \
    --quantize_x_integer 3 \
    --quantize_x 8 \
    --quantize_o_integer 7 \
    --quantize_o 16 \
    --output_directory post_process_shift \
    --output_conv out_28_512_conv.dat \
    --output_bias out_28_512_bias.dat \
    --output_relu out_28_512_leakyrelu.dat
    ```

  - $56 \times 56 \times 256$; 移位; w4a8; 激活值整数位宽为3; 下述设置中, `quantize_w_integer` 不生效; 第二层; 输入值为16位, 整数位宽为7.
    ```sh
    python test_postprocess.py \
    --img_dat img_56_256_layer2_shift.dat \
    --ckpt_directory post_process_bias_shift \
    --quantize shift \
    --quantize_w_integer 4 \
    --quantize_w 4 \
    --quantize_x_integer 3 \
    --quantize_x 8 \
    --quantize_o_integer 7 \
    --quantize_o 16 \
    --output_directory post_process_shift \
    --output_conv out_56_256_conv_layer2.dat \
    --output_bias out_56_256_bias_layer2.dat \
    --output_relu out_56_256_leakyrelu_layer2.dat
    ```

### 1.1.3 `convert_act_structure.py`

  - 对于输入通道小于8的情况, 将输入通道补成8来处理.

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

  - $208 \times 208 \times 16$; 移位; w3a8; 二进制上板
    ```sh
    python convert_act_structure.py \
    --input img_208_16.bin \
    --img_size 208 \
    --img_channels 16 \
    --quantize_x_integer 3 \
    --quantize_x 8 \
    --output img_208_16_process_shift.dat \
    --bin
    ```

  - $208 \times 208 \times 16$; 移位; w3a8; 仿真
    ```sh
    python convert_act_structure.py \
    --input img_208_16.bin \
    --img_size 208 \
    --img_channels 16 \
    --quantize_x_integer 3 \
    --quantize_x 8 \
    --output img_208_16_process_shift_sim.txt \
    --txt
    ```

  - $104 \times 104 \times 32$; 移位; w3a8; 二进制上板
    ```sh
    python convert_act_structure.py \
    --input img_104_32.bin \
    --img_size 104 \
    --img_channels 32 \
    --quantize_x_integer 3 \
    --quantize_x 8 \
    --output img_104_32_process_shift.dat \
    --bin
    ```

  - $416 \times 416 \times 8$; 移位; w3a8; 仿真
    ```sh
    python convert_act_structure.py \
    --input img_416_8.bin \
    --img_size 416 \
    --img_channels 8 \
    --quantize_x_integer 3 \
    --quantize_x 8 \
    --output img_416_8_process_shift.dat \
    --bin
    ```

  - $13 \times 13 \times 256$; 移位; w3a8; 仿真
    ```sh
    python convert_act_structure.py \
    --input input_yolo_bench1_l11.dat \
    --img_size 13 \
    --img_channels 256 \
    --quantize_x_integer 3 \
    --quantize_x 8 \
    --output img_input_yolo_bench1_sim.txt \
    --txt
    ```


### 1.1.4 `convert_h52txt.py`

  - 乘法和移位需要修改函数 `Store4DBinConvert` 中有关量化函数的部分.

  - PC 侧模拟移位和 FPGA 侧移位所用的权重数值不一样, FPGA 侧将输入映射到高8位, 并将PC中的左右移统一转化为右移, 所以在将数据转换为 FPGA 平台需要的数据时, 需要将 PC 侧的权重数据除8, 才能得到一致的计算结果.

  - 对于输出通道小于 64 的情况, 将数据补为输出通道 64 来处理.

  - 用于保存的量化函数已在 `Round2Fixed` 和 `RoundPower2` 的基础上添加 `Fix2Int` 的转整数函数.

  - 用于上板测试的数据, 以 16 位整数的格式保存为二进制文件, 相邻两个数颠倒以满足硬件内存排布的需求.

  - 用于上板测试的数据, 4位移位, 以 16 位整数的格式保存为二进制文件.
    ```sh
    python convert_h52txt.py \
    --img_w 56 \
    --img_ch 256 \
    --quantize_w_method shift \
    --quantize_w 4 \
    --input_file weight_56_256.h5 \
    --output_file_weight weight_56_256_shift_process_16bit.dat \
    --bin
    ```

  - 用于上板测试的数据, 8位乘法, 以 16 位整数的格式保存为二进制文件.
    ```sh
    python convert_h52txt.py \
    --img_w 56 \
    --img_ch 256 \
    --quantize_w_method mul \
    --quantize_w_integer 4 \
    --quantize_w 12 \
    --input_file weight_56_256.h5 \
    --output_file_weight weight_56_256_mul_process_16bit.dat \
    --bin
    ```

  - 用于仿真的数据, 4位移位, 以 16 位整数的格式保存为文本文件.
    ```sh
    python convert_h52txt.py \
    --img_w 56 \
    --img_ch 256 \
    --quantize_w_method shift \
    --quantize_w 4 \
    --input_file weight_56_256.h5 \
    --output_file_weight weight_56_256_shift_process_16bit_sim.txt \
    --txt
    ```

  - YOLO 用于上板测试的数据, $3 \times 3 \times 16 \times 32$, 4 位移位, 以 16 位整数的格式保存为二进制文件. 由于默认权重转换的并行度是64, 大于这一层的输出通道数, 因此这一层的数据需要特殊处理. 设置`paral_w` 为 32. 得到的结果用 `weight_assign4hw.py` 处理一次.
    ```sh
    python convert_h52txt.py \
    --img_w 208 \
    --img_ch 16 \
    --paral_w 32 \
    --quantize_w_method shift \
    --quantize_w_integer 4 \
    --quantize_w 4 \
    --quantize_b_method mul \
    --quantize_b_integer 7 \
    --quantize_b 16 \
    --input_file weight_208_16.h5 \
    --output_file_weight weight_208_16_shift_process.dat \
    --output_file_bias bias_208_16_shift_process.dat \
    --bin
    ```

  - YOLO 用于仿真的数据.
    ```sh
    python convert_h52txt.py \
    --img_w 208 \
    --img_ch 16 \
    --paral_w 32 \
    --quantize_w_method shift \
    --quantize_w_integer 4 \
    --quantize_w 4 \
    --quantize_b_method mul \
    --quantize_b_integer 7 \
    --quantize_b 16 \
    --input_file weight_208_16.h5 \
    --output_file_weight weight_208_16_shift_process_sim.txt \
    --output_file_bias bias_208_16_shift_process_sim.txt \
    --txt
    ```

  - YOLO 用于上板测试的数据, $3 \times 3 \times 32 \times 64$, 4 位移位.
    ```sh
    python convert_h52txt.py \
    --img_w 104 \
    --img_ch 32 \
    --paral_w 64 \
    --quantize_w_method shift \
    --quantize_w_integer 4 \
    --quantize_w 4 \
    --quantize_b_method mul \
    --quantize_b_integer 7 \
    --quantize_b 16 \
    --input_file weight_104_32.h5 \
    --output_file_weight weight_104_32_shift_process.dat \
    --output_file_bias bias_104_32_shift_process.dat \
    --bin
    ```

  - YOLO 用于上板测试的数据, 测试硬件适配后的不含上采样的完整一支 YOLO 的结果.
    ```sh
    python convert_h52txt.py \
    --img_w 416 \
    --img_ch 8 \
    --paral_w 64 \
    --quantize_w_method shift \
    --quantize_w_integer 4 \
    --quantize_w 4 \
    --quantize_b_method mul \
    --quantize_b_integer 7 \
    --quantize_b 16 \
    --input_file yolo_tiny_bench1.h5 \
    --output_file_weight weight_yolo_tiny_bench1_shift_process.dat \
    --output_file_bias bias_yolo_tiny_bench1_shift_process.dat \
    --bin
    ```


### 1.1.5 `convert_bias_bin2txt.py`

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

### 1.1.6 `convert_out_structure.py`

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

  - $208 \times 208 \times 16$; 激活计算结果; 移位; a8; 激活值整数位宽为3.
    ```sh
    python convert_out_structure.py \
    --directory yolo \
    --paral_out 8 \
    --input out_208_16.dat \
    --output out_208_16_process.dat \
    --img_size 208 \
    --img_channels 32 \
    --quantize_x_integer 3 \
    --quantize_x 8
    ```

  - $104 \times 104 \times 32$; 激活计算结果; 移位; a8; 激活值整数位宽为3.
    ```sh
    python convert_out_structure.py \
    --directory yolo \
    --paral_out 8 \
    --input out_104_32.dat \
    --output out_104_32_process.dat \
    --img_size 104 \
    --img_channels 64 \
    --quantize_x_integer 3 \
    --quantize_x 8
    ```

  - YOLO test one bench; 激活计算结果; 移位; a8; 激活值整数位宽为3.
    ```sh
    python convert_out_structure.py \
    --directory yolo \
    --paral_out 8 \
    --input out_yolo_bench1.dat \
    --output out_yolo_bench1_process.dat \
    --img_size 13 \
    --img_channels 256 \
    --quantize_x_integer 3 \
    --quantize_x 8
    ```


### 1.1.7 `calculate_config.py`

SDK 配置寄存器说明

| 地址 |   位宽   | 说明 |
|:----:|:--------:|:----|
|0x000 | [ 0:  0] | conf_33. Choose $3 \times 3$ convolution or $1 \times 1$ convolution.
|      |          | 0 indicate $3 \times 3$ convolution.
|      |          | 1 indicate $1 \times 1$ convolution.
|      | [ 1:  1] | act source.
|      |          | 0 indicate DDR source.
|      |          | 1 indicate SDK source.
|      | [ 2:  2] | weight or bias source
|      |          | 0 indicate DDR source.
|      |          | 1 indicate SDK source.
|      | [ 3:  3] | ReLU mode.
|      |          | 0 indicate ReLU.
|      |          | 1 indicate LeakyReLU.
|      | [ 4:  4] | ReLU switch
|      |          | 0 indicate ReLU off.
|      |          | 1 indicate ReLU on.
|      | [ 5:  5] | bias switch
|      |          | 0 indicate bias calculation off.
|      |          | 1 indicate bias calculation on.
|      | [ 6:  6] | sampling switch
|      |          | 0 indicate sampling off.
|      |          | 1 indicate sampling on.
|      | [ 7:  7] | bitintercept switch. Useless for now.
|      | [16:  8] | img_h, which is also img_w
|      | [17: 17] | Output sink.
|      |          | 0 indicate DDR is output sink.
|      |          | 1 indicate SDK is output sink.
|      | [18: 18] | Initial weight.
|      |          | 0 indicate it is calculate state.
|      |          | 1 indicate it is initial weight state.
|      | [19: 19] | finish.
|      |          | 0 indicate configuration state not finish.
|      |          | 1 indicate configuration state finish.
|      | [20: 20] | reset.
|      |          | 1 indicate reset.
|      |          | 0 indicate not to reset.
|0x020 | [11:  0] | input channel
|      | [23: 12] | output channel
|0x040 |          | Act write address. Only useful when act source is SDK.
|0x060 |          | Act read address.
|0x080 |          | Weight write address. Only useful when act source is SDK.
|0x0a0 |          | Weight read address.
|0x0c0 |          | Weight write length. Count in element.
|0x0e0 |          | Bias write address.
|0x100 |          | Bias read address.
|0x120 |          | Bias write length. Count in element.
|0x140 |          | DDR write address.

YOLO 寄存器 0 配置信息
```sh
[INFO][calculate_config.py] config_value init_weight : 0x000c000e
[INFO][calculate_config.py] config_value       reset : 0x00100000
[INFO][calculate_config.py] config_value     layer_1 : 0x0001a07a
[INFO][calculate_config.py] config_value     layer_3 : 0x0000d078
[INFO][calculate_config.py] config_value     layer_5 : 0x00006878
[INFO][calculate_config.py] config_value     layer_7 : 0x00003478
[INFO][calculate_config.py] config_value     layer_9 : 0x00001a78
[INFO][calculate_config.py] config_value    layer_11 : 0x00000d38
[INFO][calculate_config.py] config_value    layer_13 : 0x00000d38
[INFO][calculate_config.py] config_value    layer_14 : 0x00000d39
[INFO][calculate_config.py] config_value    layer_15 : 0x00000d38
[INFO][calculate_config.py] config_value    layer_16 : 0x00080d29
```


## 1.2 常用数字

$56 \times 56 \times 256 = 0x8318\_8000$


# 2 数据说明

## 2.1 数据排布

数据排布有以下三类

  - 按照 $(batch, channels, rows, columns)$ 排布的文件有:

    - `create_img.py` 生成的图像数据

    - `test_*.py` 保存的计算结果

  - 按照 $(batch, rows, columns, channels)$ 排布的文件有: `tensorflow` 计算过程中，按照该数据排布顺序排布

  - 按照硬件8通道并行顺序排布, 具体内容敬请期待


## 2.2 移位原理简介

### 2.3.1 量化函数

  - 量化函数 $Q_f(x,i,k)$ 把浮点数x转换为整数部分位为i位,位宽为k位的定点数

  - 量化函数 $Q_p(x)=-4\times(Sign(x)-1)+( \log_2(|x|))_2$

>注：$-4(Sign(x)-1)$是为了把x的符号位加到这个四位的二进制数中，若是正值则不加，若是负数则最前面一位+1

### 2.3.2 前处理过程

  - 传统前处理：$Activation=weight$$*$$input+bias$

  - 移位前处理：$Activation=Q_f(input,3,5)$ $*$ $Q_p(weight)+Q_f(bias,7,16)$

### 2.3.3 移位映射

  - 移位映射: shift code: `lx` 左移 x 位, `rx` 右移 x 位. 最高位符号位, 0 为正, 1为负.

  - 当weight为正值时，16位数的符号位继承自4位数符号位，为0，其余三位为移动的位数
    |移位操作| `l3`| `l2`| `l1`|  `0`| `r1`| `r2`| `r3`| `r4`|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |w二进制数值|0000|0001|0010|0011|0100|0101|0110|0111|

  - 当weight为负值时，16位数的符号位继承自4位数符号位，为1，其余三位为移动的位数
    |移位操作| `l3`| `l2`| `l1`|  `0`| `r1`| `r2`| `r3`| `r4`|
    |:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    |w二进制数值|1000|1001|1010|1011|1100|1101|1110|1111|


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


## 4.4 YOLO Tiny for Hardware

| Layer   | Input Layer                                                | Output Layer                       | Operation                                        | ram cost           |
| ------- | ---------------------------------------------------------- | ---------------------------------- | ------------------------------------------------ | ------------------ |
| 1       | $8 \times 416 \times 416, 1384448$                         | $64 \times 416 \times 416,11075584$ | $conv:8 \times 64 \times 3 \times 3,4608$         | 11075584, **4608**    |
| 2       | $64 \times 416 \times 416,11075584$                         | $64 \times 208 \times 208,2768896$  | $mp:2,step:2$                                    |                    |
| 3       | $64 \times 208 \times 208,2768896$                          | $64 \times 208 \times 208,2768896$ | $conv:64 \times 64 \times 3 \times 3,36864$       | 692224, **36864**   |
| 4       | $64 \times 208 \times 208,2768896$                         | $64 \times 104 \times 104,692224$  | $mp:2,step:2$                                    |                    |
| 5       | $64 \times 104 \times 104,692224$                          | $64 \times 104 \times 104,692224$  | $conv:64 \times 64 \times 3 \times 3,18432$      | 692224, **36864** |
| 6       | $64 \times 104 \times 104,692224$                          | $64 \times 52 \times 52,173056$    | $mp:2,step:2$                                    |                    |
| 7       | $64 \times 52 \times 52,173056$                            | $128 \times 52 \times 52,346112$   | $conv:64 \times 128 \times 3 \times 3,73728$     | 173056, **73728**  |
| 8       | $128 \times 52 \times 52,346112$                           | $128 \times 26 \times 26,86528$    | $mp:2,step:2$                                    |                    |
| 9       | $128 \times 26 \times 26,86528$                            | $256 \times 26 \times 26,173056$   | $conv:128 \times 256 \times 3 \times 3,294912$   | **86528**, 294912  |
| bench 1 |                                                            |                                    |                                                  |                    |
| 10      | $256 \times 26 \times 26,173056$                            | $512 \times 26 \times 26,346112$    | $conv:256 \times 512 \times 3 \times 3,1179648$  | **173056**, 1179648 |
| 11      | $512 \times 26 \times 26,346112$                           | $512 \times 13 \times 13,86528$    | $mp:2,step:2$                                    |                    |
| 12      | $512 \times 13 \times 13,86528$                            | $1024 \times 13 \times 13,173056$  | $conv:512 \times 1024 \times 3 \times 3,4718592$ | **86528**, 4718592 |
| 13      | $1024 \times 13 \times 13,173056$                          | $256 \times 13 \times 13,43264$    | $conv:1024 \times 256 \times 1 \times 1,262400$  | **173056**, 262400 |
| 14      | $256 \times 13 \times 13,43264$                            | $512 \times 13 \times 13,86528$    | $conv:256 \times 512 \times 3 \times 3,1179648$  | **43264**, 1179648 |
| 15      | $512 \times 13 \times 13,86528$                            | $256 \times 13 \times 13,43264$    | $conv:512 \times 256 \times 1 \times 1,131072$   | **86528**, 131072  |
| bench 2 |                                                            |                                    |                                                  |                    |
| 14      | $256 \times 13 \times 13,43264$                            | $128 \times 13 \times 13,21632$    | $conv:256 \times 128 \times 1 \times 1,32768$    | 43264, **32768**   |
| 15      | $128 \times 13 \times 13,21632$                            | $128 \times 26 \times 26,86528$    | $upsample$                                       |                    |
| 16      | $128 \times 26 \times 26 + 256 \times 26 \times 26,259584$ | $384 \times 26 \times 26,259584$   | $concat$                                         |                    |
| 17      | $384 \times 26 \times 26,259584$                           | $256 \times 26 \times 26,173056$   | $conv:384 \times 256 \times 3 \times 3,884736$   | **259584**, 884736 |
| 18      | $256 \times 26 \times 26,173056$                           | $256 \times 26 \times 26,172380$   | $conv:256 \times 255 \times 1 \times 1,65280$    | 173056, **65280**  |
