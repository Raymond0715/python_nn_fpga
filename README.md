**重要声明: 该测试数据生成工程是我在进行 FPGA 开发时自用的, 并不是用户友善的工程. 部分脚本之间功能重复或者有 参数/定义 依赖等情况. 欢迎并鼓励在实际使用时, 根据用户习惯与喜好自行修改.**

# 1 烂笔头

记录常见用于生成测试数据的命令以及相关代码的修改.


  - `test_postprocess.py`

    - **乘法和移位需要修改的代码**:

      - `quantization.py` 中, 函数 `QuantilizeWeight` 中的量化函数( 注: Quantilize 拼错了, 并没有这个单词).

      - `quantization.py` 中, 函数 `QuantilizeActivation` 整数部分位宽为3.

    - 生成新的数据时, 包括乘法移位操作符变化, 位宽改变, 都需要保存新的偏置数据, 即在运行程序时, 添加参数 `--ckpt_bias None --ckpt_bias_store bias_56_256.bin`.

    - $56 \times 56 \times 256$; 乘法; w12a12; 激活值和权重整数位宽为4.
      ```sh
      python test_postprocess.py
      ```

    - $56 \times 56 \times 256$; 移位; w4a8; 激活值整数位宽为3; 下述设置中, `quantize_w_integer` 和 `quantize_w` 为移位操作等价为乘法操作后的值.
      ```sh
      python test_postprocess.py \
      --ckpt_directory post_process_bias_shift \
      --quantize_w_integer 1 \
      --quantize_w 8 \
      --quantize_x_integer 3 \
      --quantize_x 8 \
      --output_directory post_process_shift \
      --output_conv out_56_256_conv.dat \
      --output_bias out_56_256_bias.dat \
      --output_relu out_56_256_leakyrelu.dat
      ```

  - `convert_act_structure.py`

    - $56 \times 56 \times 256$; 乘法; w4a12; 激活值整数位宽为4.
      ```sh
      python convert_act_structure.py
       ```

    - $56 \times 56 \times 256$; 移位; w3a8; 激活值整数位宽为3; 二进制上板.
      ```sh
      python convert_act_structure.py \
      --img_size 56 \
      --img_channels 256 \
      --quantize_x_integer 3 \
      --quantize_x 8 \
      --input img_56_256.bin \
      --output img_56_256_process_shift.dat \
      --bin
      ```

    - $56 \times 56 \times 256$; 移位; w3a8; 激活值整数位宽为3; 文本文件仿真.
      ```sh
      python convert_act_structure.py \
      --img_size 56 \
      --img_channels 256 \
      --quantize_x_integer 3 \
      --quantize_x 8 \
      --input img_56_256.bin \
      --output img_56_256_process_shift_sim.txt \
      --txt
      ```

  - `convert_h52txt.py`

    - 乘法和移位需要修改函数 `Store4DBinConvert` 中有关量化函数的部分.

    - 以 16 位整数的格式保存为二进制文件, 相邻两个数颠倒以满足硬件内存排布的需求.

  - `convert_out_structure.py`

    - 数据位宽变化时, 需要修改 `--paral_out` 参数.

    - $56 \times 56 \times 256$; 激活计算结果; 乘法; a12; 激活值位宽为4.
      ```sh
      python convert_out_structure.py
      ```

    - $56 \times 56 \times 256$; 巻积计算结果; 移位; a16; 激活值整数位宽为4.
      ```sh
      python convert_out_structure.py \
      --directory post_process_shift \
      --input out_56_256_conv.dat \
      --output out_56_256_conv_process.dat \
      --quantize_x_integer 4 \
      --quantize_x 16
      ```

    - $56 \times 56 \times 256$; 激活计算结果; 移位; a8; 激活值整数位宽为3.
      ```sh
      python convert_out_structure.py \
      --directory post_process_shift \
      --paral_out 16 \
      --input out_56_256_leakyrelu.dat \
      --output out_56_256_leakyrelu_process.dat \
      --quantize_x_integer 3 \
      --quantize_x 8
      ```


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
