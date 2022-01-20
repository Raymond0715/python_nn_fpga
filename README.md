**重要声明： 该测试数据生成工程是我在进行 FPGA 开发时自用的，并不是用户友善的工程。部分脚本之间功能重复或者有 参数/定义 依赖等情况。欢迎并鼓励在实际使用时，根据用户习惯与喜好自行修改。**

# 简介

简要介绍测试数据生成工程中各个脚本的功能

- `create_img.py`：生成不同尺寸的随机数输入数据，尺寸数据在脚本中修改

    - 参数说明：
        - `img_w`：数据数据宽
        - `img_h`：输入数据高
        - `img_ch`：输入数据通道数
        - `dat_path`：生成数据文件地址

    - 运行如下命令生成测试输入数据：
        ```sh
        python create_img.py
        ```

- `test_conv.py`：生成卷积计算测试输出数据

    - 参数说明：
        - 运行如下命令查看参数定义：
            ```sh
            python test_conv.py -h
            ```
        - （友情提醒）`quantize_w` 和 `quantize_x` 代表输入特征层数据和权重数据的数据位宽
        - （友情提醒）`image` 无效，在脚本中修改 `image_bin_path` 参数定义输入图像数据地址
        - （友情提醒）`ckpt` 是加载权重文件时权重文件名，`output_ckpt`是保存权重文件时权重文件名
    
    - 模型定义在脚本 22~56 行，可根据实际需要修改模型参数和结构
    
    - 运行如下命令生成测试数据：（友情提醒）参数有默认值
        ```sh
        python test_conv.py --quantize ste --quantize_w 12 --quantize_x 12 \
        --ckpt weight_56_256.h5 --dat out_56_256.dat --output_ckpt None \
        --img_size 56 --img_channels 256
        ```

- `test_postprocess.py`：生成后处理计算测试数据数据

    - 功能说明：包含卷积计算和后处理计算，其中卷积计算部分与 `test_conv.py` 中基本一致，后处理计算包括加偏置和激活

    - 代码风格和结构与 `test_conv.py` 极其相似

- `convert_act_structure.py`：将输入数据转换为适配FPGA计算的数据排布格式

    - 参数说明：
        - `img_w`：数据数据宽
        - `img_h`：输入数据高
        - `img_ch`：输入数据通道数
        - `dat_raw_path`：原始输入数据文件地址
        - `dat_path`：输出数据文件地址
    
    - 运行命令：
        ```sh
        python convert_act_structure.py
        ```

- `convert_out_structure.py`: 将输出数据转换为适配FPGA计算的数据排布格式，代码风格与结构和 `convert_act_structure.py` 极其相似

- `convert_h52txt.py`：将权重数据转换为适配FPGA计算的数据排布格式

    - 参数说明：
        - `--merge_bn` 无效
        - `--bin` 代表数据为文本文件或二进制文件，**必须**选择二进制文件
    
    - 运行命令：
        ```sh
        python convert_h52txt.py --bin
        ```
    
    - （友情提醒）网络结构依赖 `test_conv.py` 中的定义

- `calculate_config.py`：计算控制信号，控制信号定义详见 `interface.txt`