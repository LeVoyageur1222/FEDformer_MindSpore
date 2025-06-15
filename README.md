# FEDformer_MindSpore

本仓库为北京航空航天大学计算机学院-研究生课程-AI框架与科学计算课程大作业仓库。项目基本信息如下：

| 科学任务   | FEDformer求解时间序列预测问题                                |
| ---------- | ------------------------------------------------------------ |
| 论文标题   | FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting |
| 科学领域   | 时序                                                         |
| 论文链接   | [FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting](https://proceedings.mlr.press/v162/zhou22g) |
| 原仓库链接 | [MAZiqing/FEDformer](https://github.com/MAZiqing/FEDformer)  |

本次大作业基于MindSpore昇思深度学习框架复现FEDformer模型，目标是在保障模型结构与原论文保持一致的前提下完成实验验证。具体而言，我们在不改动网络主体结构的基础上，将模型中的FFT模块、序列分解模块、频域注意力模块等关键组件逐一迁移至MindSpore，并实现了包括FourierBlock、FourierCrossAttention在内的核心模块复现。在训练与推理过程中，充分利用MindSpore支持的动态图机制进行调试验证，同时结合原项目配置重构了数据加载与训练流程，实现在多数据集上训练与评估。

## 项目结构说明

```
FEDformer_MindSpore/
├── run_ms.py                           # 主运行脚本，支持不同模型（如 FEDformer、Autoformer）的训练与测试入口
├── Experiments Results.md              # 实验结果与模型对比整理
├── data_provider/                      # 数据加载与预处理模块
│   ├── data_factory_ms.py              # 数据读取和配置类
│   └── dataset_loader.py               # MindSpore Dataset 数据封装
├── exp/                                # 训练与测试的实验管理模块
│   ├── exp_basic_ms.py                 # 实验基类，定义训练、验证、测试流程
│   └── exp_main_ms.py                  # 主实验流程控制（指定模型、数据等）
├── layers/                             # 模型核心结构模块（注意力、时序编码、频域模块等）
│   ├── AutoCorrelation.py              # Autoformer的自相关模块
│   ├── Autoformer_EncDec.py            # Autoformer的编码器解码器结构
│   ├── Embed.py                        # 输入时间嵌入模块
│   ├── FourierCorrelation.py           # FEDformer中的频域注意力模块
│   ├── MultiWaveletCorrelation.py      # Wavelet相关结构（若使用）
│   ├── SelfAttention_Family.py         # 包含稀疏注意力机制的模块
│   ├── Transformer_EncDec.py           # Transformer结构的编码器与解码器
│   └── utils.py                        # 注意力机制中用到的辅助函数
├── models/                             # 模型主结构
│   ├── Autoformer.py                   # Autoformer模型定义
│   ├── FEDformer.py                    # FEDformer模型定义
│   ├── Informer.py                     # Informer模型定义
│   └── Transformer.py                  # Baseline Transformer模型定义
├── scripts/                            # Shell 脚本运行示例
│   ├── run_Auto.sh                     # 运行Autoformer模型示例
│   └── run_M.sh                        # 运行全部模型示例
├── utils/                              # 通用工具模块
│   ├── masking.py                      # 用于构造注意力掩码
│   ├── metrics.py                      # 评估指标（如 MSE、MAE）实现
│   ├── timefeatures.py                 # 时间特征编码函数
│   └── tools.py                        # 其他辅助函数
```

其中数据获取参考[Autoformer](https://github.com/thuml/Autoformer)，在对应链接[Autoformer - Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)下可以获取。获取到的数据放在根目录`dataset`目录下，具体数据目录结构如下所示：

```
FEDformer_MindSpore/
├── dataset/
│   ├── electricity/
│   ├── exchange_rate/
│   ├── ...
```

## 环境依赖说明

建议在 MindSpore GPU 环境下运行，推荐配置如下：

- Python ≥ 3.8
- MindSpore ≥ 2.2.0（必须支持 `ops.FFTWithSize`）
- NumPy ≥ 1.21
- tqdm（用于训练可视化）
- matplotlib（可选，用于结果可视化）

安装依赖示例（基于 Conda 环境）：

```bash
conda create -n fedformer_ms python=3.9
conda activate fedformer_ms

pip install mindspore-gpu==2.2.0  # 根据实际环境选择对应版本
pip install numpy tqdm matplotlib
```

## 项目运行说明

项目的主运行脚本为根目录下的`run_ms.py`，设置实验参数和实际运行的脚本为`scripts/run_M.sh`，运行命令如下

```bash
bash scripts/run_M.sh
```

