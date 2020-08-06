# TGN的复现

## 依赖
* Python == 3.7.5
* dgl == 0.4.3.post2 (nightly)
* torch == 1.6.0
* pandas == 1.0.5
* numpy == 1.19.0+mkl
* scipy == 1.5.1
* sklearn == 0.23.1
* tqdm == 4.47.0

## 目前完成进度
* Basic Strategy的框架
* Basic Strategy的训练

## 目录结构
* 这里脚本文件为根目录，其与数据集位置的关系如下：
```{txt}
├── ..                          // 父目录
    ├── data
    |   ├── wikipedia.csv               
    |   └── reddit.csv
    └── .                       // 当前目录
        ├── Data.py             // 预处理数据集，获取并存储数据集信息
        ├── utils.py            // 训练或验证时用到的采样、损失函数，以及训练完节点嵌入后进行的后续任务的前向传播模型
        ├── Modules.py          // 训练节点嵌入所需要用到的各个模块，包括Time Embedding、Message Passing and Node Memory、Node Embedding
        ├── Model.py            // 模型的框架
        └── Train.py            // 主程序，包括模型的训练、验证、测试
```

## 具体选项
```{txt}
optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             GPU device ID. Use -1 for CPU training
  --dataset DATASET     wikipedia or reddit
  --validation          validation
  --learn LEARN         future tasks, Link Prediction or Node Classification
  --batch-size BATCH_SIZE
                        batch size for training
  --batch-size-test BATCH_SIZE_TEST
                        batch size for evaling
  --num-heads NUM_HEADS
                        Multi Head Attention heads
  --in-feats-t IN_FEATS_T
                        time embedding feats
  --in-feats-s IN_FEATS_S
                        memory feats
  --dropout DROPOUT     dropout rate
  --lr LR               lr for embedding
  --lr-p LR_P           lr for future task(s)
  --decay DECAY         l2 norm for future task(s)
  --n-epochs N_EPOCHS   number of epoch(s)
  --log-every LOG_EVERY
                        print training results every xx step(s)
  --eval-every EVAL_EVERY
                        eval the model every xx epoch(s)
  --num-neg NUM_NEG     for each edge, sample xx negative node pairs
```
