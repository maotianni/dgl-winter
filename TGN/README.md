# TGCN的复现

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

## 具体选项

```{txt}
optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             GPU device ID. Use -1 for CPU training
  --dataset DATASET
  --validation          validation
  --learn LEARN         future tasks, Link Prediction or Node Classification
  --batch-size BATCH_SIZE
  --batch-size-test BATCH_SIZE_TEST
  --num-heads NUM_HEADS
  --in-feats-t IN_FEATS_T
                        time embedding feats
  --in-feats-s IN_FEATS_S
                        memory feats
  --dropout DROPOUT
  --lr LR               lr for embedding
  --lr-p LR_P           lr for future tasks
  --decay DECAY         l2 norm for future tasks
  --n-epochs N_EPOCHS
  --log-every LOG_EVERY
  --eval-every EVAL_EVERY
  --num-neg NUM_NEG
```
