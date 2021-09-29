# LAIC2021—刑期预测

该项目为 **LAIC2021—**刑期预测的baseline运行介绍。

## 一、运行环境

参考baseline中的requirements.txt。

### 二、 训练

#### 2.1  训练

训练参数可在`./config/application.yml`中调整。

```python
python cnn_regression.py train
```

#### 3.2 训练成果

训练完成后最优模型保存在`./output`中：

```
├── output
│   ├── best_model.h5
│   ├── tokenizer.pickle
```

### 3.3 模型预测

`刑期预测赛道一期数据_测试集`为待预测文件，`result`为输出文件。

```
python cnn_regression.py predict
```

### 3. 4 评估

评估预测结果的准确率（偏离度25%以内为正确）。

```python
python cnn_regression.py evalute
```

## 问题反馈

如有问题，请在QQ群：521382653中反馈
