# 初级部分
已经完成代码实现，需要做可视化（参考叶南阳发的可视化文件）；

# 中级部分
参考文章：[Spikeformer](https://link.zhihu.com/?target=https%3A//github.com/ZK-Zhou/spikformer)
[纯加法Transformer](https://link.zhihu.com/?target=https%3A//github.com/BICLab/Spike-Driven-Transformer)
已经复制了Spikeformer的 model.py 部分，需要手写train的部分，打算采用 fashionmnist 数据集；

# 高级部分

## 数据集准备
1. **KITTI数据集**
   - 下载[原始KITTI数据集](http://www.cvlibs.net/datasets/kitti/raw_data.php)
   - 运行以下命令处理数据：
   ```bash
   wget -i splits/kitti_archives_to_download.txt -P kitti_data/
   unzip "kitti_data/*.zip" -d kitti_data/
   ```
数据集共有 175G，需要大量的空间，下载还需要挂梯子，请提前准备。

我们的默认设置需要您使用以下命令将png图像转换为jpeg格式，**该命令同时会删除原始的KITTI `.png` 文件**：
```shell
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```


## 训练命令
训练具体设置：
<!-- - t: 时间步长(time_window)
- d: 量化比特数，提供2,4和8比特 -->
- bs: batch_size
- pre: 是否加载 SNN_resnet18 在 Imagenet 上的预训练模型
- log_dir: 日志保存路径
- data_path: 数据集路径，默认为 kitti_data（option里不是，这里需要改一下）
```shell
python train.py --model_name mono_model --batch_size --log_dir 
```

## 测试命令

```shell
python evaluate_depth.py --load_weights_folder ~/tmp/snn_t=20/mono_model/models/weights_19/ --eval_mono
```