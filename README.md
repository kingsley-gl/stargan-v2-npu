## StarGan v2 迁移npu 
#
[详细操作请参考官方链接](https://github.com/clovaai/stargan-v2)  
[数据集参考paddle的afhq和celebahq的公开数据](https://aistudio.baidu.com/aistudio/datasetoverview)

### 运行环境  
CANN 6.0.1(商用) 6.3.RC1.alpha001(社区)  
pytorch 1.8.1 (20221230)  
torch_npu (20221230)  


### 单机单卡
`
python main.py  
--mode train --num_domains 2 --w_hpf 1 
--lambda_reg 1 --lambda_sty 1 --lambda_ds 1
--lambda_cyc 1 --train_img_dir /path/to/dataset/train
--val_img_dir /path/to/dataset/val --batch_size 24 --num_workers 0
--amp --epoch 10
`

### 单机多卡
`
python main.py  
--mode train --num_domains 2 --w_hpf 1 
--lambda_reg 1 --lambda_sty 1 --lambda_ds 1
--lambda_cyc 1 --train_img_dir /path/to/dataset/train
--val_img_dir /path/to/dataset/val --batch_size 24 --num_workers 0
--amp --epoch 10 --distribute
`

### 参数详解:  
| 参数 |    功能     |
| :---:|:---------:|
|amp| 开启或关闭混合精度 |
|epoch| 设置训练轮次|
|distribute| 单机多卡训练|
更多详细参数功能参考官方链接

## Realease Note

### 2023.3.1
- 删除total_iter参数, 修改为根据数据集和batch_size大小自动计算获得
- 添加epoch参数, 可设置训练轮次
