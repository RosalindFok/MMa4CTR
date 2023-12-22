r""" 批量训练超参数 """
import os
from load_path import tiktok,wechat,finish,like

r"""lr_change传入args会变成字符串 用bool麻烦 直接用字符串
    modal_select:
        tri_concat <-> 三种模态拼接
        dou_concat <-> 两种模态拼接
        only_one   <-> 单个模态
        tri_cross  <-> 三种模态哈达玛交叉
        dou_cross  <-> 两种模态哈达玛交叉
    """

### 选择实验数据集
while (True):
    dataset = str(input(f'Select dataset from {tiktok} and {wechat}: '))
    if dataset.lower() == wechat:
        dataset = wechat
        break
    elif dataset.lower() == tiktok:
        dataset += '_'+like
        break
    else:
        print(f'Error: Your dataset = {dataset} is wrong.')

r""" 性能对比实验 """
# 1. 性能对比实验 learning rate可调节: 每5个epoch变为原来的一半 
lr_change = True
modal_select = 'tri_concat'
batch_size = 8192
learning_rate = 0.001
# for epochs in list(range(10, 101, 10)):
for epochs in [10,20,30,50]:
    os.system(f'python mlp.py --modal_select {modal_select} --lr_change {lr_change} --epochs {epochs} --learning_rate {learning_rate} --batch_size {batch_size} --dataset {dataset}')

r""" 超参数敏感性实验 """
# 共同配置
lr_change = False
modal_select = 'tri_concat'
# 2. 超参数敏感性实验---learning_rate 
batch_size = 8192
for epochs in [10, 30]:
    for learning_rate in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]:
        os.system(f'python mlp.py --modal_select {modal_select} --lr_change {lr_change} --epochs {epochs} --learning_rate {learning_rate} --batch_size {batch_size} --dataset {dataset}')

# 3. 超参数敏感性实验---batch_size
learning_rate = 0.001
for batch_size in [int(2**i) for i in range(6, 16)]:
    for epochs in [10, 30]:
        os.system(f'python mlp.py --modal_select {modal_select} --lr_change {lr_change} --epochs {epochs} --learning_rate {learning_rate} --batch_size {batch_size} --dataset {dataset}')

# 4. 超参数敏感性实验---epoch 
batch_size = 8192
learning_rate = 0.001
for epochs in list(range(5, 101, 5)):
    os.system(f'python mlp.py --modal_select {modal_select} --lr_change {lr_change} --epochs {epochs} --learning_rate {learning_rate} --batch_size {batch_size} --dataset {dataset}')

r""" 消融实验
如果报错: AttributeError: 'StepLR' object has no attribute 'StepLR'
则请手动执行下一轮训练.
"""
# 共同配置
lr_change = True
batch_size = 8192
learning_rate = 0.001
# 5. 消融实验---单一模态 
modal_select = 'only_one'
for epochs in [10, 30]:
    os.system(f'python mlp.py --modal_select {modal_select} --lr_change {lr_change} --epochs {epochs} --learning_rate {learning_rate} --batch_size {batch_size} --dataset {dataset}')

# 6. 消融实验---双模态拼接 
modal_select = 'dou_concat'
for epochs in [10, 30]:
    os.system(f'python mlp.py --modal_select {modal_select} --lr_change {lr_change} --epochs {epochs} --learning_rate {learning_rate} --batch_size {batch_size} --dataset {dataset}')

# 7. 消融实验---双模态交叉 
modal_select = 'dou_cross'
for epochs in [10, 30]:
    os.system(f'python mlp.py --modal_select {modal_select} --lr_change {lr_change} --epochs {epochs} --learning_rate {learning_rate} --batch_size {batch_size} --dataset {dataset}')

# 8. 消融实验---三模态交叉 
modal_select = 'tri_cross'
for epochs in [10, 30]:
    os.system(f'python mlp.py --modal_select {modal_select} --lr_change {lr_change} --epochs {epochs} --learning_rate {learning_rate} --batch_size {batch_size} --dataset {dataset}')

# 9. 消融实验---用户多模态嵌入   需要在get_user_embedding.py中修改user_embedding_modal_part_length的值
modal_select = 'tri_concat'
for epochs in [10, 30]:
    os.system(f'python mlp.py --modal_select {modal_select} --lr_change {lr_change} --epochs {epochs} --learning_rate {learning_rate} --batch_size {batch_size} --dataset {dataset}')

# clear git
os.system(f'git gc --prune=now')