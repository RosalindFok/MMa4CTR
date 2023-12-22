r"""
读取训练矩阵 按照8:1:1随机划分训练集 验证集 测试集
实现多层感知机模型
"""
import os, time, torch, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, auc, roc_curve, log_loss
from torch import FloatTensor, Tensor, optim, nn
from torch.optim import lr_scheduler # 学习率的降低优化策略
from torch.nn import (
    Module,
    Linear,
    Tanh,
    Sigmoid,
    Sequential
)
from load_path import torch_model_path, matrix_path, saved_path, double_modal_cross_arr, triple_modal_cross_arr, double_modal_concat_arr, triple_modal_concat_arr, each_modal_arr,tiktok,wechat
import matplotlib.pyplot as plt

if not os.path.exists(torch_model_path):
    os.mkdir(torch_model_path)

""" 超参数 """
parser = argparse.ArgumentParser(description='Hyperparameter')
parser.add_argument('--epochs', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--lr_change', type=str)
parser.add_argument('--modal_select', type=str)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()
epochs = args.epochs
learning_rate = args.learning_rate
batch_size = args.batch_size # 每个loader的规模 = 总规模 / batch_size
lr_change = args.lr_change
modal_select = args.modal_select
dataset = args.dataset
if dataset.startswith(tiktok) or dataset == wechat:
    print(f'Now is training on {dataset}')
else:
    print(f'Error: No dataset = {dataset}.')
    exit(1)
hyperparams_set = 'epochs = '+ str(epochs) +', initial learning_rate = '  + str(learning_rate) + ', batch_size = ' + str(batch_size)
print(f'{hyperparams_set}')

""" 算力设备 """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
print(f'Device = {device}')
np.random.seed(0)

""" 制作datasets """
class GetData(Dataset):
    def __init__(self, user : list, features : np.array, targets : list) -> None:
        # features为特征矩阵 每行对应到一个短视频 每列对应到一种特征
        # targets为点击率目标01值
        self.user = user
        self.features = FloatTensor(features)
        self.targets = FloatTensor(targets)
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        # 每个loader返回特征矩阵+点击目标
        return self.user[index], self.features[index], self.targets[index]
    def __len__(self) -> int:
        assert len(self.features) == len(self.targets) == len(self.user)
        return len(self.features)

""" 划分训练集 验证集 测试集 """
def get_train_value_dataloader(root_dir:str, label_tag:str):
    def make_xy(data : np.array): 
        x, y, user = [],[],[] # y向量为矩阵的最后一列 即点击的01值; x为特征矩阵; user用用户id
        for i, _ in zip(data, tqdm(range(len(data)))):
            y.append(int(i[-1]))
            tmp = [] 
            for v in i[1:-1]:
                tmp.append(float(v))
            x.append(tmp)
            user.append(int(i[0]))
        # 特征矩阵部分进行归一化
        start = time.time()
        x = MinMaxScaler().fit_transform(np.array(x))
        end = time.time()
        print(f'It took {end-start} seconds to normalize feature matrix.')
        return user, x, y
    
    def make_dataloader(user : list, x : np.array, y : list) -> DataLoader:
        dataset = GetData(user, x, y)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    # 加载训练矩阵
    file_path = os.path.join(root_dir, label_tag+'_matrix.npy')
    matrix = []
    start = time.time()
    matrix = (np.load(file_path, allow_pickle=True))
    end = time.time()
    print(f'Reading {file_path} took {round((end-start), 3)} seconds...')
    
    # 训练集 : 测试集 : 验证集 = 8 : 1 : 1
    train_matrix = matrix[ : int(len(matrix)*0.8)]
    val_matrix = matrix[int(len(matrix)*0.8) : int(len(matrix)*0.9)]
    test_matrix = matrix[int(len(matrix)*0.9) : ]

    val_user, val_x, val_y = make_xy(val_matrix)
    test_user, test_x, test_y = make_xy(test_matrix)
    train_user, train_x, train_y = make_xy(train_matrix)
    
    # 返回 train_loader, val_loader, test_loader
    return make_dataloader(train_user,train_x,train_y),make_dataloader(val_user,val_x,val_y),make_dataloader(test_user,test_x,test_y)

"""多层感知机模型"""
class MLP(Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        # 激活函数
        self.activation_function = Tanh()
        # 自适应的网络架构
        if in_features > 200 and in_features <= 500:
            self._features = Sequential(
                Linear(in_features, 256), self.activation_function,
                Linear(256, 64), self.activation_function,
                Linear(64, 16), self.activation_function,
                Linear(16, 4), self.activation_function)
        elif in_features > 100 and in_features <= 200:
            self._features = Sequential(
                Linear(in_features, 256), self.activation_function,
                Linear(256, 128), self.activation_function,
                Linear(128, 64), self.activation_function,
                Linear(64, 16), self.activation_function,
                Linear(16, 4), self.activation_function)
        elif in_features > 60 and in_features <= 100:
            self._features = Sequential(
                Linear(in_features, 64), self.activation_function,
                Linear(64, 16), self.activation_function,
                Linear(16, 4), self.activation_function)
        elif in_features > 30 and in_features <=60:
            self._features = Sequential(
                Linear(in_features, 32), self.activation_function,
                Linear(32, 16), self.activation_function,
                Linear(16, 4), self.activation_function)
        elif in_features > 10 and in_features <=30:
            self._features = Sequential(
                Linear(in_features, 16), self.activation_function,
                Linear(16, 4), self.activation_function)
        elif in_features >= 5 and in_features <=10:
            self._features = Sequential(
                Linear(in_features, 8), self.activation_function,
                Linear(8, 4), self.activation_function)
        else:
            print(f'Error: in_features = {in_features} is too big or wrong, please set appropriate in_features.')
            exit(1)
        # 分类层
        self._classifier = Linear(4, 1)
        # 预测层
        self.predict = Sigmoid() 
    
    def forward(self, x: Tensor) -> Tensor:
        x = self._features(x)
        features_x = x
        x = self._classifier(x)
        x = x.squeeze(-1)
        return features_x, self.predict(x)

# 两种方式的API计算AUC值
def calculate_AUC(pred_list : list, true_list : list):
    pred_np = np.array(pred_list)
    true_np = np.array(true_list)
    
    fpr, tpr, thresholds = roc_curve(true_np, pred_np, pos_label=1)
    roc_auc = auc(fpr, tpr)

    assert roc_auc == roc_auc_score(true_np, pred_np)
    return roc_auc, pred_list

if __name__ == '__main__':
    root_dir = matrix_path
    
    # 存放AUC和LogLoss值的字典[modal_name : key, AUC : value]
    auc_dict, logloss_dict = {}, {}

    if modal_select == 'tri_concat':
        label_tag_arr = triple_modal_concat_arr
    elif modal_select == 'tri_cross':
        label_tag_arr = triple_modal_cross_arr
    elif modal_select == 'only_one':
        label_tag_arr = each_modal_arr
    elif modal_select == 'dou_concat':
        label_tag_arr = double_modal_concat_arr
    elif modal_select == 'dou_cross':
        label_tag_arr = double_modal_cross_arr
    else:
        print(f'Error: modal_select = {modal_select}, Pleas check adjust_hyperparameter.py!')
        exit(1)

    for label_tag in label_tag_arr:
        start_time = time.time()
        label_tag = dataset + '_' + label_tag
        train_loader, val_loader, test_loader = get_train_value_dataloader(root_dir, label_tag)
        
        in_features = next(iter(train_loader))[1].shape[-1]  
        print(f'Tag = {label_tag}.\tThere are {len(train_loader)} samples in train set, {len(val_loader)} in validate set, ' + 
                f'{len(test_loader)} in test set. Features Number is: {in_features}\n')

        model = MLP(in_features)
        print(f'{str(model)}')
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'The number of trainable parametes is {trainable_parameters}')

        if torch.cuda.is_available: # 将模型迁移到GPU上
                model = model.cuda()
        # 损失函数
        loss = nn.CrossEntropyLoss() # 交叉熵损失

        # 优化函数
        optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
        
        if lr_change.lower() == 'True'.lower():
            # 学习率降低策略
            print(f'Learning rate WILL change with epochs.')
            scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) # 每5个epoch, 让learning rate降低为原来的0.5
        elif lr_change.lower() == 'False'.lower(): 
            print(f'Learning rate NOT change.')
        else:
            print(f'Error: Something wrong with lr_change, Please check adjust_hyperparameter.py')
            exit(1)

        # 利用训练集和验证集更新模型参数
        y_train_loss, y_valid_loss = [], [] # 用于损失函数值绘图 
        for ep in range(epochs):
            ep_start = time.time()
            train_loss_list = []
            
            # 训练
            model.train()
            print(f'Training...')
            pred_list = []
            true_list = []
            for (user, xt, yt), _ in zip(train_loader, tqdm(range(len(train_loader)))):
                if torch.cuda.is_available: # 将数据迁移到GPU上
                    xt, yt = xt.cuda(), yt.cuda()
                _, y_pred = model(xt)
                pred_list += y_pred.cpu()
                true_list += yt.cpu()
                l = loss(y_pred, yt)
                train_loss_list.append(l.item())
                # 反向传播的三步
                optimizer.zero_grad() # 清除梯度
                l.backward() # 反向传播
                optimizer.step() # 优化更新
            
            if lr_change.lower() == 'True'.lower(): 
                # 更新学习率
                scheduler.step()

            # 验证
            model.eval()
            print(f'Validating...')
            val_loss_list = []
            pred_list = []
            true_list = []
            with torch.no_grad():
                for (user, xv, yv), _ in zip(val_loader, tqdm(range(len(val_loader)))):
                    if torch.cuda.is_available: # 将数据迁移到GPU上
                        xv, yv = xv.cuda(), yv.cuda()
                    _, y_pred = model(xv)
                    pred_list += y_pred.cpu()
                    true_list += yv.cpu()
                    l = loss(y_pred, yv)
                    val_loss_list.append(l.item())
                    
            roc_auc, _ = calculate_AUC(pred_list, true_list)

            # 验证中断机制
            if roc_auc <= 0.5:
                learning_rate /= 2
                ep -= 1 if not ep == 0 else 0
                print(f'AUC = {roc_auc}, new learning rate ={learning_rate/10}, now epoch = {ep}, restart to train')
                continue

            ep_end = time.time()
            mean_train_loss, mean_val_loss = round(np.mean(train_loss_list),12), round(np.mean(val_loss_list),12)
            y_train_loss.append(mean_train_loss)
            y_valid_loss.append(mean_val_loss)
            print(f'epoch: {ep}; train loss: {mean_train_loss}; val loss: {mean_val_loss}; val AUC: {roc_auc}; {round((ep_end-ep_start)/60,3)} minutes.')

        assert len(y_train_loss) == len(y_valid_loss)
        # 保存训练和验证损失
        x = np.array(list(range(1, epochs+1)))
        y_train_loss = np.array(y_train_loss)
        y_valid_loss = np.array(y_valid_loss)
        assert_path = os.path.join('.', 'assert')
        if not os.path.exists(assert_path):
            os.mkdir(assert_path)
        
        # # 设置中文字体
        # Train Loss
        # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        # matplotlib.rc("font",family='AR PL UMing CN') # Linux:中文字体
        plt.title('Train Loss', fontdict={'size':20})
        # plt.title('训练损失函数值', fontdict={'size':20})
        plt.xlabel('Train Epoch', fontdict={'size':16})
        # plt.xlabel('训练周期', fontdict={'size':16})
        plt.ylabel('Loss', fontdict={'size':16})
        # plt.ylabel('损失函数值', fontdict={'size':16})
        plt.plot(x, y_train_loss, label='Train Loss', color='#000000')
        plt.savefig(os.path.join('.','assert', dataset+'_'+str(epochs)+'_TrainLoss.svg'), dpi=300, format='svg')
        plt.close()
        # Valid Loss
        # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        # matplotlib.rc("font",family='AR PL UMing CN') # Linux:中文字体
        plt.title('Valid Loss', fontdict={'size':20})
        # plt.title('验证损失函数值', fontdict={'size':20})
        plt.xlabel('Train Epoch', fontdict={'size':16})
        # plt.xlabel('训练周期', fontdict={'size':16})
        plt.ylabel('Loss', fontdict={'size':16})
        # plt.ylabel('损失函数值', fontdict={'size':16})
        plt.plot(x, y_valid_loss, label='Valid Loss', color='#000000')
        plt.savefig(os.path.join('.','assert', dataset+'_'+str(epochs)+'_ValidLoss.svg'), dpi=300, format='svg')
        print(f'Train Loss and Valid Loss Pic has been saved.\n')
        plt.close()

        # 在测试集上计算AUC, LogLoss
        print(f'Testing...')
        model.eval()
        pred_list = []
        true_list = []
        temp_path = os.path.join('..','temp')
        if not os.path.exists(temp_path):
            os.mkdir(temp_path)
        with open(os.path.join('..','temp', str(dataset)+'_mlp_hidden_layer.txt'), 'w') as f:
            with torch.no_grad():
                for (user, xv, yv), _ in zip(test_loader, tqdm(range(len(test_loader)))):
                    if torch.cuda.is_available: # 将模型和数据迁移到GPU上
                        xv, yv = xv.cuda(), yv.cuda()
                    features_x, y_pred = model(xv)
                    write_x = features_x.cpu().detach().numpy().tolist()
                    write_y = yv.cpu().detach().numpy().tolist()
                    assert len(write_x) == len(write_y)
                    for w_x, w_y in zip(write_x, write_y):
                        for x in w_x:
                            f.write(str(x)+' ')
                        f.write(str(w_y)+'\n')
                    pred_list += y_pred.cpu()
                    true_list += yv.cpu()
        
        roc_auc, pred_list = calculate_AUC(pred_list, true_list)
        y_true, y_pred = np.array(true_list), np.array(pred_list)
        logLoss = log_loss(y_true, y_pred)
        
        # 保存测试集中的预测结果 进一步的可视化增强可解释性
        if not os.path.exists(saved_path):
            os.mkdir(saved_path)
        with open(os.path.join(saved_path, str(dataset)+'_'+str(epochs)+'_predict_result.txt'), 'w') as f:
            for x,y in zip(pred_list, true_list):
                f.write(str(x)+' '+str(y)+'\n')

        # 更新AUC字典中对应项
        auc_dict[label_tag] = roc_auc
        logloss_dict[label_tag] = logLoss

        # 保存模型参数
        torch.save(model.state_dict(), os.path.join(torch_model_path, label_tag+'.pth'))
        
        end_time = time.time()
        print(f'It took {round((end_time-start_time)/60, 2) } minutes to train {label_tag} model. Test AUC = {roc_auc}. Test LogLoss = {logLoss}\n')


        # 每个模态写一次文件 记录超参数、模型架构、AUC值
        hyperparams_txt = os.path.join(saved_path, 'hyperparams.txt')
        with open(hyperparams_txt, 'a') as f:
            f.write(str(model)+'\n')
            f.write(hyperparams_set+'\n')
            f.write(label_tag+': '+str(auc_dict[label_tag])+' '+str(logloss_dict[label_tag])+'\n')
            f.write('\n')

exit(0)