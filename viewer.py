r"""
将测试集中的点击率预估结果进行可视化表达 提升模型的可解释性
1. ROC曲线图
2. 原始数据的UMAP分析
3. MLP隐藏层输出的UMAP分析

!!!Attention: There is no 'Times New Roman' in Linux.
"""

import numpy as np, os, re, time
import matplotlib.pyplot as plt
from load_path import saved_path, matrix_path
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import umap

""" 绘制ROC曲线 """
if os.path.exists(saved_path):
    all_files = os.listdir(saved_path)
    for file in all_files:
        if file.endswith('_predict_result.txt'):
            epoch = file.split('_')[0]
            print(epoch)

            pred_list, true_list = [], []
            with open(os.path.join(saved_path,file), 'r') as f:
                for line in f:
                    arr = re.findall('\d+\.?\d*', line)
                    pred_list.append(float(arr[0]))
                    true_list.append(float(arr[1]))

            assert len(pred_list) == len(true_list)
            pred_np, true_np = np.array(pred_list), np.array(true_list)
            fpr, tpr, thresholds = roc_curve(true_np, pred_np, pos_label=1)
            
            # X轴-fpr Y轴-tpr
            plt.title(f'ROC: Epoch = {epoch}', fontdict={'family':'Times New Roman','size':20})
            plt.xlabel('FPR', fontdict={'family':'Times New Roman','size':16})
            plt.ylabel('TPR', fontdict={'family':'Times New Roman','size':16})
            plt.plot(fpr, tpr, color='#ff7f0e')
            plt.grid()
            plt.show()
else:
    print(f'Error: {saved_path} not found')
    while(True):
        choose = str(input('Would you like to continue UMAP? Please enter yes or y to UMAP, else no or n: '))
        if choose.lower() == 'yes' or choose.lower() == 'y':
            break
        elif choose.lower() == 'no' or choose.lower() == 'n':
            exit(0)
        else:
            print(f'Please Enter yes/y or no/n.')        
            

""" UMAP降维分析 原始数据 """
color = ['#75bbfd','#ff7f0e']
marker = ['o', 's']
matrix_name_feature = 'vat_concat_matrix.npy' # 只分析有该字段的matrix
if os.path.exists(matrix_path):
    all_files = [file for file in os.listdir(matrix_path) if file.endswith(matrix_name_feature)]
    for file in all_files:
        tag_name = file[:len(file) - len(matrix_name_feature)]
        print(f'Now is precessing {tag_name[:-1]}')

        ori_matrix = np.load(os.path.join(matrix_path, file), allow_pickle=True)
        # 控制下正负样本比例
        matrix = []
        neg_cnt, pos_cnt = 0,0
        threshold = 5000
        for x in ori_matrix:
            if neg_cnt >= threshold and pos_cnt >= threshold:
                break
            else:
                if neg_cnt < threshold and x[-1] == 0:
                    matrix.append(x)
                    neg_cnt += 1
                if pos_cnt < threshold and x[-1] == 1:
                    matrix.append(x)
                    pos_cnt += 1

        encoding, label = [],[]
        for line,_ in zip(matrix, tqdm(range(len(matrix)))):
            encoding.append(line[:-1])
            label.append(int(line[-1]))
        encoding = MinMaxScaler().fit_transform(encoding)
        assert len(encoding) == len(label) == len(matrix)
        print(f'Matrix Loaded, Start to UMAP to 2D...')

        # 二维
        start = time.time()
        mapper = umap.UMAP(n_neighbors=len(label)-1, n_components=2, random_state=12).fit(encoding)
        X_umap_2d = mapper.embedding_
        assert len(encoding) == len(X_umap_2d)
        end = time.time()
        print(f'It took {(end-start)/60} minutes to UMAP to 2D')

        X_umap_2d = MinMaxScaler().fit_transform(X_umap_2d)
        for (index, value), _ in zip(enumerate(X_umap_2d), tqdm(range(len(X_umap_2d)))):
            plt.scatter(value[0], value[1], color=color[label[index]], marker=marker[label[index]])
        # plt.title('Feature Matrix UMAP 2D', fontdict={'family':'Times New Roman','size':20})
        plt.title('Feature Matrix UMAP 2D', fontdict={'size':16})
        # plt.show()
        plt.savefig(os.path.join('.', 'assert', tag_name+'UMAP_2D.svg'), dpi=300, format='svg')
        plt.close()
        
        # 三维
        start = time.time()
        print(f'Start to UMAP to 3D')
        mapper = umap.UMAP(n_neighbors=len(label)-1, n_components=3, random_state=12).fit(encoding)
        X_umap_3d = mapper.embedding_
        assert len(encoding) == len(X_umap_3d)
        end = time.time()
        print(f'It took {(end-start)/60} minutes to UMAP to 3D')

        X_umap_3d = MinMaxScaler().fit_transform(X_umap_3d)
        ax = plt.subplot(projection = '3d')  
        for (index, value), _ in zip(enumerate(X_umap_3d), tqdm(range(len(X_umap_3d)))):
            ax.scatter3D(value[0], value[1], value[2], color=color[label[index]], marker=marker[label[index]])
        # plt.title('Feature Matrix UMAP 3D', fontdict={'family':'Times New Roman','size':20})
        plt.title('Feature Matrix UMAP 3D', fontdict={'size':16})
        plt.savefig(os.path.join('.', 'assert', tag_name+'UMAP_3D.svg'), dpi=300, format='svg')
        plt.close()
        # plt.show()

else:
    print(f'Error: {matrix_path} not found')
    exit(1)

""" UMAP降维分析 中间数据"""
color = ['#75bbfd','#ff7f0e']
marker = ['o', 's']
if os.path.exists(os.path.join('..', 'temp')):
    all_files = [file for file in os.listdir(os.path.join('..', 'temp')) if file.endswith('_mlp_hidden_layer.txt')]
    for file in all_files:
        tag_name = file[:len(file) - len('_mlp_hidden_layer.txt')+1]
        print(f'Now is precessing {tag_name[:-1]}')
        content_x, content_y = [],[]
        with open(os.path.join('..', 'temp', file), 'r') as f:
            for line in f:
                temp = []
                line = line.split(' ')
                for x in line[:-1]:
                    temp.append(float(x))
                y = line[-1]
                content_y.append(int(float(y)))
                content_x.append(temp)
        # content_x = MinMaxScaler().fit_transform(content_x).tolist()
        assert len(content_x) == len(content_y)
        # WeChat中2172个正样本 TikTok中4374个正样本
        pos_cnt = len([y for y in content_y if y == 1])

        encoding, label = [], []

        cnt = 0
        for x, y in zip(content_x, content_y):
            if y == 1:
                encoding.append(x)
                label.append(y)
            elif y == 0:
                if cnt < pos_cnt:
                    encoding.append(x)
                    label.append(y)
                    cnt += 1
                else:
                    continue
        assert len(encoding) == len(label)
        
        # 二维
        start = time.time()
        # mapper = umap.UMAP(n_neighbors=len(label)-1, n_components=2, random_state=2).fit(encoding)
        mapper = umap.UMAP(n_neighbors=50, n_components=2, random_state=2).fit(encoding)
        X_umap_2d = mapper.embedding_
        assert len(encoding) == len(X_umap_2d)
        end = time.time()
        print(f'It took {(end-start)/60} minutes to UMAP to 2D')

        X_umap_2d = MinMaxScaler().fit_transform(X_umap_2d)
        for (index, value), _ in zip(enumerate(X_umap_2d), tqdm(range(len(X_umap_2d)))):
            plt.scatter(value[0], value[1], color=color[label[index]], marker=marker[label[index]])
        # plt.title('Feature Matrix UMAP 2D', fontdict={'family':'Times New Roman','size':20})
        plt.title('Feature Matrix UMAP 2D', fontdict={'size':16})
        # plt.show()
        plt.savefig(os.path.join('.', 'assert', tag_name+'_hidden_UMAP_2D.svg'), dpi=300, format='svg')
        plt.close()

        # 三维
        start = time.time()
        print(f'Start to UMAP to 3D')
        # TODO 关注neighbors和random_state
        # mapper = umap.UMAP(n_neighbors=len(label)-1, n_components=3, random_state=2).fit(encoding)
        mapper = umap.UMAP(n_neighbors=50, n_components=3, random_state=2).fit(encoding)
        X_umap_3d = mapper.embedding_
        assert len(encoding) == len(X_umap_3d)
        end = time.time()
        print(f'It took {(end-start)/60} minutes to UMAP to 3D')

        X_umap_3d = MinMaxScaler().fit_transform(X_umap_3d)
        ax = plt.subplot(projection = '3d')  
        for (index, value), _ in zip(enumerate(X_umap_3d), tqdm(range(len(X_umap_3d)))):
            ax.scatter3D(value[0], value[1], value[2], color=color[label[index]], marker=marker[label[index]])
        # plt.title('Feature Matrix UMAP 3D', fontdict={'family':'Times New Roman','size':20})
        plt.title('Feature Matrix UMAP 3D', fontdict={'size':16})
        # plt.show()
        plt.savefig(os.path.join('.', 'assert', tag_name+'_hidden_UMAP_3D.svg'), dpi=300, format='svg')
        plt.close()
        exit(0)
    
else:
    print(f'Error: Temp does not exist')
    exit(1)
