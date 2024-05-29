# README

✨ The Code Of the Paper: Multi-modal information augmented model for micro-video recommendation(多模态信息增强的短视频推荐模型)✨

Cite as:

``` text
Yufu HUO,Beihong JIN,Zhaoyi LIAO. Multi-modal information augmented model for micro-video recommendation. Journal of ZheJiang University (Engineering Science), 2024, 58(6): 1142-1152.
```

## 1. Computational Device
**Experiment Environment:** 
|Device|Information|
|:-:|:-:|
|GPU|NVIDIA GeForce RTX 3090; Driver Version: 470.161.03   CUDA Version: 11.4|
|CPU|Intel(R) Xeon(R) Silver 4215R CPU @ 3.20GHz|
|OS|linux-Precision-7920-Tower 5.4.0-146-generic #163~18.04.1-Ubuntu SMP Mon Mar 20 15:02:59 UTC 2023 x86_64 x86_64 x86_64 GNU/Linux|

## 2. MMa4CTR and Baselines
**Dataset Tree:** 
```shell
// For details about WeChat and TikTok dataset, Go to 3. Dataset Description
dataset // Attention: You should put this folder at the same path as MMa4CTR.
├── TikTok2019ICME
│   └── track1
│       ├── final_track1_train.txt
│       ├── track1_audio_features_part1.txt
│       ├── track1_audio_features_part2.txt
│       ├── track1_audio_features_part3.txt
│       ├── track1_audio_features_part4.txt
│       ├── track1_title.txt
│       ├── track1_video_features_part10.txt
│       ├── track1_video_features_part11.txt
│       ├── track1_video_features_part1.txt
│       ├── track1_video_features_part2.txt
│       ├── track1_video_features_part3.txt
│       ├── track1_video_features_part4.txt
│       ├── track1_video_features_part5.txt
│       ├── track1_video_features_part6.txt
│       ├── track1_video_features_part7.txt
│       ├── track1_video_features_part8.txt
│       └── track1_video_features_part9.txt
└── wechat
    ├── feed_embeddings.csv
    ├── feed_info.csv
    └── user_action.csv
```

**Project Tree**
```shell
./dataset 
├── TikTok2019ICME
└── wechat

./MMa4CTR  
├── adjust_hyperparameter.py
├── assert 
│   └── *.svg
├── baseline // Baseline models
│   ├── bert4rec.yaml
│   ├── comp.py
│   ├── dataset
│   │   ├── README.txt
│   │   ├── tiktok
│   │   │   └── tiktok.inter
│   │   └── wechat
│   │       └── wechat.inter
│   ├── gcsan.yaml
│   ├── tiktok.yaml
│   └── wechat.yaml
├── clear_git.sh
├── cross_matrix.py
├── gen_matrix.py
├── get_user_embedding.py
├── load_path.py
├── main.go
├── main_linux
├── main_windows.exe
├── mlp.py
├── path.json
├── preprocess_inter.py
├── preprocess_multimodal.py
├── README.md
├── SimHei.ttf
├── statis_and_draw.py
├── utils.py
└── viewer.py
```

**Install Requirements for MMa4CTR:** 
1. Prepare your conda environment and then activate it: `conda create -n MMa4CTR python=3.9.5`$\rightarrow$`conda activate MMa4CTR`. <br>`python version = 3.9.5`, `golang version = go1.10.4 linux/amd64`.
2. Git clone MMa4CTR project: `git clone https://gitee.com/FluorineHow/MMa4CTR.git`$\rightarrow$`cd MMa4CTR`
3. Requirements list
```shell
conda install numpy=1.24.2
conda install rich=12.5.1
conda install tqdm=4.64.1
conda install orjson=3.8.5
conda install keras-preprocessing=1.1.2
pip install torch torchvision torchaudio // I failed install it with conda 
conda install scikit-learn=1.2.0 
conda install matplotlib=3.7.1
conda install umap-learn=0.5.3
pip install umap-learn
```
To check torch-cpu or torch-gpu, run python as follow:
```python
>>> import torch
>>> torch.cuda.is_available()
True
# if print True, it is torch-gpu
```

**Install Requirements for Baselines:** 
[Install RecBole](https://recbole.io/docs/get_started/install.html). My python = 3.7.12, recbole = 1.1.1, ray = 2.3.0

**Run baselines code:** `cd ./MMa4CTR/baseline` -> change `wechat.yaml\tiktok.yaml` and `comp.py` -> `python comp.py`

**Run MMa4CTR code:**<br>
Raw Data Preprocess: `python preprocess_inter.py` -> You could run `./main_windows.exe` on Windows Operating System or run `main_linux` on Linux(x86_64); if it does not work, please run `go build -o main main.go` and then `./main` on your own machine -> `python preprocess_inter.py` again -> `python preprocess_multimodal.py`<br>
Generate embeddings of each user: `python get_user_embedding.py`<br>
Generate matrix for MLP: `python cross_matrix.py` and `python gen_matrix.py`. cross_matrix.py: modal cross and modal concat, gen_matrix.py: single modal.<br>
Start train/valid/test: `python adjust_hyperparameter.py`<br>
Draw pictures:`python statis_and_draw.py` and `python viewer.py`

## 3. Dataset Description
### 3.1. WeChat
[2021 中国高校计算机大赛——微信大数据挑战赛: 赛题描述——微信视频号推荐算法](https://algo.weixin.qq.com/2021/problem-description)
#### Feed信息表
该数据包含了视频(简称为feed)的基本信息和文本、音频、视频等多模态特征. 具体字段如下: 
|字段名	|类型	|说明|	备注|
|:-:|:-:|:-:|:-:|
|feedid|	String|	Feed视频ID|	已脱敏|
|authorid|	String|	视频号作者ID|	已脱敏|
|videoplayseconds|	Int|	Feed时长|	单位"秒|
|description|	String|	Feed配文,以词为单位使用空格分隔|	已脱敏;存在空值|
|ocr|	String|	图像识别信息,以词为单位使用空格分隔|	已脱敏;存在空值|
|asr|	String|	语音识别信息,以词为单位使用空格分隔|	已脱敏;存在空值|
|description_char|	String|	Feed配文,以字为单位使用空格分隔|	已脱敏;存在空值|
|ocr_char|	String|	图像识别信息,以字为单位使用空格分隔|	已脱敏;存在空值|
|asr_char|	String|	语音识别信息,以字为单位使用空格分隔|	已脱敏;存在空值|
|bgm_song_id|	Int|	背景音乐ID|	已脱敏;存在空值|
|bgm_singer_id|	Int|	背景音乐歌手ID|	已脱敏;存在空值|
|manual_keyword_list|	String|	人工标注的关键词,多个关键词使用英文分号";"分隔 |	已脱敏;存在空值|
|machine_keyword_list|	String|	机器标注的关键词,多个关键词使用英文分号";"分隔| 	已脱敏;存在空值|
|manual_tag_list|	String|	人工标注的分类标签,多个标签使用英文分号";"分隔 |	已脱敏;存在空值|
|machine_tag_list|	String|	机器标注的分类标签,多个标签使用英文分号";"分隔 |	已脱敏;存在空值|
|feed_embedding|	String|	融合了ocr、asr、图像、文字的多模态的内容理解特征向量| 	512维向量|

**说明:**<br>
$\circ$训练集和测试集涉及的feed均在此表中<br>
$\circ$description, orc, asr三个字段为原始文本数据以词为单位使用空格分隔和脱敏处理后得到的.例如"文本"我参加了中国高校计算机大赛"经过处理后得到类似"2 32 100 25 12 89 27"的形式(此处只是一个样例,不代表实际脱敏结果).此外,我们还提供了以字为单位使用空格分隔和脱敏的结果,对应的字段分别为description_char、ocr_char、asr_char<br>
$\circ$machine_tag_list字段比manual_tag_list字段增加了每个标签对应的预测概率值(取值区间[0,1]).脱敏后的标签和概率值之间用空格分隔.例如""1025 0.32657512;2034 0.87653981;35 0.47265462"<br>
$\circ$manual_keyword_list和machine_keyword_list共享相同的脱敏映射表.如果原先两个字段都包含同个关键词,那么脱敏后两个字段都会包含同个id<br>
$\circ$manual_tag_list和machine_tag_list共享相同的脱敏映射表.如果原先两个字段都包含同个分类标签,那么脱敏后两个字段都会包含同个id<br>
$\circ$feed_embedding字段为String格式,包含512维,数值之间用空格分隔<br>

#### 用户行为表
该数据包含了用户在视频号内一段时间内的历史行为数据(包括停留时长、播放时长和各项互动数据).具体字段如下:
|字段名|	类型|	说明|	备注|
|:-:|:-:|:-:|:-:|
|userid|	String|	用户ID|	已脱敏|
|feedid|	String|	Feed视频ID|	已脱敏|
|device|	Int|	设备类型ID|	已脱敏|
|date_|	Int|	日期|	已脱敏为1-n,n代表第n天|
|play|	Int|	视频播放时长|	单位:毫秒;若播放时长大于视频|时长,则属于重播的情况|
|stay|	Int|	用户停留时长|	单位:毫秒|
|read_comment|Bool|是否查看评论|取值{0, 1},0代表"否",1代表"是" |
|like|Bool|是否点赞|取值{0, 1},0代表"否",1代表"是" |
|click_avatar|Bool|是否点击头像|取值{0, 1},0代表"否",1代表"是" |
|favorite|Bool|是否收藏|取值{0, 1},0代表"否",1代表"是" |
|forward|Bool|是否转发|取值{0, 1},0代表"否",1代表"是" |
|comment|Bool|是否发表评论|取值{0, 1},0代表"否",1代表"是" |
|follow|Bool|是否关注|取值{0, 1},0代表"否",1代表"是" |

**七个点击行为的权重如下:**
|字段名|	字段说明|	权重|
|:-:|:-:|:-:|
|read_comment|	是否查看评论|	4|
|like|	是否点赞|	3|
|click_avatar|	是否点击头像|	2|
|forward|	是否转发|	1|
|favorite|	是否收藏|	1|
|comment|	是否发表评论|	1|
|follow|	是否关注|	1|

**说明:**<br>
$\circ$用户行为表中每个用户对应的数据已按照时间戳顺序由小到大排列,数据中不提供时间戳字段.<br>

### 3.2. 抖音数据集(track2 小规模数据赛道)
**final_track2_train.txt:**
|字段|字段描述|数据类型|备注|
|:-:|:-:|:-:|:-:|
|uid|用户id|int|已脱敏|
|user_city|用户所在城市|int|已脱敏|
|item_id|作品id|int|已脱敏|
|author_id|作者id|int|已脱敏|
|item_city|作品城市|int|已脱敏|
|channel|观看到作品的来源|int|已脱敏|
|finish|是否浏览完作品|bool|已脱敏|
|like|是否对作品点赞|bool|已脱敏|
|music_id|音乐id|int|已脱敏|
|device|设备id|int|已脱敏|
|time|作品发布时间|int|已脱敏|
|duration_time|作品时长|int|单位:秒|

**track2_title.txt\track2_audio_features.txt\track2_video_features.txt:**<br>
$\circ$User Interaction Behavior: click、like、follow<br>
$\circ$Video: Face Features(Face Score、Expression、Gender); Video Content Features(video embedding)<br>
$\circ$NLP: Title Features(word embedding)<br>
$\circ$Audio: BGM Feature

### 3.3. 抖音数据集(track1 大规模数据赛道)
**final_track1_train.txt:**
|字段|字段描述|数据类型|备注|
|:-:|:-:|:-:|:-:|
|uid|用户id|int|已脱敏|
|user_city|用户所在城市|int|已脱敏|
|item_id|作品id|int|已脱敏|
|author_id|作者id|int|已脱敏|
|item_city|作品城市|int|已脱敏|
|channel|观看到作品的来源|int|已脱敏|
|finish|是否浏览完作品|bool|已脱敏|
|like|是否对作品点赞|bool|已脱敏|
|music_id|音乐id|int|已脱敏|
|device|设备id|int|已脱敏|
|time|作品观看的起始时间|int|已脱敏|
|duration_time|作品时长|int|单位:秒|

**visual feature:**
track1_video_features_part1.txt ~ track1_video_features_part11.txt<br>
**acoustic feature:**
track1_audio_features_part1.txt ~ track1_audio_features_part4.txt<br>
**textual feature:**
track1_title.txt<br>

## 4. Hyperparameter Sensitivity Study
### 4.1. WeChat
matrix : vat_concat ; user embedding length(each modal)=22
```shell
MLP(
   (activation_function): Tanh()
   (predict): Sigmoid()
   (_features): Sequential(
     (0): Linear(in_features=194, out_features=128, bias=True)
     (1): Tanh()
     (2): Linear(in_features=128, out_features=32, bias=True)
     (3): Tanh()
     (4): Linear(in_features=32, out_features=4, bias=True)
     (5): Tanh()
   )
   (_classifier): Linear(in_features=4, out_features=1, bias=True)
)
```
#### 4.1.1. epochs
batch size = 8192, learning rate = 0.001
|epoch|AUC|epoch|AUC|
|:-:|:-:|:-:|:-:|
|5|0.945460024418023|10|0.9517885227306362|
|15|0.940094171008244|20|0.9332946151885952|
|25|0.9303479920771285|30|0.9302333722455105|
|35|0.9454909968123039|40|0.931314721874389|
|45|0.9436334450978896|50|0.9191246774372999|
|55|0.9043831338203693|60|0.9179239438039649|
|65|0.950838422077388|70|0.9412803176153527|
|75|0.9510260709268231|80|0.9179849675147189|
|85|0.9105302167006715|90|0.9165578019242125|
|95|0.9165578019242125|100|0.9099362888635572|

#### 4.1.2. learning rate
batch size = 8192
|learning rate|epoch = 10|epoch = 30|
|:-:|:-:|:-:|
|0.01|0.50|0.7433700177013759|
|0.005|0.5003212664049789|0.50|
|0.002|0.9178128989138242|0.9329537409609425|
|0.001|0.952164271415028|0.9406701855950172|
|0.0005|0.9479440665078159|0.9435869902948786|
|0.0002|0.9470969901862731|0.9516189673280946|
|0.0001|0.9427243119826154|0.9485197599032935|
|0.00005|0.9473557290463805|0.9453024031663192|
|0.00002|0.9494794050441399|0.9426944178028069|
|0.00001|0.9443388322442728|0.9489048386183262|

#### 4.1.3. batch size
learning rate = 0.001
|batch size |epoch = 10|epoch = 30|
|:-:|:-:|:-:|
|32768|0.9455507298281898|0.9521014180668835|
|16384|0.9489413147489928|0.9526058415845631|
|8192|0.9421495409826534|0.9204879061943578|
|4096|0.9356626120162714|0.9380819045762812|
|2048|0.9028231356987881|0.921970903067719|
|1024|0.8968900996392982|0.9094429884475153|
|512|0.8677431554029484|0.9166537300695596|
|256|0.8976556740813587|0.900815323448885|
|128|0.8892057843470222|0.6139191381470472|
|64|0.7942007641962512|0.7480759458748958|

### 4.2. TikTok like
matrix : vat_concat ; user embedding length(each modal)=22
```shell
MLP(
  (activation_function): Tanh()
  (_features): Sequential(
    (0): Linear(in_features=169, out_features=128, bias=True)
    (1): Tanh()
    (2): Linear(in_features=128, out_features=32, bias=True)
    (3): Tanh()
    (4): Linear(in_features=32, out_features=4, bias=True)
    (5): Tanh()
  )
  (_classifier): Linear(in_features=4, out_features=1, bias=True)
  (predict): Sigmoid()
)
```
#### 4.2.1. epochs
batch size = 8192, learning rate = 0.001
|epoch|AUC|epoch|AUC|
|:-:|:-:|:-:|:-:|
|5|0.8919237827653452|10|0.8924156054410297|
|15|0.8898916309110074|20|0.8887982075041272|
|25|0.8811722433496609|30|0.8878198213523519|
|35|0.8624564145928819|40|0.8839217041282715|
|45|0.892258330682395|50|0.8811387045093966|
|55|0.8910089502177834|60|0.8875505918175111|
|65|0.8852166390732102|70|0.8843582388074401|
|75|0.8707360641209556|80|0.8727951748528887|
|85|0.8785564537866188|90|0.8652849773950988|
|95|0.8631421800847775|100|0.8732818181066146|

#### 4.2.2. learning rate
batch size=8192
|learning rate|epoch = 10|epoch = 30|
|:-:|:-:|:-:|
|0.01|0.7025513378399374|0.5|
|0.005|0.851888848528819|0.5|
|0.002|0.8938259264063324|0.891967135060601|
|0.001|0.8899793012138245|0.8843747178047459|
|0.0005|0.8871947126500858|0.8888243789200523|
|0.0002|0.8898606120770207|0.8867022735108548|
|0.0001|0.8909920911888324|0.8908740415624474|
|0.00005|0.8893171449058381|0.8924454166511333|
|0.00002|0.8902644893379209|0.8892847165057919|
|0.00001|0.892661124901457|0.889038190880157|

#### 4.2.3. batch size
learning rate = 0.001
|batch size|epoch = 10|epoch = 30|
|:-:|:-:|:-:|
|32768|0.8937748569544772|0.8856265634788756|
|16384|0.8892840004410765|0.8809282957504021|
|8192|0.8883746059884121|0.8885827916106332|
|4096|0.888349772419258|0.8889139816144829|
|2048|0.8770224056059608|0.890411599619969|
|1024|0.8766193833356783|0.8725818720191482|
|512|0.8759438430803062|0.8383831920604533|
|256|0.8549247883787177|0.8789210306503084|
|128|0.871308732163039|0.6802240806085553|
|64|0.6643586901651264|0.6130949017628576|


## 5. Ablation Study
### 5.1. WeChat
batch size=8192, learning rate = 0.001 & auto-adjustable, user embedding length(each modal) = 22
#### 5.1.1. Two modal Concat
|concat|epoch = 10|epoch = 30|
|:-:|:-:|:-:|
|va_concat: 22\*2+128=172|0.9448939390895489|0.9497949740096747|
|vt_concat: 22\*2+128=172|0.9520634193247621|0.9517589586366516|
|at_concat: 22\*2+128=172|0.9513680655356005|0.9469269811627941|
#### 5.1.2. Two modal Cross
|cross|epoch = 10|epoch = 30|
|:-:|:-:|:-:|
|va_cross: 22\*2+128=172|0.9459119105931197|0.9439084731333766|
|vt_cross: 22\*2+73=117|0.951661220290105|0.9503867671600241|
|at_cross: 22\*2+73=117|0.9516077085025452|0.9525714003242896|
#### 5.1.3. Three modal Cross
|cross|epoch = 10|epoch = 30|
|:-:|:-:|:-:|
|vat_cross: 22\*3+73=139|0.951278614912762|0.9499676444736435|
#### 5.1.4. Single modal
|Modal|epoch = 10|epoch = 30|
|:-:|:-:|:-:|
|visual|0.9472118047092303|0.9466316301186815|
|acoustic|0.9442278562043687|0.9429589058223927|
|textual|0.8862324331783548|0.8935200933624496|
#### 5.1.5. Lengths of user multi-modal Embedding
|user-modal-embedding length|in_feature|epoch = 10|epoch = 30|
|:-:|:-:|:-:|:-:|
|0|128+1|0.6162375743342403|0.621318420047907|
|5|128+6\*3=146|0.9499813423764779|0.9512949604201534|
|11|128+12\*3=164|0.9511351517733696|0.9418210202663709|
|15|128+16\*3=176|0.9453357365636931|0.9484451180110319|
|21|128+22\*3=194|0.9527299209116211|0.940642049566947|
|30|128+31\*3=221|0.9478221001253427|0.9456434582493786|
#### 5.1.6. 验证中断与学习率自动半衰
|Dataset|with|without|
|:-:|:-:|:-:|
|WeChat|0.9483886193091363|0.9016327111530487|

### 5.2. TikTok like
batch size=8192, learning rate = 0.001 & auto-adjustable, user embedding length(each modal) = 22
#### 5.2.1. Two modal Concat
|concat|epoch = 10|epoch = 30|
|:-:|:-:|:-:|
|va_concat: 22\*2+81=125|0.8886791505843656|0.8880425237643812|
|vt_concat: 22\*2+93=137|0.888935155404758|0.8880141917444553|
|at_concat: 22\*2+32=76|0.888637093104308|0.8907652737013179|
#### 5.2.2. Two modal Cross
|cross|epoch = 10|epoch = 30|
|:-:|:-:|:-:|
|va_cross: 22\*2+10=54|0.8915278258925563|0.8921931320472225|
|vt_cross: 22\*2+22=66|0.889746197409806|0.8912906610039774|
|at_cross: 22\*2+10=54|0.893817758304634|0.8932477960943104|
#### 5.2.3. Three modal Cross
|cross|epoch = 10|epoch = 30|
|:-:|:-:|:-:|
|vat_cross: 22\*3+10=76|0.8911525498003717|0.8942150533465278|
#### 5.2.4. Single modal
|Modal|epoch = 10|epoch = 30|
|:-:|:-:|:-:|
|visual|0.8821479378078083|0.8873140938688661|
|acoustic|0.8868338550087214|0.8873125659659982|
|textual|0.8942832033403586|0.8933923393578772|
#### 5.2.5. Lengths of user multi-modal Embedding
|user-modal-embedding length|in_feature|epoch = 10|epoch = 30|
|:-:|:-:|:-:|:-:|
|0|103+1=104|0.5761963427453821|0.5783002540186957|
|5|103+6\*3=121|0.8910880922107908|0.8902739912572171|
|11|103+12\*3=139|0.8895959543647738|0.8884687634217409|
|15|103+16\*3=151|0.8900106870249341|0.8880963353105578|
|21|103+22\*3=169|0.8895607898335833|0.8910121877392787|
|30|103+31\*3=196|0.8891491617321927|0.8875773363430756|
#### 5.2.6. 验证中断与学习率自动半衰
|Dataset|with|without|
|:-:|:-:|:-:|
|TikTok|0.888649536964931|0.8769545378850909|

## 6. Performance Comparison Experiments
### 6.1. WeChat
Dataset: WeChat Channel
Train Set : Valid Set : Test set = 8 : 1 : 1
batchsize: {all model : 8192} 
Indicator of Rec performance: AUC
MMa4CTR: initial learning rate = 0.001, learning rate reduces by half every 5 epochs.

All baseline models are from [RecBole](https://recbole.io/docs/index.html).

**Recommendation performance comparison experiments:**
BERT4Rec's batch size = 1024; others' batch size = 8192;
@n: epoch = n
|Model|AUC@10|AUC@20|AUC@30|AUC@50|LogLoss@10|LogLoss@20|LogLoss@30|LogLoss@50|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|BPR     |0.7365|0.7713|0.7763|0.7755|18.0170|17.4122|17.2362|16.9897|
|FPMC    |0.8174|0.8438|0.8473|0.8467|17.1369|17.1292|17.1779|17.1611|
|NGCF    |0.7196|0.7648|0.7754|0.7816|15.8703|16.7471|17.0796|17.2763|
|LightGCN|0.7113|0.7615|0.7729|0.7817|21.7398|20.5728|20.366 |20.4267|
|BERT4Rec|0.9022|0.9022|0.9023|0.9022|16.3755|16.2804|16.2741|16.2775|
|GCSAN   |0.8991|0.9087|0.9079|      |15.6458|16.6468|16.8030|       |
|DIN     |0.9024|0.9020|0.9029|0.9031| 0.6474| 0.6171| 0.6714| 0.5927|
|DIEN    |0.9013|0.9014|0.8992|0.9000| 0.6567| 0.6564| 0.5939| 0.5947|
|MMa4CTR |0.9475831601555518|0.9517740318970852|0.9523486006288877|0.9513642267231205|1.393633134988851|1.4321310928158022|1.4718578852987794|1.4773581285335016|

|Epoch|BPR|FPMC|NGCF|LightGCN|BERT4Rec|GCSAN|DIN|DIEN|MMa4CTR|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|10|0.5113|0.5204|0.6112|0.5068|0.7478|0.7445|0.8324|0.7972| 0.9527299209116211|
|20|0.5981|0.6338|0.6159|0.5663|0.7566|0.7508|0.8436|0.7987|0.9421524906717417|
|30|0.6386|0.6761|0.6310|0.6110|0.7716|0.7534|0.8498|0.8028|0.940642049566947|
|40|0.6607|0.6943|0.6429|0.6489|0.7714|0.7823|0.8498|0.8027|0.942112964706188|
|50|0.6716|0.7038|0.6612|0.6744|0.7714|0.7834|0.8494|0.8027|0.9468996222529457|
|60|0.6775|0.7078|0.6722|0.6878|0.7714|0.7891|0.8525|0.8028|0.9498167296961924|
|70|0.6812|0.7107|0.6843|0.6961|0.7715|0.7891|0.8523|0.8026|0.9436186037540427|
|80|0.6818|0.7103|0.6891|0.6999|0.7714|0.7890|0.8523|0.8027|0.9527255952060796|
|90|0.6818|0.7106|0.6948|0.7024|0.7715|0.7892|0.8509|0.8029|0.9507852430110678|
|100|0.6815|0.7105|0.6959|0.7040|0.7713|0.7903|0.8521|0.8027|0.9446970198357546|

**Computational performance comparison experiments:**
|Model|Train Time(s)|Test Time(s)|Number of trainable parameters|
|:-:|:-:|:-:|:-:|
|BPR|54.77|45|7,460,224|
|FPMC|62.43|48|19,820,544|
|NGCF|126.54|109|7,485,184|
|LightGCN|91.54|79|7,460,224|
|BERT4Rec|471.41|317|6,915,904|
|GCSAN|4756.09|4385|6,908,032|
|DIN|170.44|108|1,449,310|
|DIEN|643.73|243|1,587,847|
|MMa4CTR|180.78|17|29,225|

**Baseline Frameworks:**
```shell
########## BPR ##########
INFO  BPR(
  (user_embedding): Embedding(20001, 64)
  (item_embedding): Embedding(96565, 64)
  (loss): BPRLoss()
)

########## FPMC ##########
INFO  FPMC(
  (UI_emb): Embedding(20001, 64)
  (IU_emb): Embedding(96565, 64)
  (LI_emb): Embedding(96565, 64, padding_idx=0)
  (IL_emb): Embedding(96565, 64)
  (loss_fct): BPRLoss()
)

########## NGCF ##########
INFO  NGCF(
  (sparse_dropout): SparseDropout()
  (user_embedding): Embedding(20001, 64)
  (item_embedding): Embedding(96565, 64)
  (GNNlayers): ModuleList(
    (0): BiGNNLayer(
      (linear): Linear(in_features=64, out_features=64, bias=True)
      (interActTransform): Linear(in_features=64, out_features=64, bias=True)
    )
    (1): BiGNNLayer(
      (linear): Linear(in_features=64, out_features=64, bias=True)
      (interActTransform): Linear(in_features=64, out_features=64, bias=True)
    )
    (2): BiGNNLayer(
      (linear): Linear(in_features=64, out_features=64, bias=True)
      (interActTransform): Linear(in_features=64, out_features=64, bias=True)
    )
  )
  (mf_loss): BPRLoss()
  (reg_loss): EmbLoss()
)

########## LightGCN ##########
INFO  LightGCN(
  (user_embedding): Embedding(20001, 64)
  (item_embedding): Embedding(96565, 64)
  (mf_loss): BPRLoss()
  (reg_loss): EmbLoss()
)

########## BERT4Rec ##########
INFO  BERT4Rec(
  (item_embedding): Embedding(106446, 64, padding_idx=0)
  (position_embedding): Embedding(51, 64)
  (trm_encoder): TransformerEncoder(
    (layer): ModuleList(
      (0): TransformerLayer(
        (multi_head_attention): MultiHeadAttention(
          (query): Linear(in_features=64, out_features=64, bias=True)
          (key): Linear(in_features=64, out_features=64, bias=True)
          (value): Linear(in_features=64, out_features=64, bias=True)
          (softmax): Softmax(dim=-1)
          (attn_dropout): Dropout(p=0.2, inplace=False)
          (dense): Linear(in_features=64, out_features=64, bias=True)
          (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
          (out_dropout): Dropout(p=0.2, inplace=False)
        )
        (feed_forward): FeedForward(
          (dense_1): Linear(in_features=64, out_features=256, bias=True)
          (dense_2): Linear(in_features=256, out_features=64, bias=True)
          (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.2, inplace=False)
        )
      )
      (1): TransformerLayer(
        (multi_head_attention): MultiHeadAttention(
          (query): Linear(in_features=64, out_features=64, bias=True)
          (key): Linear(in_features=64, out_features=64, bias=True)
          (value): Linear(in_features=64, out_features=64, bias=True)
          (softmax): Softmax(dim=-1)
          (attn_dropout): Dropout(p=0.2, inplace=False)
          (dense): Linear(in_features=64, out_features=64, bias=True)
          (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
          (out_dropout): Dropout(p=0.2, inplace=False)
        )
        (feed_forward): FeedForward(
          (dense_1): Linear(in_features=64, out_features=256, bias=True)
          (dense_2): Linear(in_features=256, out_features=64, bias=True)
          (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.2, inplace=False)
        )
      )
    )
  )
  (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
  (dropout): Dropout(p=0.2, inplace=False)
)

########## GCSAN ##########
INFO  GCSAN(
  (item_embedding): Embedding(106445, 64, padding_idx=0)
  (gnn): GNN(
    (linear_edge_in): Linear(in_features=64, out_features=64, bias=True)
    (linear_edge_out): Linear(in_features=64, out_features=64, bias=True)
  )
  (self_attention): TransformerEncoder(
    (layer): ModuleList(
      (0): TransformerLayer(
        (multi_head_attention): MultiHeadAttention(
          (query): Linear(in_features=64, out_features=64, bias=True)
          (key): Linear(in_features=64, out_features=64, bias=True)
          (value): Linear(in_features=64, out_features=64, bias=True)
          (softmax): Softmax(dim=-1)
          (attn_dropout): Dropout(p=0.2, inplace=False)
          (dense): Linear(in_features=64, out_features=64, bias=True)
          (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
          (out_dropout): Dropout(p=0.2, inplace=False)
        )
        (feed_forward): FeedForward(
          (dense_1): Linear(in_features=64, out_features=256, bias=True)
          (dense_2): Linear(in_features=256, out_features=64, bias=True)
          (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.2, inplace=False)
        )
      )
    )
  )
  (reg_loss): EmbLoss()
  (loss_fct): CrossEntropyLoss()
)

########## DIN ##########
INFO  DIN(
  (attention): SequenceAttLayer(
    (att_mlp_layers): MLPLayers(
      (mlp_layers): Sequential(
        (0): Dropout(p=0.0, inplace=False)
        (1): Linear(in_features=40, out_features=256, bias=True)
        (2): Sigmoid()
        (3): Dropout(p=0.0, inplace=False)
        (4): Linear(in_features=256, out_features=256, bias=True)
        (5): Sigmoid()
        (6): Dropout(p=0.0, inplace=False)
        (7): Linear(in_features=256, out_features=256, bias=True)
        (8): Sigmoid()
      )
    )
    (dense): Linear(in_features=256, out_features=1, bias=True)
  )
  (dnn_mlp_layers): MLPLayers(
    (mlp_layers): Sequential(
      (0): Dropout(p=0.0, inplace=False)
      (1): Linear(in_features=30, out_features=256, bias=True)
      (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): Dice(
        (sigmoid): Sigmoid()
      )
      (4): Dropout(p=0.0, inplace=False)
      (5): Linear(in_features=256, out_features=256, bias=True)
      (6): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (7): Dice(
        (sigmoid): Sigmoid()
      )
      (8): Dropout(p=0.0, inplace=False)
      (9): Linear(in_features=256, out_features=256, bias=True)
      (10): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (11): Dice(
        (sigmoid): Sigmoid()
      )
    )
  )
  (embedding_layer): ContextSeqEmbLayer(
    (token_embedding_table): ModuleDict(
      (user): FMEmbedding(
        (embedding): Embedding(20001, 10)
      )
      (item): FMEmbedding(
        (embedding): Embedding(96565, 10)
      )
    )
    (float_embedding_table): ModuleDict()
    (token_seq_embedding_table): ModuleDict(
      (user): ModuleList()
      (item): ModuleList()
    )
    (float_seq_embedding_table): ModuleDict(
      (user): ModuleList()
      (item): ModuleList()
    )
  )
  (dnn_predict_layers): Linear(in_features=256, out_features=1, bias=True)
  (sigmoid): Sigmoid()
  (loss): BCEWithLogitsLoss()
)

########## DIEN ##########
INFO  DIEN(
  (interset_extractor): InterestExtractorNetwork(
    (gru): GRU(10, 10, batch_first=True)
    (auxiliary_net): MLPLayers(
      (mlp_layers): Sequential(
        (0): Dropout(p=0.0, inplace=False)
        (1): Linear(in_features=20, out_features=256, bias=True)
        (2): Dropout(p=0.0, inplace=False)
        (3): Linear(in_features=256, out_features=256, bias=True)
        (4): Dropout(p=0.0, inplace=False)
        (5): Linear(in_features=256, out_features=256, bias=True)
        (6): Dropout(p=0.0, inplace=False)
        (7): Linear(in_features=256, out_features=1, bias=True)
      )
    )
  )
  (interest_evolution): InterestEvolvingLayer(
    (attention_layer): SequenceAttLayer(
      (att_mlp_layers): MLPLayers(
        (mlp_layers): Sequential(
          (0): Dropout(p=0.0, inplace=False)
          (1): Linear(in_features=40, out_features=256, bias=True)
          (2): Sigmoid()
          (3): Dropout(p=0.0, inplace=False)
          (4): Linear(in_features=256, out_features=256, bias=True)
          (5): Sigmoid()
          (6): Dropout(p=0.0, inplace=False)
          (7): Linear(in_features=256, out_features=256, bias=True)
          (8): Sigmoid()
        )
      )
      (dense): Linear(in_features=256, out_features=1, bias=True)
    )
    (dynamic_rnn): DynamicRNN(
      (rnn): AUGRUCell()
    )
  )
  (embedding_layer): ContextSeqEmbLayer(
    (token_embedding_table): ModuleDict(
      (user): FMEmbedding(
        (embedding): Embedding(20001, 10)
      )
      (item): FMEmbedding(
        (embedding): Embedding(96565, 10)
      )
    )
    (float_embedding_table): ModuleDict()
    (token_seq_embedding_table): ModuleDict(
      (user): ModuleList()
      (item): ModuleList()
    )
    (float_seq_embedding_table): ModuleDict(
      (user): ModuleList()
      (item): ModuleList()
    )
  )
  (dnn_mlp_layers): MLPLayers(
    (mlp_layers): Sequential(
      (0): Dropout(p=0.0, inplace=False)
      (1): Linear(in_features=30, out_features=256, bias=True)
      (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): Dice(
        (sigmoid): Sigmoid()
      )
      (4): Dropout(p=0.0, inplace=False)
      (5): Linear(in_features=256, out_features=256, bias=True)
      (6): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (7): Dice(
        (sigmoid): Sigmoid()
      )
      (8): Dropout(p=0.0, inplace=False)
      (9): Linear(in_features=256, out_features=256, bias=True)
      (10): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (11): Dice(
        (sigmoid): Sigmoid()
      )
    )
  )
  (dnn_predict_layer): Linear(in_features=256, out_features=1, bias=True)
  (sigmoid): Sigmoid()
  (loss): BCEWithLogitsLoss()
)

########## MMa4CTR ##########
MLP(
   (activation_function): Tanh()
   (predict): Sigmoid()
   (_features): Sequential(
     (0): Linear(in_features=194, out_features=128, bias=True)
     (1): Tanh()
     (2): Linear(in_features=128, out_features=32, bias=True)
     (3): Tanh()
     (4): Linear(in_features=32, out_features=4, bias=True)
     (5): Tanh()
   )
   (_classifier): Linear(in_features=4, out_features=1, bias=True)
)
```
### 6.2. TikTok like
Dataset: TikTok trace1
Train Set : Valid Set : Test set = 8 : 1 : 1
batch size: {BRP:8192, FPMC:8192, NGCF:16?, LightGCN:8192, BERT4Rec:56, GCSAN:512 ,DIN:8192, DIEN:8192, MMa4CTR:8192}
Indicator of Rec performance: AUC
MMa4CTR: initial learning rate = 0.001, learning rate reduces by half every 5 epochs.

All baseline models are from [RecBole](https://recbole.io/docs/index.html).

**Recommendation performance comparison experiments:**
Batch Size = {BRP : 8192, FPMC : 8192, NGCF : , LightGCN : 8192, DIN : 8192, DIEN : 8192, BERT4Rec : 32, }
@n: epoch = n
|Model|AUC@10|AUC@20|AUC@30|AUC@50|LogLoss@10|LogLoss@20|LogLoss@30|LogLoss@50|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|BPR     |0.3805|0.3805|0.3828|0.3897|18.3764|18.5995|18.2609|17.5754|
|FPMC    |0.4015|0.4070|0.4133|0.4273|17.7727|17.9348|17.6401|16.9038|
|NGCF    |||||||||
|LightGCN|0.3863|0.3862|0.3862|0.3862|25.1536|25.1566|25.1563|25.1572|
|BERT4Rec|0.4838||0.4838||13.4550|13.4543|||
|GCSAN   |||||||||
|DIN     |0.5087|0.5469|0.5480|0.5702|1.0317|0.8437|0.8796|0.7747|
|DIEN    |0.5788|0.5787|0.5787|0.5788|0.8218|0.8219|0.8218|0.8218|
|MMa4CTR |0.8948|0.8947|0.8899|0.8924|0.9510|1.2271|1.0760|1.2647|


dataset = tiktok+like
|Epoch|BPR|FPMC|NGCF|LightGCN|BERT4Rec|GCSAN|DIN|DIEN|MMa4CTR|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|10 |0.3805|0.4015||0.3863|0.5844|0.5869|0.5087|0.5788|0.8895607898335833|
|20 |0.3805|0.4070||0.3862|0.5834|0.4829|0.5469|0.5787|0.8902426666895447|
|30 |0.3828|0.4133||0.3862|0.6295|0.4550|0.5480|0.5742|0.8910121877392787|
|40 |0.3861|0.4203||0.3862|0.5877|0.4348|0.5763|0.5748|0.8900689533865059|
|50 |0.3897|0.4273||0.3862|      |0.4736|0.5619|0.5787|0.888445766027537|
|60 |0.3930|0.4341||0.3862|      |      |      |0.5775|0.8873859916991084|
|70 |0.3957|0.4414||0.3862|      |      |      |0.5370|0.8881535967962569|
|80 |0.3992|0.4486||0.3863|      |      |      |0.5370|0.8897038379029022|
|90 |0.4016|0.4558||0.3863|      |      |      |0.5374|0.8870444828207427|
|100|0.4044|0.4630||0.3863|      |      |      |0.5377|0.8915471290181558|

**Computational performance comparison experiments:**
|Model|Train Time(s)|Test Time(s)|Number of trainable parameters|
|:-:|:-:|:-:|:-:|
|BPR|619.19|601|195,060,416|
|FPMC|549.91|399|584,966,336|
|NGCF|||195,085,376|
|LightGCN|1929.59|1908|195,060,416|
|BERT4Rec|71639.71|5060|195,056,384|
|GCSAN|44006.24|37142|195,048,512|
|DIN|643.46|574|30,761,840|
|DIEN|2315.56|1464|30,900,377|
|MMa4CTR|163.8|23|26,025|

**Models' Frameworks:**
```shell
########## BPR ##########
BPR(
  (user_embedding): Embedding(1679, 64)
  (item_embedding): Embedding(3046140, 64)
  (loss): BPRLoss()
)

########## FPMC ##########
FPMC(
  (UI_emb): Embedding(1679, 64)
  (IU_emb): Embedding(3046140, 64)
  (LI_emb): Embedding(3046140, 64, padding_idx=0)
  (IL_emb): Embedding(3046140, 64)
  (loss_fct): BPRLoss()
)

########## NGCF ##########

########## LightGCN ##########
LightGCN(
  (user_embedding): Embedding(1679, 64)
  (item_embedding): Embedding(3046140, 64)
  (mf_loss): BPRLoss()
  (reg_loss): EmbLoss()
)

########## BERT4Rec ##########
BERT4Rec(
  (item_embedding): Embedding(3046141, 64, padding_idx=0)
  (position_embedding): Embedding(51, 64)
  (trm_encoder): TransformerEncoder(
    (layer): ModuleList(
      (0): TransformerLayer(
        (multi_head_attention): MultiHeadAttention(
          (query): Linear(in_features=64, out_features=64, bias=True)
          (key): Linear(in_features=64, out_features=64, bias=True)
          (value): Linear(in_features=64, out_features=64, bias=True)
          (softmax): Softmax(dim=-1)
          (attn_dropout): Dropout(p=0.2, inplace=False)
          (dense): Linear(in_features=64, out_features=64, bias=True)
          (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
          (out_dropout): Dropout(p=0.2, inplace=False)
        )
        (feed_forward): FeedForward(
          (dense_1): Linear(in_features=64, out_features=256, bias=True)
          (dense_2): Linear(in_features=256, out_features=64, bias=True)
          (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.2, inplace=False)
        )
      )
      (1): TransformerLayer(
        (multi_head_attention): MultiHeadAttention(
          (query): Linear(in_features=64, out_features=64, bias=True)
          (key): Linear(in_features=64, out_features=64, bias=True)
          (value): Linear(in_features=64, out_features=64, bias=True)
          (softmax): Softmax(dim=-1)
          (attn_dropout): Dropout(p=0.2, inplace=False)
          (dense): Linear(in_features=64, out_features=64, bias=True)
          (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
          (out_dropout): Dropout(p=0.2, inplace=False)
        )
        (feed_forward): FeedForward(
          (dense_1): Linear(in_features=64, out_features=256, bias=True)
          (dense_2): Linear(in_features=256, out_features=64, bias=True)
          (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.2, inplace=False)
        )
      )
    )
  )
  (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
  (dropout): Dropout(p=0.2, inplace=False)
)

########## GCSAN ##########
GCSAN(
  (item_embedding): Embedding(3046140, 64, padding_idx=0)
  (gnn): GNN(
    (linear_edge_in): Linear(in_features=64, out_features=64, bias=True)
    (linear_edge_out): Linear(in_features=64, out_features=64, bias=True)
  )
  (self_attention): TransformerEncoder(
    (layer): ModuleList(
      (0): TransformerLayer(
        (multi_head_attention): MultiHeadAttention(
          (query): Linear(in_features=64, out_features=64, bias=True)
          (key): Linear(in_features=64, out_features=64, bias=True)
          (value): Linear(in_features=64, out_features=64, bias=True)
          (softmax): Softmax(dim=-1)
          (attn_dropout): Dropout(p=0.2, inplace=False)
          (dense): Linear(in_features=64, out_features=64, bias=True)
          (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
          (out_dropout): Dropout(p=0.2, inplace=False)
        )
        (feed_forward): FeedForward(
          (dense_1): Linear(in_features=64, out_features=256, bias=True)
          (dense_2): Linear(in_features=256, out_features=64, bias=True)
          (LayerNorm): LayerNorm((64,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.2, inplace=False)
        )
      )
    )
  )
  (reg_loss): EmbLoss()
  (loss_fct): CrossEntropyLoss()
)

########## DIN ##########
DIN(
  (attention): SequenceAttLayer(
    (att_mlp_layers): MLPLayers(
      (mlp_layers): Sequential(
        (0): Dropout(p=0.0, inplace=False)
        (1): Linear(in_features=40, out_features=256, bias=True)
        (2): Sigmoid()
        (3): Dropout(p=0.0, inplace=False)
        (4): Linear(in_features=256, out_features=256, bias=True)
        (5): Sigmoid()
        (6): Dropout(p=0.0, inplace=False)
        (7): Linear(in_features=256, out_features=256, bias=True)
        (8): Sigmoid()
      )
    )
    (dense): Linear(in_features=256, out_features=1, bias=True)
  )
  (dnn_mlp_layers): MLPLayers(
    (mlp_layers): Sequential(
      (0): Dropout(p=0.0, inplace=False)
      (1): Linear(in_features=30, out_features=256, bias=True)
      (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): Dice(
        (sigmoid): Sigmoid()
      )
      (4): Dropout(p=0.0, inplace=False)
      (5): Linear(in_features=256, out_features=256, bias=True)
      (6): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (7): Dice(
        (sigmoid): Sigmoid()
      )
      (8): Dropout(p=0.0, inplace=False)
      (9): Linear(in_features=256, out_features=256, bias=True)
      (10): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (11): Dice(
        (sigmoid): Sigmoid()
      )
    )
  )
  (embedding_layer): ContextSeqEmbLayer(
    (token_embedding_table): ModuleDict(
      (user): FMEmbedding(
        (embedding): Embedding(1679, 10)
      )
      (item): FMEmbedding(
        (embedding): Embedding(3046140, 10)
      )
    )
    (float_embedding_table): ModuleDict()
    (token_seq_embedding_table): ModuleDict(
      (user): ModuleList()
      (item): ModuleList()
    )
    (float_seq_embedding_table): ModuleDict(
      (user): ModuleList()
      (item): ModuleList()
    )
  )
  (dnn_predict_layers): Linear(in_features=256, out_features=1, bias=True)
  (sigmoid): Sigmoid()
  (loss): BCEWithLogitsLoss()
)

########## DIEN ##########
DIEN(
  (interset_extractor): InterestExtractorNetwork(
    (gru): GRU(10, 10, batch_first=True)
    (auxiliary_net): MLPLayers(
      (mlp_layers): Sequential(
        (0): Dropout(p=0.0, inplace=False)
        (1): Linear(in_features=20, out_features=256, bias=True)
        (2): Dropout(p=0.0, inplace=False)
        (3): Linear(in_features=256, out_features=256, bias=True)
        (4): Dropout(p=0.0, inplace=False)
        (5): Linear(in_features=256, out_features=256, bias=True)
        (6): Dropout(p=0.0, inplace=False)
        (7): Linear(in_features=256, out_features=1, bias=True)
      )
    )
  )
  (interest_evolution): InterestEvolvingLayer(
    (attention_layer): SequenceAttLayer(
      (att_mlp_layers): MLPLayers(
        (mlp_layers): Sequential(
          (0): Dropout(p=0.0, inplace=False)
          (1): Linear(in_features=40, out_features=256, bias=True)
          (2): Sigmoid()
          (3): Dropout(p=0.0, inplace=False)
          (4): Linear(in_features=256, out_features=256, bias=True)
          (5): Sigmoid()
          (6): Dropout(p=0.0, inplace=False)
          (7): Linear(in_features=256, out_features=256, bias=True)
          (8): Sigmoid()
        )
      )
      (dense): Linear(in_features=256, out_features=1, bias=True)
    )
    (dynamic_rnn): DynamicRNN(
      (rnn): AUGRUCell()
    )
  )
  (embedding_layer): ContextSeqEmbLayer(
    (token_embedding_table): ModuleDict(
      (user): FMEmbedding(
        (embedding): Embedding(1679, 10)
      )
      (item): FMEmbedding(
        (embedding): Embedding(3046140, 10)
      )
    )
    (float_embedding_table): ModuleDict()
    (token_seq_embedding_table): ModuleDict(
      (user): ModuleList()
      (item): ModuleList()
    )
    (float_seq_embedding_table): ModuleDict(
      (user): ModuleList()
      (item): ModuleList()
    )
  )
  (dnn_mlp_layers): MLPLayers(
    (mlp_layers): Sequential(
      (0): Dropout(p=0.0, inplace=False)
      (1): Linear(in_features=30, out_features=256, bias=True)
      (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): Dice(
        (sigmoid): Sigmoid()
      )
      (4): Dropout(p=0.0, inplace=False)
      (5): Linear(in_features=256, out_features=256, bias=True)
      (6): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (7): Dice(
        (sigmoid): Sigmoid()
      )
      (8): Dropout(p=0.0, inplace=False)
      (9): Linear(in_features=256, out_features=256, bias=True)
      (10): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (11): Dice(
        (sigmoid): Sigmoid()
      )
    )
  )
  (dnn_predict_layer): Linear(in_features=256, out_features=1, bias=True)
  (sigmoid): Sigmoid()
  (loss): BCEWithLogitsLoss()
)

########## MMa4CTR ##########
MLP(
  (activation_function): Tanh()
  (_features): Sequential(
    (0): Linear(in_features=169, out_features=128, bias=True)
    (1): Tanh()
    (2): Linear(in_features=128, out_features=32, bias=True)
    (3): Tanh()
    (4): Linear(in_features=32, out_features=4, bias=True)
    (5): Tanh()
  )
  (_classifier): Linear(in_features=4, out_features=1, bias=True)
  (predict): Sigmoid()
)

```

### 6.3. Baselines: 
**BPR**: BPR Bayesian Personalized Ranking from Implicit Feedback<br>
**FPMC**: Factorizing personalized Markov chains for next-basket recommendation<br>
**NGCF**: Neural Graph Collaborative Filtering<br>
**LightGCN**: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation<br>
**BERT4Rec**: BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer<br>
**GCSAN**: Graph Contextualized Self-Attention Network for Session-based Recommendation<br>
**DIN**: Deep Interest Network for Click-Through Rate Prediction<br>
**DIEN**: Deep Interest Evolution Network for Click-Through Rate Prediction<br>`
