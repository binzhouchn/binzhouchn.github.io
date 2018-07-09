---
title:  "用pytorch构建一个DSSM神经网络"
layout: post
categories: python
tags:  Python Pytorch DSSM
author: binzhou
---

* content
{:toc}

```python
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from config import opt
from sklearn.externals import joblib #jbolib模块
```

# 数据准备
## 把数据处理成DSSM要求的格式，一个query，一个pos_doc，四个neg_doc

```python
# 处理q1,q2
import numpy as np
import pandas as pd
train = pd.read_csv('mojing/train.csv')
```

```python
train.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>q1</th>
      <th>q2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Q397345</td>
      <td>Q538594</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Q193805</td>
      <td>Q699273</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>Q085471</td>
      <td>Q676160</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>Q189314</td>
      <td>Q438123</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Q267714</td>
      <td>Q290126</td>
    </tr>
  </tbody>
</table>
</div>

<!--more-->

```python
a = train[train.label == 1].copy()
a.columns = ['label','query','pos_doc']
reorder_col = ['label','pos_doc','query']
b = a.loc[:, reorder_col].copy()
b.columns = ['label','query','pos_doc']
a = a.append(b,ignore_index=True)
a.drop_duplicates(subset=['query'],inplace=True)
a.index = np.arange(len(a)) # 重排序
```


```python
## 抽不为1的4个doc (neg)
b = train[train.label == 0]
suffle_pool = list(b.q1) + list(b.q2)
```


```python
def ff(s):
    l = []
    l += b[b.q1 == s].q2.tolist()
    l += b[b.q2 == s].q1.tolist()
    l = list(set(l))[:4]
    l_ = l.copy()
    l_.append(s)
    if len(l) < 4:
        tmp = np.random.choice(suffle_pool,5,replace=False).tolist()
        cha = set(tmp) - set(l_)
        l += list(cha)[:4-len(l)]
    return l
```


```python
%%time
a['neg_doc'] = a['query'].apply(ff)
```

    CPU times: user 27.5 s, sys: 174 ms, total: 27.7 s
    Wall time: 27.7 s
    


```python
a.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>query</th>
      <th>pos_doc</th>
      <th>neg_doc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Q397345</td>
      <td>Q538594</td>
      <td>[Q521609, Q175780, Q068667, Q632305]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Q369715</td>
      <td>Q658908</td>
      <td>[Q696189, Q428940, Q198861, Q578218]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Q537991</td>
      <td>Q268444</td>
      <td>[Q011513, Q022092, Q229357, Q498790]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Q639518</td>
      <td>Q053248</td>
      <td>[Q392805, Q657314, Q647857, Q539673]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Q683881</td>
      <td>Q087150</td>
      <td>[Q432305, Q723726, Q272217, Q206480]</td>
    </tr>
  </tbody>
</table>
</div>



## 把question替换成words


```python
%%time
# 把问题替换成词
QUESTION_PATH = 'mojing/question.csv'
questions = pd.read_csv(QUESTION_PATH)
question_dict = {}
for key, value in zip(questions['qid'],questions['words']):
    question_dict[key] = value
```

    CPU times: user 2.53 s, sys: 293 ms, total: 2.82 s
    Wall time: 4.57 s
    


```python
%%time
a['query'] = a['query'].apply(lambda x : question_dict.get(x))
a['pos_doc'] = a['pos_doc'].apply(lambda x : question_dict.get(x))
a['neg_doc'] = a['neg_doc'].apply(lambda l : [question_dict.get(x) for x in l])
```

    CPU times: user 34.6 ms, sys: 1.96 ms, total: 36.6 ms
    Wall time: 35.9 ms
    


```python
a.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>query</th>
      <th>pos_doc</th>
      <th>neg_doc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>W04465 W04058 W05284 W02916</td>
      <td>W18238 W18843 W01490 W09905</td>
      <td>[W06579 W17705 W09745 W10938 W01490 W07863, W1...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>W12908 W19355 W08041 W06040 W18399 W01773 W16319</td>
      <td>W12908 W06112 W08041 W17342</td>
      <td>[W13157 W16564 W08020 W08924 W08276 W11824 W04...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>W16429 W14586 W03914 W09648 W02262 W18399 W06682</td>
      <td>W13522 W05733 W17917 W10691 W16319</td>
      <td>[W00022 W06756, W18830 W05733 W08276 W06179 W0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>W04182 W05733 W03914 W09400 W13868</td>
      <td>W04476 W11385 W05733 W18804 W16686 W19081 W18448</td>
      <td>[W12440 W19536 W17945 W18080 W15175 W19355 W17...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>W17378 W14586 W01661 W03914 W04182 W12803 W02262</td>
      <td>W07777 W05733 W04476 W11385 W10628 W08815 W047...</td>
      <td>[W18238 W05284 W09158 W04745 W03390, W17378 W0...</td>
    </tr>
  </tbody>
</table>
</div>



## 把words进行编码


```python
import numpy as np
from tqdm import tqdm

class BOW(object):
    def __init__(self, X, min_count=10, maxlen=100):
        """
        X: [[w1, w2],]]
        """
        self.X = X
        self.min_count = min_count
        self.maxlen = maxlen
        self.__word_count()
        self.__idx()
        self.__doc2num()

    def __word_count(self):
        wc = {}
        for ws in tqdm(self.X, desc='   Word Count'):
            for w in ws:
                if w in wc:
                    wc[w] += 1
                else:
                    wc[w] = 1
        self.word_count = {i: j for i, j in wc.items() if j >= self.min_count}

    def __idx(self):
        self.idx2word = {i + 1: j for i, j in enumerate(self.word_count)}
        self.word2idx = {j: i for i, j in self.idx2word.items()}

    def __doc2num(self):
        doc2num = []
        for text in tqdm(self.X, desc='Doc To Number'):
            s = [self.word2idx.get(i, 0) for i in text[:self.maxlen]]
            doc2num.append(s + [0]*(self.maxlen-len(s)))  # 未登录词全部用0表示
        self.doc2num = np.asarray(doc2num)
def get_ids(qids):
    ids = []
    for t_ in qids:
        ids.append(int(t_[1:]))
    return np.asarray(ids)


def get_texts(file_path, question_path):
    qes = pd.read_csv(question_path)
    file = pd.read_csv(file_path)
    q1id, q2id = file['q1'], file['q2']
    id1s, id2s = get_ids(q1id), get_ids(q2id)
    all_words = qes['words']
    texts = []
    for t_ in zip(id1s, id2s):
        texts.append(all_words[t_[0]] + ' ' + all_words[t_[1]])
    return texts
TRAIN_PATH = 'mojing/train.csv'
TEST_PATH = 'mojing/test.csv'
QUESTION_PATH = 'mojing/question.csv'
train_texts = get_texts(TRAIN_PATH, QUESTION_PATH)
test_texts = get_texts(TEST_PATH, QUESTION_PATH)
a1 = train_texts + test_texts
a1 = [x.split(' ') for x in a1]
bow = BOW(a1,min_count=1,maxlen=24) # count大于1，句子(q1,q2)相加最大长度为24
del a1
```

       Word Count: 100%|██████████| 427342/427342 [00:01<00:00, 332864.53it/s]
    Doc To Number: 100%|██████████| 427342/427342 [00:03<00:00, 133272.96it/s]
    


```python
##############################
# 之前编码好的已经存成embedding matrix
# 可以自己训练
# 训练代码
word_embed = pd.read_csv('mojing/word_embed.txt',header=None)
word_embed.columns = ['wv']
word_embed_dict = dict()
for s in word_embed.wv.values:
    l = s.split(' ')
    word_embed_dict[l[0]] = list(map(float,l[1:]))
word_embed_dict['UNK'] = [0]*300
vocab_size = len(word_embed_dict)
embedding_matrix = np.zeros((vocab_size+1,300))
for key, value in bow.word2idx.items():
    embedding_matrix[value] = word_embed_dict.get(key)
embedding_matrix
# np.save('save/embedding_matrix.npz',embedding_matrix)
# embedding_matrix = np.load('save/embedding_matrix.npz.npy')
```




    array([[ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 2.29952765, -4.29687977,  3.71340919, ...,  0.99011242,
             0.41728863,  3.15365911],
           [-1.52279055,  2.12538552, -0.3590863 , ..., -2.17771411,
             1.37241161, -3.44047666],
           ...,
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ]])




```python
def fill_unkQ1(s):
    l1 = s.split(' ')
    l1 = [bow.word2idx.get(x) if x in bow.word2idx.keys() else 0 for x in l1]
    return l1
def fill_unkQ2(l):
    l1 = [[bow.word2idx.get(x) if x in bow.word2idx.keys() else 0 for x in s.split(' ')] for s in l]
    return l1
a['query'] = a['query'].apply(fill_unkQ1)
a['pos_doc'] = a['pos_doc'].apply(fill_unkQ1)
a['neg_doc'] = a['neg_doc'].apply(fill_unkQ2)
```


```python
a.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>query</th>
      <th>pos_doc</th>
      <th>neg_doc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>[1, 2, 3, 4]</td>
      <td>[5, 6, 7, 8]</td>
      <td>[[723, 1649, 27, 151, 7, 25], [124, 773, 99, 2...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[45, 21, 46, 47, 48, 49, 30]</td>
      <td>[45, 50, 46, 51]</td>
      <td>[[39, 324, 837, 66, 287, 238, 1394, 53, 25], [...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>[52, 26, 53, 54, 55, 48, 56]</td>
      <td>[57, 58, 59, 60, 30]</td>
      <td>[[951, 1383], [2317, 58, 287, 593, 25], [570, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>[71, 58, 53, 72, 73]</td>
      <td>[10, 74, 58, 75, 76, 77, 78]</td>
      <td>[[333, 116, 594, 764, 698, 21, 613], [5, 28, 6...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>[20, 26, 98, 53, 71, 99, 55]</td>
      <td>[100, 58, 10, 74, 101, 102, 103, 48, 104]</td>
      <td>[[5, 3, 382, 103, 40], [20, 111, 7, 30, 5, 293...</td>
    </tr>
  </tbody>
</table>
</div>


## 数据处理成dssm要求的tensor格式样式


```python
J = 4
l_Qs = [[] for j in range(len(a))]
pos_l_Ds = [[] for j in range(len(a))]
# neg_l_Ds = [[] for j in range(J)]
neg_l_Ds = np.zeros((J,len(a))).tolist()
for i in range(len(a['query'])):
    l_Qs[i] = Variable(torch.from_numpy(np.array(a['query'][i]).reshape(1,len(a['query'][i]))).long())
    pos_l_Ds[i] = Variable(torch.from_numpy(np.array(a['pos_doc'][i]).reshape(1,len(a['pos_doc'][i]))).long())
    for j in range(J):
        neg_l_Ds[j][i] = Variable(torch.from_numpy(np.array(a['neg_doc'][i][j]).reshape(1,len(a['neg_doc'][i][j]))).long())
```

# 构建模型


```python
from BasicModule import BasicModule
import torch as t
import numpy as np
from torch import nn

LETTER_GRAM_SIZE = 1 # See section 3.2. trigram_based word_uni_gram 暂时没用到
WINDOW_SIZE = 3 # See section 3.2. 暂时没用到
TOTAL_LETTER_GRAMS = opt.vocab_size # Determined from data. See section 3.2. 20893 暂时没用到
WORD_DEPTH =300 # See equation (1).  这里我用词向量训练好的embedding 300维
K = 128 # Dimensionality of the max-pooling layer. See section 3.4.
L = 64 # Dimensionality of latent semantic space. See section 3.5.
J = 4 # Number of random unclicked documents serving as negative examples for a query. See section 4.
FILTER_LENGTH = 1 # We only consider one time step for convolutions.
sample_size = 10000

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class CDSSM(BasicModule):
    def __init__(self):
        super(CDSSM, self).__init__()
        # layers for query
        self.query_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        self.query_sem = nn.Linear(K, L)
        # layers for docs
        self.doc_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        self.doc_sem = nn.Linear(K, L)
        # learning gamma
        self.learn_gamma = nn.Conv1d(1, 1, 1)
        # embedding
        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)
        if opt.embedding_path:
            self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)))
    def forward(self, q, pos, negs):
        # Query model. The paper uses separate neural nets for queries and documents (see section 5.2).
        # To make it compatible with Conv layer we reshape it to: (batch_size, WORD_DEPTH, query_len)
#         q = self.encoder(q)
#         pos = self.encoder(pos)
#         negs = [self.encoder(neg) for neg in negs]
        q = q.transpose(1,2)
        # In this step, we transform each word vector with WORD_DEPTH dimensions into its
        # convolved representation with K dimensions. K is the number of kernels/filters
        # being used in the operation. Essentially, the operation is taking the dot product
        # of a single weight matrix (W_c) with each of the word vectors (l_t) from the
        # query matrix (l_Q), adding a bias vector (b_c), and then applying the tanh activation.
        # That is, h_Q = tanh(W_c • l_Q + b_c). Note: the paper does not include bias units.
        q_c = F.tanh(self.query_conv(q))
        # Next, we apply a max-pooling layer to the convolved query matrix.
        q_k = kmax_pooling(q_c, 2, 1)
        q_k = q_k.transpose(1,2)
        # In this step, we generate the semantic vector represenation of the query. This
        # is a standard neural network dense layer, i.e., y = tanh(W_s • v + b_s). Again,
        # the paper does not include bias units.
        q_s = F.tanh(self.query_sem(q_k))
        q_s = q_s.resize(L)
        # # The document equivalent of the above query model for positive document
        pos = pos.transpose(1,2)
        pos_c = F.tanh(self.doc_conv(pos))
        pos_k = kmax_pooling(pos_c, 2, 1)
        pos_k = pos_k.transpose(1,2)
        pos_s = F.tanh(self.doc_sem(pos_k))
        pos_s = pos_s.resize(L)
        # # The document equivalent of the above query model for negative documents
        negs = [neg.transpose(1,2) for neg in negs]
        neg_cs = [F.tanh(self.doc_conv(neg)) for neg in negs]
        neg_ks = [kmax_pooling(neg_c, 2, 1) for neg_c in neg_cs]
        neg_ks = [neg_k.transpose(1,2) for neg_k in neg_ks]
        neg_ss = [F.tanh(self.doc_sem(neg_k)) for neg_k in neg_ks]
        neg_ss = [neg_s.resize(L) for neg_s in neg_ss]
        # Now let us calculates the cosine similarity between the semantic representations of
        # a queries and documents
        # dots[0] is the dot-product for positive document, this is necessary to remember
        # because we set the target label accordingly
        dots = [q_s.dot(pos_s)]
        dots = dots + [q_s.dot(neg_s) for neg_s in neg_ss]
        # dots is a list as of now, lets convert it to torch variable
        dots = torch.stack(dots)
        # In this step, we multiply each dot product value by gamma. In the paper, gamma is
        # described as a smoothing factor for the softmax function, and it's set empirically
        # on a held-out data set. We're going to learn gamma's value by pretending it's
        # a single 1 x 1 kernel.
        with_gamma = self.learn_gamma(dots.resize(J+1, 1, 1))
        # Finally, we use the softmax function to calculate P(D+|Q).
        prob = F.softmax(with_gamma)
        return prob

model = CDSSM()
model.cuda()
```




    CDSSM(
      (query_conv): Conv1d(300, 128, kernel_size=(1,), stride=(1,))
      (query_sem): Linear(in_features=128, out_features=64, bias=True)
      (doc_conv): Conv1d(300, 128, kernel_size=(1,), stride=(1,))
      (doc_sem): Linear(in_features=128, out_features=64, bias=True)
      (learn_gamma): Conv1d(1, 1, kernel_size=(1,), stride=(1,))
      (encoder): Embedding(20893, 300)
    )



# 跑DSSM模型 SGD 一个个样本跑


```python
encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)
if opt.embedding_path:
            encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)))
```


```python
%%time
# cpu version
# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# output variable, remember the cosine similarity with positive doc was at 0th index
y = np.ndarray(1)
# CrossEntropyLoss expects only the index as a long tensor
y[0] = 0
y = Variable(torch.from_numpy(y).long())

for i in range(sample_size):
    y_pred = model(l_Qs[i], pos_l_Ds[i], [neg_l_Ds[j][i] for j in range(J)])  
    loss = criterion(y_pred.resize(1,J+1), y)
    print('%d training loss'%i, loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```


```python
l_Qs[8400:8500]
```




    [tensor([[ 749,   48,    7,  142,    5,  773]]),
     tensor([[ 280,  234,  907,  280]]),
     tensor([[   5,   66,  227,   30,   78]]),
     tensor([[  58,  234,   90,  493,   16,   35,  267,  268,  493,   13]]),
     tensor([[  20,    7,  134,   30,  159,   78,   13]]),
     tensor([[ 235,   95,  257,  301,  260,   84,   30]]),
     tensor([[   20,  1000,   578,   225]]),
     tensor([[ 543,  258,  222,   18,  138,  680,  330,   53,   66,  269,
               688,  215,   10,  107,   78]]),
     tensor([[ 175,    7,  555]]),
     tensor([[   94,    95,    53,    26,    35,  3944,    99,    58]]),
     tensor([[  10,   94,   95,   21,   58,    3,  113,  109]]),
     tensor([[  58,  348,  234]]),
     tensor([[ 58,  88,  90,  84]]),
     tensor([[ 832,  103,   21,  992]]),
     tensor([[  51,  158,   17,  603,  232,   20]]),
     tensor([[ 907,  670,   20,  757,   45,  319]]),
     tensor([[  20,   21,  235,  305,   98,   53,   35,   10,   55,   48,
                56]]),
     tensor([[   10,    55,    21,   106,   107,  2155,   426,    52,   481]]),
     tensor([[  20,   33,   57,   58,   35,    5,  261,   30]]),
     tensor([[  103,  2500,  3197,   350]]),
     tensor([[  20,   33,  246,  648,  152]]),
     tensor([[ 3954,     7]]),
     tensor([[  22,  104,   35,  437,  108,  109]]),
     tensor([[ 472,   20,   21,  252,    5,  816]]),
     tensor([[  20,   21,    7,  147]]),
     tensor([[  58,   40,   35,  560,  150,  108,  109]]),
     tensor([[ 227,   18,  570,  227]]),
     tensor([[ 296,    7,   46,  349,  234,   75,   15,  220]]),
     tensor([[  20,   21,  648,  570,   59,  227,   20,  444,   21,  320,
               392,   30]]),
     tensor([[  103,    21,   530,  1539,    30]]),
     tensor([[  20,   33,    6,  353,    7,    8]]),
     tensor([[   9,  467,   30,  287,   58,   25]]),
     tensor([[  11,  426,  568,   66,  269,   25]]),
     tensor([[ 379,  299,  181,  287,  108,   25]]),
     tensor([[   74,   227,    31,   152,    59,  3959]]),
     tensor([[ 3961,   108,    24,     5,     7]]),
     tensor([[   20,  1169,   146,     3,    84,   106]]),
     tensor([[ 106,   22,  400,   75,  267,  436,   25]]),
     tensor([[  197,   197,   376,   968,  3962,   244,    21]]),
     tensor([[  10,  478,   40,   41,  106,    3,  107,  108,  109,   13]]),
     tensor([[   5,    3,  382,  400,   22]]),
     tensor([[ 235,   95,  379,   12]]),
     tensor([[  20,  158,   58,   40,   10,   74,  841,  795,   30]]),
     tensor([[ 1346,    30,   106,     3,   113,    24]]),
     tensor([[ 31,  59,  39]]),
     tensor([[   58,  3964,    83,    25,    13]]),
     tensor([[ 3965,    12,    78,    13]]),
     tensor([[ 107,  448,    7,  381,  501,   20]]),
     tensor([[ 207,  293,   22,  498]]),
     tensor([[ 112,  257,  258,   21]]),
     tensor([[ 215,   11,   31,   78]]),
     tensor([[ 124,    6,   20,    7,  555]]),
     tensor([[ 45,  21,  50,  46]]),
     tensor([[   15,    81,    20,   198,  3143,    30,    41,   192,   280,
               1066]]),
     tensor([[ 19,   5,  44,  57,  58]]),
     tensor([[  5,  39,   9]]),
     tensor([[ 1445,   269,    55,   288,    25]]),
     tensor([[ 603,  281,  349,  234,   13]]),
     tensor([[  638,   505,  1856,   235,    95,  1041]]),
     tensor([[  107,   378,  3547,   104,    25]]),
     tensor([[  58,   32,  563,  654,  108,  109]]),
     tensor([[   10,   463,  2534,    30,   343,   413,   291]]),
     tensor([[ 1203,   207,   312]]),
     tensor([[  58,   26,  771,  661,   25,   13]]),
     tensor([[  58,   32,  150,  287,  108,  109]]),
     tensor([[ 142,   21,  257,    5,  359,  362]]),
     tensor([[ 472,  975,   25,   13]]),
     tensor([[  20,   10,   99,  379,   30]])]




```python
%%time
# gpu version
# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
criterion.cuda()

# output variable, remember the cosine similarity with positive doc was at 0th index
y = np.ndarray(1)
# CrossEntropyLoss expects only the index as a long tensor
y[0] = 0
y = Variable(torch.from_numpy(y).long()).cuda()

for i in range(sample_size):
#     y_pred = model(l_Qs[i], pos_l_Ds[i], [neg_l_Ds[j][i] for j in range(J)])
    y_pred = model(encoder(l_Qs[i]).cuda(), encoder(pos_l_Ds[i]).cuda(), [encoder(neg_l_Ds[j][i]).cuda() for j in range(J)])
    loss = criterion(y_pred.resize(1,J+1), y)
    if i % 100 == 0:
        print('%d training loss'%i, loss.cpu().data[0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

    /fdp/.local/lib/python3.6/site-packages/torch/tensor.py:255: UserWarning: non-inplace resize is deprecated
      warnings.warn("non-inplace resize is deprecated")
    /usr/local/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:85: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
    /usr/local/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:17: UserWarning: invalid index of a 0-dim tensor. This will be an error in PyTorch 0.5. Use tensor.item() to convert a 0-dim tensor to a Python number
    

    0 training loss tensor(1.6347)
    100 training loss tensor(1.6490)
    200 training loss tensor(1.6017)
    300 training loss tensor(1.6053)
    400 training loss tensor(1.6080)
    500 training loss tensor(1.6090)
    600 training loss tensor(1.5162)
    700 training loss tensor(1.6037)
    800 training loss tensor(1.5920)
    900 training loss tensor(1.5668)
    1000 training loss tensor(1.5862)
    1100 training loss tensor(1.5944)
    1200 training loss tensor(1.5928)
    1300 training loss tensor(1.6171)
    1400 training loss tensor(1.4929)
    1500 training loss tensor(1.6628)
    1600 training loss tensor(1.5990)
    1700 training loss tensor(1.6028)
    1800 training loss tensor(1.5567)
    1900 training loss tensor(1.4474)
    2000 training loss tensor(1.6271)
    2100 training loss tensor(1.6788)
    2200 training loss tensor(1.6578)
    2300 training loss tensor(1.4889)
    2400 training loss tensor(1.6383)
    2500 training loss tensor(1.3773)
    2600 training loss tensor(1.3692)
    2700 training loss tensor(1.6813)
    2800 training loss tensor(1.5980)
    2900 training loss tensor(1.5012)
    3000 training loss tensor(1.7437)
    3100 training loss tensor(1.5664)
    3200 training loss tensor(1.6467)
    3300 training loss tensor(1.2001)
    3400 training loss tensor(1.6904)
    3500 training loss tensor(1.2545)
    3600 training loss tensor(1.3611)
    3700 training loss tensor(1.6140)
    3800 training loss tensor(1.5766)
    3900 training loss tensor(0.9233)
    4000 training loss tensor(1.4807)
    4100 training loss tensor(1.7822)
    4200 training loss tensor(1.5655)
    4300 training loss tensor(1.6642)
    4400 training loss tensor(1.1545)
    4500 training loss tensor(1.5872)
    4600 training loss tensor(1.0026)
    4700 training loss tensor(1.0715)
    4800 training loss tensor(1.0602)
    4900 training loss tensor(1.7952)
    5000 training loss tensor(1.5088)
    5100 training loss tensor(1.6724)
    5200 training loss tensor(1.7706)
    5300 training loss tensor(1.4433)
    5400 training loss tensor(1.4279)
    5500 training loss tensor(0.9569)
    5600 training loss tensor(1.8893)
    5700 training loss tensor(1.5960)
    5800 training loss tensor(0.9451)
    5900 training loss tensor(1.7464)
    6000 training loss tensor(1.5589)
    6100 training loss tensor(1.7813)
    6200 training loss tensor(1.1629)
    6300 training loss tensor(1.4071)
    6400 training loss tensor(1.8934)
    6500 training loss tensor(1.8045)
    6600 training loss tensor(1.5014)
    6700 training loss tensor(0.9430)
    6800 training loss tensor(1.3834)
    6900 training loss tensor(1.6786)
    7000 training loss tensor(1.1853)
    7100 training loss tensor(1.7041)
    7200 training loss tensor(1.3836)
    7300 training loss tensor(1.4180)
    7400 training loss tensor(1.7372)
    7500 training loss tensor(1.7524)
    7600 training loss tensor(1.0825)
    7700 training loss tensor(1.7825)
    7800 training loss tensor(1.7910)
    7900 training loss tensor(1.8072)
    8000 training loss tensor(1.8051)
    8100 training loss tensor(0.9370)
    8200 training loss tensor(1.2102)
    8300 training loss tensor(1.3358)
    8400 training loss tensor(1.7570)
    


    

    IndexErrorTraceback (most recent call last)

    <timed exec> in <module>()
    

    IndexError: list index out of range



```python
# 好像没有收敛。。还有报错
```


```python

```
