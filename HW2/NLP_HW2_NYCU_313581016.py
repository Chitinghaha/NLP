

# !pip install opencc-python-reimplemented
# ! pip install seaborn
# ! pip install opencc
# ! pip install -U scikit-learn

import numpy as np
import pandas as pd
import torch
import torch.nn
import torch.nn.utils.rnn
import torch.utils.data
import matplotlib.pyplot as plt
import seaborn as sns
import opencc
import os
from sklearn.model_selection import train_test_split

data_path = ''

df_train = pd.read_csv(os.path.join(data_path, 'arithmetic_train.csv'))
df_eval = pd.read_csv(os.path.join(data_path, 'arithmetic_eval.csv'))
df_train.head()

# transform the input data to string
df_train['tgt'] = df_train['tgt'].apply(lambda x: str(x))
df_train['src'] = df_train['src'].add(df_train['tgt'])
df_train['len'] = df_train['src'].apply(lambda x: len(x))

df_eval['tgt'] = df_eval['tgt'].apply(lambda x: str(x))
df_eval['src'] = df_eval['src'].add(df_eval['tgt'])
df_eval['len'] = df_eval['src'].apply(lambda x: len(x))

"""# Build Dictionary
 - The model cannot perform calculations directly with plain text.
 - Convert all text (numbers/symbols) into numerical representations.
 - Special tokens
    - '&lt;pad&gt;'
        - Each sentence within a batch may have different lengths.
        - The length is padded with '&lt;pad&gt;' to match the longest sentence in the batch.
    - '&lt;eos&gt;'
        - Specifies the end of the generated sequence.
        - Without '&lt;eos&gt;', the model will not know when to stop generating.
"""

char_to_id = {}
id_to_char = {}

# write your code here
# Build a dictionary and give every token in the train dataset an id
# The dictionary should contain <eos> and <pad>
# char_to_id is to conver charactors to ids, while id_to_char is the opposite

# 定義字符到ID的dict，包含 <pad> 和 <eos>
char_to_id = {
    '<pad>': 0,  # 用來填充
    '<eos>': 1,  # 用來表示結束
}

# 將數字 '0' 到 '9' 加入 dict
for i in range(10):
    char_to_id[str(i)] = len(char_to_id)  # 將每個數字轉換為對應的 id

# 將基本的運算符號 +, -, *, 和 () 加入 dict
operators = ['+', '-', '*', '(', ')','=','a']
for op in operators:
    char_to_id[op] = len(char_to_id)  # 每個符號分配唯一的 id

# 反向 dict，將ID轉換回字符
id_to_char = {v: k for k, v in char_to_id.items()}

vocab_size = len(char_to_id)
print('Vocab size ：{}'.format(vocab_size))

print("char_to_id:", char_to_id)
print("id_to_char:", id_to_char)

"""# Data Preprocessing
 - The data is processed into the format required for the model's input and output.
 - Example: 1+2-3=0
     - Model input: 1 + 2 - 3 = 0
     - Model output: / / / / / 0 &lt;eos&gt;  (the '/' can be replaced with &lt;pad&gt;)
     - The key for the model's output is that the model does not need to predict the next character of the previous part. What matters is that once the model sees '=', it should start generating the answer, which is '0'. After generating the answer, it should also generate&lt;eos&gt;

"""

from tqdm import tqdm

def generate_ids(row):
    # src 的 char_id_list，包含 <eos>
    char_id_list = [char_to_id.get(char, char_to_id['<pad>']) for char in row['src']]+ [char_to_id['<eos>']]

    # tgt 的 label_id_list，包含 <eos>
    label_id_list = [char_to_id.get(char, char_to_id['<pad>']) for char in row['tgt']] + [char_to_id['<eos>']]

    # 用 0 填充 label_id_list 以匹配 char_id_list 的長度
    if len(label_id_list) < len(char_id_list):
        label_id_list = [0] * (len(char_id_list) - len(label_id_list)) + label_id_list

    return pd.Series([char_id_list, label_id_list])

# 將進度條應用於 DataFrame 操作
tqdm.pandas(desc="Processing training data")
df_train[['char_id_list', 'label_id_list']] = df_train.progress_apply(generate_ids, axis=1)

tqdm.pandas(desc="Processing evaluation data")
df_eval[['char_id_list', 'label_id_list']] = df_eval.progress_apply(generate_ids, axis=1)

# 輸出結果
df_train.head()
df_eval.head()

# # 結果儲存到 CSV 檔案
# df_train.to_csv("./drive/MyDrive/NLP_HW2/processed_train_data.csv", index=False)
# df_eval.to_csv("./drive/MyDrive/NLP_HW2/processed_eval_data.csv", index=False)

df_train = pd.read_csv( 'processed_train.csv')
df_eval = pd.read_csv( 'processed_eval_data.csv')

df_train.head()

df_train = df_train.iloc[:, 1:]
df_eval = df_eval.iloc[:, 1:]

df_train.shape

"""# Hyper Parameters

|Hyperparameter|Meaning|Value|
|-|-|-|
|`batch_size`|Number of data samples in a single batch|64|
|`epochs`|Total number of epochs to train|10|
|`embed_dim`|Dimension of the word embeddings|256|
|`hidden_dim`|Dimension of the hidden state in each timestep of the LSTM|256|
|`lr`|Learning Rate|0.001|
|`grad_clip`|To prevent gradient explosion in RNNs, restrict the gradient range|1|
"""

batch_size = 64
epochs = 2
embed_dim = 256
hidden_dim = 256
lr = 0.001
grad_clip = 1

"""# Data Batching
- Use `torch.utils.data.Dataset` to create a data generation tool called  `dataset`.
- The, use `torch.utils.data.DataLoader` to randomly sample from the `dataset` and group the samples into batches.
"""

import ast

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        # return the amount of data
        return len(self.sequences)# Write your code here

    def __getitem__(self, index):
        sequence = self.sequences.iloc[index]
        x = ast.literal_eval(sequence['char_id_list']) # 提取 char_id_list
        y = ast.literal_eval(sequence['label_id_list'])  # 提取 label_id_list

        # x 不包含最後一個元素，y 右偏移一格
        x = x[:-1]
        y = y[1:]

        return x, y

# collate function, used to build dataloader
def collate_fn(batch):
    batch_x = [torch.tensor(data[0]) for data in batch]
    batch_y = [torch.tensor(data[1]) for data in batch]
    batch_x_lens = torch.LongTensor([len(x) for x in batch_x])
    batch_y_lens = torch.LongTensor([len(y) for y in batch_y])

    # Pad the input sequence
    pad_batch_x = torch.nn.utils.rnn.pad_sequence(batch_x,
                                                  batch_first=True,
                                                  padding_value=char_to_id['<pad>'])

    pad_batch_y = torch.nn.utils.rnn.pad_sequence(batch_y,
                                                  batch_first=True,
                                                  padding_value=char_to_id['<pad>'])

    return pad_batch_x, pad_batch_y, batch_x_lens, batch_y_lens

ds_train = Dataset(df_train[['char_id_list', 'label_id_list']])
ds_eval = Dataset(df_eval[['char_id_list', 'label_id_list']])

from torch.utils.data import DataLoader

# Build dataloader of train set and eval set, collate_fn is the collate function
dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dl_eval = DataLoader(ds_eval, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

"""# Model Design

## Execution Flow
1. Convert all characters in the sentence into embeddings.
2. Pass the embeddings through an LSTM sequentially.
3. The output of the LSTM is passed into another LSTM, and additional layers can be added.
4. The output from all time steps of the final LSTM is passed through a Fully Connected layer.
5. The character corresponding to the maximum value across all output dimensions is selected as the next character.

## Loss Function
Since this is a classification task, Cross Entropy is used as the loss function.

## Gradient Update
Adam algorithm is used for gradient updates.
"""

class CharRNN(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharRNN, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=embed_dim,
                                            padding_idx=char_to_id['<pad>'])

        self.rnn_layer1 = torch.nn.LSTM(input_size=embed_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True)

        self.rnn_layer2 = torch.nn.LSTM(input_size=hidden_dim,
                                        hidden_size=hidden_dim,
                                        batch_first=True)

        self.linear = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_dim,
                                                          out_features=hidden_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(in_features=hidden_dim,
                                                          out_features=vocab_size))

    def forward(self, batch_x, batch_x_lens):
        return self.encoder(batch_x, batch_x_lens)

    # The forward pass of the model
    def encoder(self, batch_x, batch_x_lens):
        batch_x = self.embedding(batch_x)

        batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x,
                                                          batch_x_lens,
                                                          batch_first=True,
                                                          enforce_sorted=False)

        batch_x, _ = self.rnn_layer1(batch_x)
        batch_x, _ = self.rnn_layer2(batch_x)

        batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x,
                                                            batch_first=True)

        batch_x = self.linear(batch_x)

        return batch_x

    def generator(self, start_char, max_len=200):
        char_list = [char_to_id[c] for c in start_char]
        next_char = None

        while len(char_list) < max_len:
            # Pack the char_list to tensor
            # print(char_list)
            input_tensor = torch.tensor(char_list).unsqueeze(0).to(device)  # Add batch dimension
            input_tensor = input_tensor.to(device)  # Move to the appropriate device (GPU or CPU)

            logits = self.encoder(input_tensor, torch.tensor([len(char_list)]))
            # Get the prediction for the last time step
            y = logits[0, -1, :] # Use the output from the last time step
            print(y)

            # Obtain the next token prediction y
            next_char = torch.argmax(y, dim=0).item()  # Get the index of the max log-probability

            if next_char == char_to_id['<eos>']:
                break

            char_list.append(next_char)
            # print(char_list)

        return [id_to_char[ch_id] for ch_id in char_list]

"""改用 RNN"""

# class CharRNN(torch.nn.Module):
#     def __init__(self, vocab_size, embed_dim, hidden_dim):
#         super(CharRNN, self).__init__()

#         self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
#                                             embedding_dim=embed_dim,
#                                             padding_idx=char_to_id['<pad>'])

#         # 使用 RNN 層代替 LSTM 層
#         self.rnn_layer1 = torch.nn.RNN(input_size=embed_dim,
#                                         hidden_size=hidden_dim,
#                                         batch_first=True)

#         self.rnn_layer2 = torch.nn.RNN(input_size=hidden_dim,
#                                         hidden_size=hidden_dim,
#                                         batch_first=True)

#         self.linear = torch.nn.Sequential(torch.nn.Linear(in_features=hidden_dim,
#                                                           out_features=hidden_dim),
#                                           torch.nn.ReLU(),
#                                           torch.nn.Linear(in_features=hidden_dim,
#                                                           out_features=vocab_size))

#     def forward(self, batch_x, batch_x_lens):
#       return self.encoder(batch_x, batch_x_lens)

#     # The forward pass of the model
#     def encoder(self, batch_x, batch_x_lens):
#         batch_x = self.embedding(batch_x)

#         batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch_x,
#                                                           batch_x_lens,
#                                                           batch_first=True,
#                                                           enforce_sorted=False)

#         batch_x, _ = self.rnn_layer1(batch_x)
#         batch_x, _ = self.rnn_layer2(batch_x)

#         batch_x, _ = torch.nn.utils.rnn.pad_packed_sequence(batch_x,
#                                                             batch_first=True)

#         batch_x = self.linear(batch_x)

#         return batch_x

#     def generator(self, start_char, max_len=200):
#         char_list = [char_to_id[c] for c in start_char]
#         next_char = None

#         while len(char_list) < max_len:
#             # Pack the char_list to tensor
#             input_tensor = torch.tensor(char_list, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension
#             input_tensor = input_tensor.to(device)  # Move to the appropriate device (GPU or CPU)

#             logits = self.encoder(input_tensor, torch.tensor([len(char_list)]))
#             # Get the prediction for the last time step
#             y = logits[0, -1, :] # Use the output from the last time step

#             # Obtain the next token prediction y
#             next_char = torch.argmax(y, dim=0).item()  # Get the index of the max log-probability

#             if next_char == char_to_id['<eos>']:
#                 break

#             char_list.append(next_char)
#             print(char_list)

#         return [id_to_char[ch_id] for ch_id in char_list]

torch.manual_seed(2)


device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# Write your code here. Specify a device (cuda or cpu)

# print(device)
model = CharRNN(vocab_size,
                embed_dim,
                hidden_dim).to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=char_to_id['<pad>'])# Write your code here. Cross-entropy loss function. The loss function should ignore <pad>
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)# Write your code here. Use Adam or AdamW for Optimizer

# 因為 loss 降不下來˙ 調整 learning rate
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每 5 epochs 降低學習率

"""# Training
1. The outer `for` loop controls the `epoch`
    1. The inner `for` loop uses `data_loader` to retrieve batches.
        1. Pass the batch to the `model` for training.
        2. Compare the predicted results `batch_pred_y` with the true labels `batch_y` using Cross Entropy to calculate the loss `loss`
        3. Use `loss.backward` to automatically compute the gradients.
        4. Use `torch.nn.utils.clip_grad_value_` to limit the gradient values between `-grad_clip` &lt; and &lt; `grad_clip`.
        5. Use `optimizer.step()` to update the model (backpropagation).
2.  After every `1000` batches, output the current loss to monitor whether it is converging.
"""

from tqdm import tqdm
from copy import deepcopy


for epoch in range(1, epochs+1):
    # The process bar
    model.train()
    bar = tqdm(dl_train, desc=f"Train epoch {epoch}")
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        # Write your code here
        # Clear the gradient
        optimizer.zero_grad()

        batch_pred_y = model(batch_x.to(device), batch_x_lens)

        # Write your code here
        # Reshape `batch_y` to match the shape required by the loss function
        batch_y = batch_y.to(device).view(-1)
        batch_pred_y = batch_pred_y.view(-1, vocab_size)


        # Input the prediction and ground truths to loss function
        loss = criterion(batch_pred_y, batch_y)
        # Back propagation
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip) # gradient clipping

        # Write your code here
        # Optimize parameters in the model
        optimizer.step()

        i+=1
        if i%50==0:
            bar.set_postfix(loss = loss.item())
        # bar.set_postfix(loss = loss.item())
    # scheduler.step()  # 更新學習率

    # Evaluate your model
    model.eval()
    bar = tqdm(dl_eval, desc=f"Validation epoch {epoch}")
    matched = 0
    total = 0
    for batch_x, batch_y, batch_x_lens, batch_y_lens in bar:
        # 生成 predictions
        predictions = model(batch_x.to(device), batch_x_lens)

        # Write your code here.
        # Check whether the prediction match the ground truths
        # Compute exact match (EM) on the eval dataset
        # EM = correct/total

        # 用 argmax 取得預測的值
        pred_tokens = predictions.argmax(dim=-1)

        # 遍歷每個序列來找等號 `=` 與 `<eos>` 之間的accuracy
        for i in range(batch_y.size(0)):

            # print(f"pred_seq = {pred_tokens[i]}")
            # print(f"y_seq  ={batch_y[i]}")

            y_seq = batch_y[i].to(device)
            pred_seq = pred_tokens[i].to(device)


            # 找到 y_seq 中首個不為 0 的位置
            first_nonzero_idx = (y_seq != 0).nonzero(as_tuple=True)[0]

            # 找到 y_seq 中首個出現 1 的位置
            first_one_idx = (y_seq == 1).nonzero(as_tuple=True)[0]

            # 轉換為標量
            if first_nonzero_idx.numel() > 0:
                y_start_idx = first_nonzero_idx[0].item()  # 第一個不為 0 的idx
            else:
                y_start_idx = None  # 如果沒有非零元素，設為 None

            if first_one_idx.numel() > 0:
                y_end_idx = first_one_idx[0].item()  # 第一個出現 1 的idx
            else:
                y_end_idx = None  # 如果沒有出現 1，設為 None

            # 提取預測與目標的對應區段
            pred_seq = pred_seq[y_start_idx:y_end_idx+1]
            y_seq = y_seq[y_start_idx:y_end_idx+1]

            # 檢查整段是否完全match
            if torch.equal(pred_seq, y_seq):
                matched += 1

            total += 1

    # 計算accuracy
    accuracy = matched / total
    print(f"Accuracy: {accuracy:.4f}")

# y_seq [0,0,0,0,17,2,2,4,1,0,0,0]
# pred_seq [6,5,4,3,17,2,2,4,3,4,1,0]

# torch.save(model, "model_full.pth")

"""# Generation
Use `model.generator` and provide an initial character to automatically generate a sequence.
"""

model = model.to("cuda")  # 確保模型在 GPU 上
print("".join(model.generator('1+1=')))  # 使用 generator 生成

# model = model.to("cuda")  
# print("".join(model.generator('a+a='))) 

# model = model.to("cuda") 
# print("".join(model.generator('103S0303030303030303030+1303030303030303030303='))) 