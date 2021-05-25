import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import torch
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from utils import CustomDataset, BERTClass, load_json


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#input(f'cuda: {torch.cuda.is_available()}')

torch.manual_seed(12345)

# load train/dev set
prefix = './data'
model_prefix = './saved_ernie/token'   #  './saved' 

os.makedirs(model_prefix, exist_ok=True)

train = load_json(os.path.join(prefix, 'processed', 'train_set.json'))
dev = load_json(os.path.join(prefix, 'processed', 'dev_set.json'))

# Defining some key variables that will be used later on in the training
MAX_LEN = 128
TRAIN_BATCH_SIZE = 24 #32
VALID_BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 1e-05

# pretrained model link
pclbert = ('/data1/zpl/jddc/nlp_baai/pretrained_models_and_embeddings/bert/MTBERT')  
ernie = ('/data1/zpl/jddc/nlp_baai/pretrained_models_and_embeddings/ERNIE_stable-1.0.1-pytorch')  
bert_base_chinese = ('/data1/zpl/jddc/nlp_baai/pretrained_models_and_embeddings/bert/bert-base-chinese')    
mc_bert = ('/data1/cls/MedicalChatbox/mc_bert_base_pytorch')    

model_pth = ernie

tokenizer = BertTokenizer.from_pretrained(model_pth)  		#('bert-base-chinese')

train_set = CustomDataset(train, tokenizer, MAX_LEN)
dev_set = CustomDataset(dev, tokenizer, MAX_LEN)


train_params = {
    'batch_size': TRAIN_BATCH_SIZE,
    # 'shuffle': True,
    'num_workers': 1
}

dev_params = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 1
}

weights = [sample[3] for sample in train]
print('weights: max = {}, min = {}, mean = {}'.format(np.max(weights), np.min(weights), np.mean(weights)))
sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True)

train_loader = DataLoader(train_set, sampler=sampler, **train_params)
dev_loader = DataLoader(dev_set, **dev_params)




model = BERTClass(model_pth)
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def loss_fn(outputs, targets, weight):
    loss_hard = torch.nn.BCEWithLogitsLoss()(outputs, targets)


    # ''' add label num '''
    # count_1 = torch.nonzero(targets, out=None)
    # num_targets = torch.zeros(targets.size()[0], 1).to(device, dtype=torch.float)
    # for b, idx in count_1:
    #     num_targets[b] += torch.ones(1).to(device, dtype=torch.float)

    # thresh_mat = outputs > 0.5
    # thresh_out = outputs * thresh_mat
    # count_nonzero = torch.nonzero(thresh_out, out=None)
    # num_out = torch.zeros(targets.size()[0], 1).to(device, dtype=torch.float)
    # for b, idx in count_nonzero:
    #     num_out[b] += torch.ones(1).to(device, dtype=torch.float)

    # num_criteria = torch.nn.MSELoss()
    # loss_num = num_criteria(num_out, num_targets)

    # total_loss = loss_hard + loss_num

    # return total_loss, loss_hard, loss_num

    # ''' add label num done'''

    total_loss = loss_hard 

    return total_loss, loss_hard , loss_hard 


def train_epoch(_epoch):
    model.train()
    for _, data in tqdm(enumerate(train_loader)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        weight = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)
        outputs2 = model(ids, mask, token_type_ids)
        # print(f'1: {outputs[:50]}\n')
        # input(f'2: {outputs2[:50]}')



        optimizer.zero_grad()
        loss, loss_hard, loss_num = loss_fn(outputs, targets, weight)
        if _ % 100 == 0:
            print(f'Epoch: {_epoch + 1}, Loss:  {loss.item()}       Loss_hard:  {loss_hard.item()}     Loss_num:  {loss_num.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model, os.path.join(model_prefix, 'model_{}.pkl'.format(_epoch + 1)))


def validate():
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(dev_loader), total=len(dev_set) // VALID_BATCH_SIZE + 1):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    fin_outputs = np.array(fin_outputs) >= 0.5		# Threshold=0.5
    accuracy = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_score_micro = metrics.f1_score(fin_targets, fin_outputs, average='micro')
    f1_score_macro = metrics.f1_score(fin_targets, fin_outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

    return accuracy, f1_score_micro, f1_score_macro


max_acc_epoch = 0
max_micro_epoch = 0
max_macro_epoch = 0

max_acc = 0
max_micro = 0
max_macro = 0

for epoch in range(EPOCHS):
    train_epoch(epoch)
    accuracy, micro_f1, macro_f1 = validate()

    if  accuracy > max_acc:
        max_acc_epoch = epoch+1
        max_acc = accuracy

    if  micro_f1 > max_micro:
        max_micro_epoch = epoch+1
        max_micro = micro_f1

    if  macro_f1 > max_macro:
        max_macro_epoch = epoch+1
        max_macro = macro_f1



input(f'\n\nmax_acc_epoch: {max_acc_epoch}  max_micro_epoch: {max_micro_epoch}  max_macro_epoch: {max_macro_epoch}')

# load model
best_epoch = max_micro_epoch
model = torch.load(os.path.join(model_prefix, 'model_{}.pkl'.format(best_epoch)))
# load model end
accuracy, micro_f1, macro_f1 = validate()

input('end')