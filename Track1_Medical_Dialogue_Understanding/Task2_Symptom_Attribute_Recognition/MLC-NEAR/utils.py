import torch
from torch.utils.data import Dataset
import transformers
import json


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]

        inputs = self.tokenizer.encode_plus(
            x,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        weights = self.data[index][3]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(y, dtype=torch.float),
            'weights': torch.tensor(weights, dtype=torch.float)}


class BERTClass(torch.nn.Module):
    def __init__(self, pretrained_pth):
        super(BERTClass, self).__init__()
        self.pth = pretrained_pth
        self.H = 768
        self.add_blstm = False
        self.token_level = True

        self.flat_att = UnflatSelfAttention(self.H, 0.3)

        self.l1 = transformers.BertModel.from_pretrained(self.pth)  # ('/data1/zpl/jddc/nlp_baai/pretrained_models_and_embeddings/ERNIE_stable-1.0.1-pytorch')  #('/data1/zpl/jddc/nlp_baai/pretrained_models_and_embeddings/bert/bert-base-chinese')    #('/data1/cls/MedicalChatbox/mc_bert_base_pytorch')    # ('/data1/MedicalChatbox/PCL-MedBERT/MTBERT')        #('bert-base-chinese')
        self.l2 = torch.nn.Dropout(p=0.3)

        if self.add_blstm:
            self.rnn_dim = 256
            self.rnn = torch.nn.LSTM(self.H, self.rnn_dim, num_layers=1, bidirectional = True, batch_first= True)
            self.H = 2*self.rnn_dim

        self.l3 = torch.nn.Linear(self.H, 987)

    def forward(self, ids, mask, token_type_ids):
        sequence_output, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        
        # input(f'pretrained out 1 : {sequence_output.size()}    out2: {output_1.size()}')

        if self.add_blstm:
            sequence_output,  (h_n, c_n) = self.rnn(sequence_output)
            input(f'rnn out 1 : {sequence_output.size()}    out2: {h_n[0].size()}')
            sequence_output = torch.mean(sequence_output, dim=1)     # should try torch.mean(sequence_output, dim=1)

            sequence_output = self.l2(sequence_output)
            output = self.l3(sequence_output)

            # # last hidden
            # h_n = torch.cat((h_n[0], h_n[1]), dim=1)
            # h_n = self.l2(h_n)
            # output = self.l3(h_n)
        elif self.token_level:
            output_token = self.l2(sequence_output)     # drop out

            # output_token = sequence_output      # add att
            # output_token = self.flat_att(output_token, output_token.size()[1])      # add att

            output_token = self.l3(output_token)
            output = torch.sum(output_token, dim = 1)
            # input(f'token level output: {output_token.size()}       output: {output.size()}')
        else:
            output_2 = self.l2(output_1)
            output = self.l3(output_2)

        # input(output.size())
        
        return output

class UnflatSelfAttention(torch.nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """
    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = torch.nn.Linear(d_hid, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)
        # max_len = max(lens)
        # for i, l in enumerate(lens):
        #     if l < max_len:
        #         scores.data[i, l:] = -np.inf
        scores = torch.nn.functional.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp)
        # input(f'context: {context.size()}')
        return context


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(data, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        if isinstance(data, list):
            print('writing {} records to {}'.format(len(data), path))