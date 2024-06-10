import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.nn import functional as nnf
from transformers import T5ForConditionalGeneration, AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorForSeq2Seq
import argparse
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle , json, time , sys
from torch.nn.utils.rnn import pad_sequence
from cider import readJSON, readPickle, getGTCaptions, BLEUScore, CIDERScore
from config import CONFIG
from torch import Tensor
import torch.nn.functional as F
from typing import Optional

# STAGCN
from net.st_agcn import STA_GCN as st_agcn
from net.Utils_attention.attention_branch import *
from net.Utils_attention.perception_branch import *
from net.Utils_attention.feature_extractor import *
from net.Utils_attention.graph_convolution import *

# alignment
from alignment.alignment import *

# dim convertor
from convertor.dim_convertor import dim_conv as dim_conv

# T5 visualize tool
from transformers import utils
from visualize_model import model_view, head_view, neuron_view
from utils import time_elapsed

from bert_score import score
from nlgmetricverse import NLGMetricverse,load_metric

bonelink = [(0, 1), (0, 2), (0, 3), (1, 4), (2,5), (3,6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18,  20), (19, 21)]


def generate_data(current_keypoints ):
    joint_coordinate = np.zeros((3,len(current_keypoints),22))
    bone_coordinate = np.zeros((3,len(current_keypoints),22))

    for i in range(len(current_keypoints)):
        for j in range(0,len(current_keypoints[i]),3):
            joint_coordinate[:, i, j//3] = current_keypoints[i,j:j+3]

    for v1, v2 in bonelink:
        bone_coordinate[:, :, v1] = joint_coordinate[:, :, v1] - joint_coordinate[:, :, v2]

    coordinates = np.concatenate((joint_coordinate, bone_coordinate), axis=0)
    return coordinates

'''
# Standard joints for alignment
# For the Pretrained Model:
# The standard joints are those of a motionless person, compared to the HumanML3D Dataset.
# For the Fine-tuning Model:
# The standard joints are those of a professional athlete performing an Axel movement.
'''
standard = None

class HumanMLDataset(Dataset):
    def __init__(self, pkl_file, tokenizer, finetune,transform=None):
        with open(pkl_file, 'rb') as f:
            self.data_list = pickle.load(f)
        self.samples = []

        global standard
        for item in self.data_list:
            features =  generate_data(item['features'])
            video_name = item['video_name']

            if(video_name == 'standard'):
                standard = torch.FloatTensor(features)
                standard = standard.unsqueeze(0)

            else:
                for label in item['labels']:
                    if(finetune == False) : 
                        label = "Motion Description : " + label
                    else :
                        label = "Motion Instruction : " + label
                    self.samples.append((features, label, video_name))

        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features, label, video_name = self.samples[idx]     
        current_len = torch.tensor(len(features[0]))
        keypoints_mask = torch.ones(22)  
        tokenized_label = self.tokenizer(label, return_tensors="pt", padding="max_length", truncation=True, max_length=50)
       
        return video_name, \
                torch.FloatTensor(features), \
                torch.FloatTensor(keypoints_mask), \
                tokenized_label['input_ids'].squeeze(0), \
                current_len

class HumanMLDataset_val(Dataset):
    def __init__(self, pkl_file, tokenizer, transform=None):
        with open(pkl_file, 'rb') as f:
            self.data_list = pickle.load(f)
        self.tokenizer = tokenizer
        self.transform = transform
        global standard
        for item in self.data_list:
            features =  generate_data(item['features'])
            video_name = item['video_name']

            if(video_name == 'standard'):
                standard = torch.FloatTensor(features)
                standard = standard.unsqueeze(0)
                break

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data_list[idx]
        features, video_name ,label= item['features'], item['video_name'], item['labels']
        features = generate_data(features)  
        current_len = torch.tensor(len(features[0]))
        #keypoints_mask = torch.ones(current_len)
        keypoints_mask = torch.ones(22)
        tokenized_label = self.tokenizer(label, return_tensors="pt", padding="max_length", truncation=True, max_length=50)
        
        return video_name, \
                torch.FloatTensor(features), \
                torch.FloatTensor(keypoints_mask), \
                tokenized_label['input_ids'].squeeze(0), \
                current_len


class SimpleT5Model(nn.Module):

    def __init__(self):
        super(SimpleT5Model, self).__init__()
        config = AutoConfig.from_pretrained('t5-base')
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base', config=config)
        self.out_channel = CONFIG.OUT_CHANNEL 

        self.STAGCN  = st_agcn(num_class=1024, 
                                in_channels=6, 
                                residual=True, 
                                dropout=0.5, 
                                num_person=1, 
                                t_kernel_size=9,
                                layout='SMPL',
                                strategy='spatial',
                                hop_size=3,num_att_A=4 )
        
        self.dim_conv = dim_conv(alignment = True)
    def collate_fn(batch):
        video_name, keypoints, keypoints_mask, label,seq_len= zip(*batch)
        def collect_video_from_batch(batch):
            seq = []
            for b in batch:
                seq.append(b[1].permute(1,0,2)) # convert to F , coordinates, nodes 
            return seq
        keypoints = collect_video_from_batch(batch)

        padded_sequences = pad_sequence(keypoints,batch_first=True,padding_value=0) # B , F , coordinates, nodes
        padded_sequences = padded_sequences.permute(0,2,1,3) 

        keypoints_mask = pad_sequence(keypoints_mask,batch_first=True,padding_value=0) 
        seq_len = torch.stack(seq_len,dim=0)
        label = torch.stack(label,dim=0)
        return (video_name),padded_sequences,keypoints_mask,  label,seq_len
    
    # @time_elapsed
    def _get_stagcn_feature(self, src):
        embedding, attention_node, attention_matrix  = self.STAGCN(src)
        return embedding, attention_node, attention_matrix  

    '''
    # - _get_alignment_feature()
    # @ query : (1, T, 22, 512) standard
    # @ key : (batch_size, T, 22, 512) 
    # @ return : batchsize, vertex(22), seq_length, channel (512)
    '''
    # @time_elapsed
    def _get_alignment_feature(self, key,query,seq_len,finetune = False): # key : standard, query : user
        max_len = max(seq_len)
        def interpolate_sequence(sequence):
            step = torch.div(max_len, sequence.size(0), rounding_mode='floor')
            new_sequence = torch.zeros(max_len, sequence.size(1))
            
            for i in range(sequence.size(0)-1):
                new_index = int(i * step)
                new_sequence[new_index, :] = sequence[i, :]
                if i < sequence.size(1) - 1:
                    for j in range(1, int(step)):
                        ratio = j / step
                        interpolated_vector = sequence[i, :] + ratio * (sequence[i+1, :] - sequence[i, :])
                        new_sequence[new_index + j, :] = interpolated_vector
            return new_sequence
            
        avg_query = nn.AvgPool2d((22,1))(query).squeeze(2) ## b , T , 512
        avg_key = nn.AvgPool2d((22,1))(key).squeeze(2)     ## 1 , T , 512
        for i in range(len(query)):
            video_alignment = None
            current_len = seq_len[i]
            if finetune == True :
                start_frame = align(avg_query[i,:current_len,:], avg_key[0] )
            else :
                start_frame = align(avg_query[i,:current_len,:], avg_key[i] )

            for j in range(0,22):
                
                if finetune == True and (current_len > key[0].size(0)):
                    subtraction = query[i,:key[0].size(0),j,:] - key[0,start_frame:start_frame+key[0].size(0),j,:]
                elif finetune == False and (current_len > key[i].size(0)):
                    subtraction = query[i,:key[i].size(0),j,:] - key[i,start_frame:start_frame+key[i].size(0),j,:]
                elif finetune == True and (current_len <= key[0].size(0)):
                    subtraction = query[i,:current_len,j,:] - key[0,start_frame:start_frame+current_len,j,:] ## Tu x 22 x C
                elif finetune == False and (current_len <= key[i].size(0)): 
                    subtraction = query[i,:current_len,j,:] - key[i,start_frame:start_frame+current_len,j,:] 
                
                subtraction = interpolate_sequence(subtraction) ## seq_len x 22 x C
                subtraction = subtraction.unsqueeze(0) ## 1 x seq_len x 22 x C
                if j == 0: video_alignment = subtraction
                else : video_alignment = torch.cat([video_alignment,subtraction],dim=0)

                
            video_alignment = video_alignment.unsqueeze(0)
            if i == 0: batch_alignment = video_alignment
            else : batch_alignment = torch.cat([batch_alignment,video_alignment],dim=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        query = query.permute(0,2,1,3)
        batch_alignment = torch.cat([query,batch_alignment.to(device)],dim=-1)
        return batch_alignment
   
    '''
    # - _get_dim_convertor()
    # @ x : batchsize, vertex(22), seq_length, channel (512)
    # @ return : batchsize, vertex(22), 768
    '''
    # @time_elapsed
    def _get_dim_convertor(self, x):
        return self.dim_conv(x)  

    def forward(self, input_ids, attention_mask, seq_len,decoder_input_ids=None, labels=None,alignment = True,finetune = False):
        # STAGCN
        self.STAGCN.train()
   
        embeddings, attention_node, attention_matrix  = self._get_stagcn_feature(input_ids)
        
        if alignment == True:
            global standard
            # Use standard joints as input of STAGCN to generate standard embedding
            self.STAGCN.eval()
            with torch.no_grad():
                if finetune == False :
                    standard = input_ids
                    for i in range(0,input_ids.shape[0]):
                        for j in range(1,seq_len[i]):
                            for k in range(0,6):
                                standard[i][k][j] = input_ids[i][k][0]
              
                standard_embedding, std_attention_node, std_attention_matrix  = self._get_stagcn_feature(standard.to('cuda'))
                
            # Alignment
            embeddings = self._get_alignment_feature(standard_embedding, embeddings,seq_len,finetune)
        # Dim Convertor
        input_embeds = self._get_dim_convertor(embeddings)
 
        output = self.t5(inputs_embeds=input_embeds, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)        
        return output

    def generate(self,**kwargs):
        input_ids = kwargs['input_ids']
        attention_mask = kwargs['attention_mask']
        decoder_input_ids = kwargs['decoder_input_ids']
        tokenizer = kwargs['tokenizer'] 
        seq_len = kwargs['sequence_length']
        alignment = kwargs['alignment']
        finetune = kwargs['fine_tune']

        # STAGCN
        embeddings, attention_node, attention_matrix  = self._get_stagcn_feature(input_ids)
        
        if alignment == True:
            # Use standard joints as input of STAGCN to generate standard embedding
            self.STAGCN.eval()
            global standard
            with torch.no_grad():
                if finetune == False :
                    standard = input_ids
                    for i in range(0,input_ids.shape[0]):
                        for j in range(1,seq_len[i]):
                            for k in range(0,6):
                                standard[i][k][j] = input_ids[i][k][0]
                    standard_embedding, _ , _ = self._get_stagcn_feature(standard)

                standard_embedding, std_attention_node, std_attention_matrix  = self._get_stagcn_feature(standard.to('cuda'))

            # Alignment
            embeddings = self._get_alignment_feature(standard_embedding, embeddings,seq_len,finetune = False)

        # Dim Convertor
        input_embeds = self._get_dim_convertor(embeddings)

        beam_size = 1
        generated_ids = self.t5.generate( inputs_embeds=input_embeds, 
                                          attention_mask=attention_mask, 
                                          decoder_input_ids=decoder_input_ids, 
                                          max_length=50,
                                          num_beams=beam_size, 
                                          repetition_penalty=3.5,
                                          length_penalty=1.0,
                                          # temperature=1.5,
                                          return_dict_in_generate=True,
                                          output_attentions=True,   
                                          # Set do_sample=True if you want to demo
                                          do_sample=False,           
                                          early_stopping=True)
        if(kwargs['fine_tune'] == True) :

            decoded_text = tokenizer.convert_ids_to_tokens(generated_ids.sequences[0])
            out = self.t5(inputs_embeds=input_embeds, attention_mask=attention_mask, decoder_input_ids=generated_ids.sequences, output_attentions=True, return_dict=True)
            
            encoder_attentions = out.encoder_attentions
            cross_attentions = out.cross_attentions
            decoder_attentions = out.decoder_attentions
            '''
            encoder_attentions = encoder_attentions.resahpe(out.encoder_attentions.shape[0], out.encoder_attentions.shape[1], -1,22, out.encoder_attentions.shape[3])
            encoder_attentions = encoder_attentions.mean(dim=2)
                
            cross_attentions = cross_attentions.resahpe(out.cross_attentions.shape[0], out.cross_attentions.shape[1], -1,22, out.cross_attentions.shape[3])
            cross_attentions = cross_attentions.mean(dim=2)
            '''
            html_object = model_view(
                                        encoder_attention=encoder_attentions,
                                        decoder_attention=decoder_attentions,
                                        cross_attention=cross_attentions,
                                        encoder_tokens= len(input_embeds[0]),
                                        decoder_tokens=decoded_text,
                                        html_action='return'
                                        )

            html_object_head = head_view(
                                        encoder_attention=encoder_attentions,
                                        decoder_attention=decoder_attentions,
                                        cross_attention=cross_attentions,
                                        encoder_tokens= len(input_embeds[0]),
                                        decoder_tokens=decoded_text,
                                        html_action='return'
                                        )
            

            if not os.path.exists(kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch'])):
                os.makedirs(kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch']))

            # @name : kwargs['name'][0] since its batch size is one in inference dataset 
            with open(kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch']) + "/"+ kwargs['name'][0] + "_model_view.html", 'w') as file:
                file.write(html_object.data)
            with open(kwargs['result_dir'] + "/HTML/epoch" + str(kwargs['epoch']) + "/"+ kwargs['name'][0] + "_head_view.html", 'w') as file:
                file.write(html_object_head.data)

        return generated_ids.sequences , attention_node , attention_matrix

def train(train_dataset, model, tokenizer, args, eval_dataset=None, lr=1e-3, warmup_steps=5000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.bs
    epochs = args.epochs

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=SimpleT5Model.collate_fn)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        ################ Train #########################################
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=args.prefix)
        loss_list = []
        for idx, (video_name,src_batch,keypoints_mask_batch,tgt_batch,seq_len) in enumerate(train_dataloader):
            model.zero_grad()
            
            src_batch = src_batch.to(device)
            keypoints_mask_batch = keypoints_mask_batch.to(device)
            seq_len = seq_len.to(device)

            tgt_input = tgt_batch[:, :-1].to(device)
            tgt_labels = tgt_batch[:, 1:].to(device)
            outputs = model(input_ids=src_batch.contiguous(), 
                            attention_mask=keypoints_mask_batch.contiguous(), 
                            seq_len=seq_len.contiguous(),
                            decoder_input_ids=tgt_input.contiguous(),         # text
                            labels=tgt_labels.contiguous(),
                            alignment = args.alignment,
                            finetune = args.finetune)
                                        
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())

            progress.set_postfix({
                'loss': np.mean(loss_list),
                'lr': scheduler.optimizer.param_groups[0]['lr'],
            })
            progress.update()
        ################ Eval #########################################
        if eval_dataset is not None:
            print(f"Epoch {epoch}: Train Loss: {np.mean(loss_list):.4f}")
            torch.save(model.state_dict(), os.path.join(args.out_dir , f"{args.prefix}_epoch{epoch}.pt"))            
            model.eval()
            # @ batch_size = 1 :
            # This is because we need to visualize the attention information of every video.
            val_data_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=SimpleT5Model.collate_fn)
            results = {}
            att_node_results = {}
            att_A_results = {}    
            with torch.no_grad():
                for (video_names,src_batch,keypoints_mask_batch,tgt_batch,seq_len)  in tqdm(val_data_loader):
                    src_batch = src_batch.to(device)
                    keypoints_mask_batch = keypoints_mask_batch.to(device)
                    seq_len = seq_len.to(device)

                    tgt_input = tgt_batch[:, :-1].to(device)
                    tgt_labels = tgt_batch[:, 1:].to(device)
                    

                    if args.finetune == True :  
                        decoder_input_ids = tokenizer(["Motion Instruction : "],
                                                  return_tensors="pt", 
                                                  padding=True, 
                                                  truncation=True, 
                                                  add_special_tokens=False)['input_ids']
                    else :
                        decoder_input_ids = tokenizer(["Motion Description : "],
                                                  return_tensors="pt", 
                                                  padding=True, 
                                                  truncation=True, 
                                                  add_special_tokens=False)['input_ids']
  
                    decoder_input_ids = decoder_input_ids.repeat(src_batch.shape[0], 1).to(device)
                    
                    input = {   "input_ids": src_batch.contiguous(),
                                "attention_mask": keypoints_mask_batch.contiguous(),
                                "decoder_input_ids": decoder_input_ids,
                                "sequence_length": seq_len.contiguous(),
                                "name": video_names,
                                "tokenizer": tokenizer,
                                "fine_tune": args.finetune,
                                "result_dir": args.result_dir,
                                "epoch": epoch,
                                "alignment": args.alignment}

                    generated_ids , att_node , att_A = model.generate(**input)
                   
                    
                    for name, gen_id in zip(video_names, generated_ids):
                        decoded_text = tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        results[name] = decoded_text
                    for name, att_node in zip(video_names, att_node):
                        att_node_results[name] = att_node.cpu().numpy().tolist()
                    for name, att_A in zip(video_names, att_A):
                        att_A_results[name] = att_A.cpu().numpy().tolist()
                    
            if not os.path.exists(args.result_dir):
                os.makedirs(args.result_dir)
            with open(args.result_dir+'/results_epoch'+str(epoch)+'.json', 'w') as f:
                json.dump(results, f,indent = 1)
            with open(args.result_dir+'/att_node_results_epoch'+str(epoch)+'.json', 'w') as f:
                json.dump(att_node_results, f)
            with open(args.result_dir+'/att_A_results_epoch'+str(epoch)+'.json', 'w') as f:
                json.dump(att_A_results, f)

            predictions = readJSON(args.result_dir+'/results_epoch'+str(epoch)+'.json')
            annotations = readPickle(args.test_data)
           
            gts = getGTCaptions(annotations)
            # Check predictions content is correct
            assert type(predictions) is dict
            assert set(predictions.keys()) == set(gts.keys())
            assert all([type(pred) is str for pred in predictions.values()])
            # CIDErScore
            cider_score = CIDERScore()(predictions, gts)
            bleu_score = BLEUScore()(predictions, gts)
            print(f"CIDEr: {cider_score}")
            print(f"BLEU: {bleu_score}")
        
        progress.close()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local',      type=bool,default = True)
    parser.add_argument('--finetune',   type=bool,default=CONFIG.Finetune)
    parser.add_argument('--data',       default=CONFIG.data)
    parser.add_argument('--out_dir',    default=CONFIG.out_dir)
    parser.add_argument('--prefix',     default=CONFIG.prefix, help='prefix for saved filenames')
    parser.add_argument('--epochs',     type=int, default=100)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--bs',         type=int, default=8)
    parser.add_argument('--pretrained', type=bool,default=CONFIG.Pretrained)
    parser.add_argument('--test_data',  default=CONFIG.test_data)
    parser.add_argument('--result_dir',  default = CONFIG.result_dir)
    parser.add_argument('--alignment',  type=bool,default=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)
    
    model = SimpleT5Model()
    if(CONFIG.Pretrained):
        weight           = CONFIG.weight_path
        model_state_dict = model.state_dict()
        state_dict = torch.load(weight)
        pretrained_dict_1 = {k: v for k, v in state_dict.items() if k in model_state_dict}
        model_state_dict.update(pretrained_dict_1)
        model.load_state_dict(model_state_dict)
        
    dataset = HumanMLDataset(args.data, tokenizer,args.finetune)
    eval_dataset = HumanMLDataset_val(args.test_data, tokenizer) 
    train(dataset, model, tokenizer, args,eval_dataset=eval_dataset)
    
if __name__ == '__main__':
    main()