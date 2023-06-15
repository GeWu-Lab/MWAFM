import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import copy
import math
from nets.multi_attention import MultiScaleSelfAttention


class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):

        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        self.lstm.flatten_parameters()
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_a, mask=None, src_key_padding_mask=None):
        output_a = src_a

        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_a, src_mask=mask,src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output_a = self.norm1(output_a)

        return output_a



class MultiAttnLayer(nn.Module):

    # d_model=512, nhead=1, dim_feedforward=512), num_layers=1
    def __init__(self, d_model, nhead, window_size, dim_feedforward=512, dropout=0.1):
        super(MultiAttnLayer, self).__init__()


        self.self_attn = MultiScaleSelfAttention(num_attention_heads = nhead, 
                                                 hidden_size = d_model,
                                                 attention_probs_dropout_prob = 0.0,
                                                 attention_window = [window_size],
                                                 attention_dilation = [1],
                                                 attention_mode = 'sliding_chunks',
                                                 autoregressive = False, 
                                                 layer_id=0)

        self.cm_attn = MultiScaleSelfAttention(num_attention_heads = nhead,   
                                                 hidden_size = d_model,
                                                 attention_probs_dropout_prob = 0.0,
                                                 attention_window = [window_size],
                                                 attention_dilation = [1],
                                                 attention_mode = 'sliding_chunks',
                                                 autoregressive = False, 
                                                 layer_id=0)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        

    def forward(self, src_q, src_kv, src_mask=None, src_key_padding_mask=None):

        src_lf_self = self.self_attn(src_q, src_q, src_q)[0]

        src_q = src_q + self.dropout12(src_lf_self)
        src_q = self.norm1(src_q)

        # src_lf_self = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        # src_q = src_q + self.dropout2(src_lf_self)
        # src_q = self.norm2(src_q)

        return src_q



class Encoder_QA(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder_QA, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_a, mask=None, src_key_padding_mask=None):
        output_a = src_a

        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_mask=mask,src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output_a = self.norm1(output_a)

        return output_a


class QAHanLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(QAHanLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_a, src_mask=None, src_key_padding_mask=None):

        src_a = src_a.permute(1, 0, 2)
        src2 = self.self_attn(src_a, src_a, src_a, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src_a = src_a + self.dropout12(src2)
        src_a = self.norm1(src_a)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_a))))
        src_a = src_a + self.dropout2(src2)
        src_a = self.norm2(src_a)

        return src_a.permute(1, 0, 2)


class MWAFM_Net(nn.Module):

    def __init__(self, d_model=512, nhead=1, dropout=0.1, dim_feedforward=512):
        super(MWAFM_Net, self).__init__()


        # self.audio_ast_fc = nn.Linear(768, 512)
        # self.fusion_ast_fc = nn.Linear(1024, 512)
 
        self.audio_fc =  nn.Linear(128, 512)

        self.question_fc = nn.Linear(300, 512)
        self.question_fc2 = nn.Linear(512, 512)

        self.question_encoder = QstEncoder(2000, 512, 512, 1, 512)
        self.word2vec = nn.Embedding(2000, 512)

        self.multi_scale_encoder_2 = Encoder(MultiAttnLayer(d_model=512, nhead=4, window_size=2, dim_feedforward=512), num_layers=1)
        self.multi_scale_encoder_4 = Encoder(MultiAttnLayer(d_model=512, nhead=4, window_size=4, dim_feedforward=512), num_layers=1)
        self.multi_scale_encoder_6 = Encoder(MultiAttnLayer(d_model=512, nhead=4, window_size=6, dim_feedforward=512), num_layers=1)
        self.multi_scale_encoder_12 = Encoder(MultiAttnLayer(d_model=512, nhead=4, window_size=12, dim_feedforward=512), num_layers=1)

        self.multi_scale_linear = nn.Linear(512, 512)
        self.multi_scale_dropout = nn.Dropout(0.1)
        self.multi_scale_norm = nn.LayerNorm(512)

        # question as query on audio and visual_feat_grd
        self.attn_qst_query = nn.MultiheadAttention(512, 4, dropout=0.1)
        self.qst_query_linear1 = nn.Linear(512, 512)
        self.qst_query_relu = nn.ReLU()
        self.qst_query_dropout1 = nn.Dropout(0.1)
        self.qst_query_linear2 = nn.Linear(512, 512)
        self.qst_query_dropout2 = nn.Dropout(0.1)
        self.qst_query_norm = nn.LayerNorm(512)

        # self-cross
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.tanh = nn.Tanh()

        
        self.combine_fc1 = nn.Linear(1024, 512)
        self.combine_fc2 = nn.Linear(512, 256)
        self.pred_fc = nn.Linear(256, 828)



        self.multi_layers = Encoder_QA(QAHanLayer(d_model=512, 
                                                    nhead=1, 
                                                    dim_feedforward=512), 
                                                    num_layers=4)

    

    ### attention, question as query on visual_feat and audio_feat
    def SelfAttn(self, quests_feat_input, key_value_feat):
        
        ### input Q, K, V: [T, B, C]

        key_value_feat_grd = key_value_feat.permute(1, 0, 2)
        qst_feat_query = key_value_feat_grd
        key_value_feat_att = self.attn_qst_query(qst_feat_query, key_value_feat_grd, key_value_feat_grd, 
                                                 attn_mask=None, key_padding_mask=None)[0]
        src = self.qst_query_linear1(key_value_feat_att)
        src = self.qst_query_relu(src)
        src = self.qst_query_dropout1(src)
        src = self.qst_query_linear2(src)
        src = self.qst_query_dropout2(src)
        
        key_value_feat_att = key_value_feat_att + src
        key_value_feat_att = self.qst_query_norm(key_value_feat_att)

        return key_value_feat_att.permute(1, 0, 2)


    def SelfCrossAttn(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
        # src_q = src_q.unsqueeze(0)
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)


    ### attention, question as query on visual_feat and audio_feat
    def QuestionQuereidAttn(self, quests_feat_input, key_value_feat):

        # qst_feat_query = quests_feat_input.unsqueeze(0)          # [1, B, C], [1, 2, 512]
        qst_feat_query = quests_feat_input.permute(1, 0, 2)

        ### input Q, K, V: [T, B, C]
        key_value_feat_grd = key_value_feat.permute(1, 0, 2)
        key_value_feat_att = self.attn_qst_query(key_value_feat_grd, qst_feat_query, qst_feat_query,  
                                                 attn_mask=None, key_padding_mask=None)[0]
        src = self.qst_query_linear1(key_value_feat_att)
        src = self.qst_query_relu(src)
        src = self.qst_query_dropout1(src)
        src = self.qst_query_linear2(src)
        src = self.qst_query_dropout2(src)
        
        key_value_feat_att = key_value_feat_att + src
        key_value_feat_att = self.qst_query_norm(key_value_feat_att)

        return key_value_feat_att.permute(1, 0, 2)


    # def forward(self, audio, audio_ast_feat, question):
    def forward(self, audio, question):

        ### feature input
        audio_feat = self.audio_fc(audio)               # [B, T, C]
        qst_feat = self.question_fc(question)

        # audio_ast_feat = self.audio_ast_fc(audio_ast_feat)
        # audio_ast_feat = F.relu(audio_ast_feat)
        
        audio_feat_grd = audio_feat
        qst_feat_grd = qst_feat
    
        ### --------------- Hybrid Attention Module start --------------- 
        qst_feat = self.SelfAttn(qst_feat, qst_feat)
        audio_feat = self.SelfCrossAttn(audio_feat, qst_feat_grd)

        ### --------------- Multi-scale Window attention start --------------- 
        ## input: [B, T, C], output: [B, T, C]
        aud_feat_scale_2 = self.multi_scale_encoder_2(audio_feat, audio_feat)
        aud_feat_scale_4 = self.multi_scale_encoder_4(audio_feat, audio_feat)
        aud_feat_scale_6 = self.multi_scale_encoder_6(audio_feat, audio_feat)
        aud_feat_scale_12 = self.multi_scale_encoder_12(audio_feat, audio_feat)

        audio_feat_kv2 = aud_feat_scale_2.permute(1, 0, 2)
        audio_feat_kv4 = aud_feat_scale_4.permute(1, 0, 2)
        audio_feat_kv6 = aud_feat_scale_6.permute(1, 0, 2)
        audio_feat_kv12 = aud_feat_scale_12.permute(1, 0, 2)

        audio_feat_kv2 = self.multi_scale_dropout(F.relu(self.multi_scale_linear(audio_feat_kv2)))
        audio_feat_kv4 = self.multi_scale_dropout(F.relu(self.multi_scale_linear(audio_feat_kv4)))
        audio_feat_kv6 = self.multi_scale_dropout(F.relu(self.multi_scale_linear(audio_feat_kv6)))
        audio_feat_kv12 = self.multi_scale_dropout(F.relu(self.multi_scale_linear(audio_feat_kv12)))

        audio_feat_ws_sum = audio_feat_kv2 + audio_feat_kv4 + audio_feat_kv6 + audio_feat_kv12
        audio_feat_kv = audio_feat + audio_feat_ws_sum.permute(1, 0, 2)
        # audio_feat_kv = self.multi_scale_norm(audio_feat_kv)

        ### --------------- Multi-scale Window attention end --------------- 

        audio_feat_kv = self.multi_layers(audio_feat)
        audio_feat_kv = audio_feat_kv.mean(dim=1)

        # cat
        # audio_feat_kv = torch.cat([audio_ast_feat.mean(-2), audio_feat_kv], dim=-1)
        # audio_feat_kv = self.fusion_ast_fc(audio_feat_kv)
        # # audio_feat_kv = F.relu(audio_feat_kv)

        # add
        # audio_feat_kv = audio_feat_kv + audio_ast_feat.mean(-2)
        qst_feat = qst_feat.mean(dim=1)
        combine_feat = torch.mul(audio_feat_kv, qst_feat)
        
        combine_feat = F.relu(self.combine_fc2(combine_feat))
        feat_output = self.pred_fc(combine_feat)

        return feat_output

