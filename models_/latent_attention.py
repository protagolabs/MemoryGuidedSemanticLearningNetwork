import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from modules_.graph_convolution import GraphConvolution
from modules_.tanh_attention import TanhAttention
from modules_.dynamic_rnn import DynamicGRU
from modules_.cross_gate import CrossGate
from modules_.ConvGRU import ConvGRUCell
from modules_.Coattention import CoAttention, CoAttention_intra
from modules_.message import Message
from modules_.multihead_attention import MultiHeadAttention
from modules_.position import PositionalEncoding
from modules_.SLA import stack_latent_attention
from utils import generate_anchors


class VideoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.args = args
        if args.dataset == 'TACOS':
            self.tran = nn.Linear(4096, args.frame_dim)
        if args.dataset == 'Charades':
            self.tran = nn.Linear(1024, args.frame_dim)
        #else:
        #    self.tran = nn.Linear(args.frame_dim, args.frame_dim)
        self.max_num_frames = args.max_num_frames
        # self.attn_layers = nn.ModuleList([
        #     MultiHeadAttention(args.frame_dim, args.num_heads)
        #     for _ in range(args.num_attn_layers)
        # ])
        self.rnn = DynamicGRU(args.frame_dim, args.d_model >> 1, bidirectional=True, batch_first=False)
        # self.attn_width = 3
        # self.self_attn_mask = torch.empty(self.max_num_frames, self.max_num_frames) \
        #     .float().fill_(float(-1e10)).cuda()
        # for i in range(0, self.max_num_frames):
        #     low = i - self.attn_width
        #     low = 0 if low < 0 else low
        #     high = i + self.attn_width + 1
        #     high = self.max_num_frames if high > self.max_num_frames else high
        #     # attn_mask[i, low:high] = 0
        #     self.self_attn_mask[i, low:high] = 0

    def forward(self, x, mask):
        if self.args.dataset in ['TACOS', 'Charades']:
            x = self.tran(x)
        #else:
        #    x = self.tran(x)
        x = x.transpose(0, 1)
        length = mask.sum(dim=-1)

        # for a in self.attn_layers:
        #     res = x
        #     x, _ = a(x, x, x, None, attn_mask=self.self_attn_mask)
        #     x = F.dropout(x, self.dropout, self.training)
        #     x = res + x

        x = self.rnn(x, length, self.max_num_frames)
        x = F.dropout(x, self.dropout, self.training)

        x = x.transpose(0, 1)
        return x


class SentenceEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.max_num_words = args.max_num_words
  
        self.rnn = DynamicGRU(args.word_dim, args.d_model >> 1, bidirectional=True, batch_first=True)

        # self.attn_layers = nn.ModuleList([
        #     MultiHeadAttention(args.word_dim, args.num_heads)
        #     for _ in range(args.num_attn_layers)
        # ])

        # self.attn_width = 3
        # self.self_attn_mask = torch.empty(self.max_num_words, self.max_num_words) \
        #     .float().fill_(float(-1e10)).cuda()
        # for i in range(0, self.max_num_words):
        #     low = i - self.attn_width
        #     low = 0 if low < 0 else low
        #     high = i + self.attn_width + 1
        #     high = self.max_num_words if high > self.max_num_words else high
        #     # attn_mask[i, low:high] = 0
        #     self.self_attn_mask[i, low:high] = 0

    def forward(self, x, mask, node_pos, node_mask, adj_mat):
        # x = x.transpose(0, 1)
        length = mask.sum(dim=-1)

        # for a in self.attn_layers:
        #     res = x
        #     x, _ = a(x, x, x, None, attn_mask=self.self_attn_mask)
        #     x = F.dropout(x, self.dropout, self.training)
        #     x = res + x

        x = self.rnn(x, length, self.max_num_words)
        x = F.dropout(x, self.dropout, self.training)
        # x = x.transpose(0, 1)
        return x


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.dropout = args.dropout
        self.max_num_frames = args.max_num_frames

        self.anchors = generate_anchors(dataset=args.dataset)
        self.num_anchors = self.anchors.shape[0]
        widths = (self.anchors[:, 1] - self.anchors[:, 0] + 1)  # [num_anchors]
        centers = np.arange(0, args.max_num_frames)  # [video_len]
        start = np.expand_dims(centers, 1) - 0.5 * (np.expand_dims(widths, 0) - 1)
        end = np.expand_dims(centers, 1) + 0.5 * (np.expand_dims(widths, 0) - 1)
        self.proposals = np.stack([start, end], -1)  # [video_len, num_anchors, 2]

        # VideoEncoder
        self.video_encoder = VideoEncoder(args)

        # SentenceEncoder
        self.sentence_encoder = SentenceEncoder(args)

        #SLA
        self.pos_v = PositionalEncoding(args.d_model, 0.2, 200)
        self.v_sla_1 = stack_latent_attention(args.d_model, True)
        self.v_sla_2 = stack_latent_attention(args.d_model)
        self.v_sla_3 = stack_latent_attention(args.d_model)
        self.v_l = nn.Linear(args.d_model*2, args.d_model, bias = False)

        self.pos_s = PositionalEncoding(args.d_model, 0.2, 20)
        self.s_sla_1 = stack_latent_attention(args.d_model, True)
        self.s_sla_2 = stack_latent_attention(args.d_model)
        self.s_sla_3 = stack_latent_attention(args.d_model)
        self.s_l = nn.Linear(args.d_model*2, args.d_model, bias = False)

        self.v2s = TanhAttention(args.d_model)
        # self.s2v = TanhAttention(args.d_model)
        #self.cross_gate = CrossGate(args.d_model)
        # self.fc = Bilinear(args.d_model, args.d_model, args.d_model)
        # 
        # self.s2v = nn.Conv1d(args.max_num_words, args.max_num_frames, kernel_size=1)
        
        self.rnn = DynamicGRU(args.d_model << 1, args.d_model >> 1, bidirectional=True, batch_first=True)

        self.fc_score = nn.Conv1d(args.d_model, self.num_anchors, kernel_size=1, padding=0, stride=1)
        self.fc_reg = nn.Conv1d(args.d_model, self.num_anchors << 1, kernel_size=1, padding=0, stride=1)

        # loss function
        self.criterion1 = nn.BCELoss()
        self.criterion2 = nn.SmoothL1Loss()

    def forward(self, frames, frame_mask, words, word_mask,
                label, label_mask, gt,
                node_pos, node_mask, adj_mat):
        frames_len = frame_mask.sum(dim=-1)

        frames = F.dropout(frames, self.dropout, self.training)
        words = F.dropout(words, self.dropout, self.training)

        frames = self.video_encoder(frames, frame_mask)
        x = self.sentence_encoder(words, word_mask, node_pos, node_mask, adj_mat)

        #SLA
        #1 layer
        v_text_0, v_vis_0 = x, frames

        h_text_0, h_vis_0 = frames, x
        z_text_0, z_vis_0 = self.pos_s(x), self.pos_v(frames)
        h_text_1, z_vis_1 = self.v_sla_1(v_vis_0, h_vis_0, z_vis_0, frame_mask, node_mask)
        h_vis_1, z_text_1 = self.s_sla_1(v_text_0, h_text_0, z_text_0, node_mask, frame_mask)
        
        #2 layer
        h_text_2, z_vis_2 = self.v_sla_2(v_vis_0, h_vis_1, z_vis_1, frame_mask, node_mask)
        h_vis_2, z_text_2 = self.s_sla_2(v_text_0, h_text_1, z_text_1, node_mask, frame_mask)
        
        #3 layer
        h_text_3, z_vis_3 = self.v_sla_2(v_vis_0, h_vis_2, z_vis_2, frame_mask, node_mask)
        h_vis_3, z_text_3 = self.s_sla_2(v_text_0, h_text_2, z_text_2, node_mask, frame_mask)

        frames1 = self.v_l(z_vis_3)
        x1 = self.s_l(z_text_3)
        #frames1, x1 = frames, x
        #a1, a2 = 1, 1
        # interactive
        # x1 = self.atten(frames1, x1, node_mask, last=True)
        x1 = self.v2s(frames1, x1, node_mask)
        #frames1, x1 = self.cross_gate(frames1, x1)
        # x1 = self.s2v(x1)
        x = torch.cat([frames1, x1], -1) #x1
        x = self.rnn(x, frames_len, self.max_num_frames)
        x = F.dropout(x, self.dropout, self.training)

        # loss
        predict = torch.sigmoid(self.fc_score(x.transpose(-1, -2))).transpose(-1, -2)
        # [batch, max_num_frames, num_anchors]
        reg = self.fc_reg(x.transpose(-1, -2)).transpose(-1, -2)
        reg = reg.contiguous().view(-1, self.args.max_num_frames * self.num_anchors, 2)
        # [batch, max_num_frames, num_anchors, 2]
        predict_flatten = predict.contiguous().view(predict.size(0), -1) * label_mask.float()
        cls_loss = self.criterion1(predict_flatten, label)
        # gt_box: [batch, 2]
        proposals = torch.from_numpy(self.proposals).type_as(gt).float()  # [max_num_frames, num_anchors, 2]
        proposals = proposals.view(-1, 2)
        if not self.training:
            # batch_now = reg.shape[0]
            # proposals = proposals.expand(batch_now, 800, 2)#1400,800
            # predict_box = proposals
            # predict_reg = reg # [nb, 2]
            # refine_box = predict_box + predict_reg
            # gt = gt.expand(800, batch_now, 2).transpose(0, 1).contiguous()
            # reg_loss = self.criterion2(refine_box, gt.float())
            # loss = cls_loss + 5e-3 * reg_loss #1e-3 5e-3
            # predict_flatten = (predict.contiguous().view(predict.size(0), -1) * label_mask.float())
            indices = torch.argmax(predict_flatten, -1)
            predict_box = proposals[indices]  # [nb, 2]
            predict_reg = reg[range(reg.size(0)), indices]  # [nb, 2]
            refine_box = predict_box + predict_reg
            reg_loss = self.criterion2(refine_box, gt.float())
            if self.args.dataset in ['TACOS', 'Charades']:
               loss = cls_loss + 5e-3 * reg_loss #1e-3 5e-3
            else:
               loss = cls_loss + 1e-3 * reg_loss #1e-3 5e-3
        else:
            # indices = torch.argmax(label, -1)
            indices = torch.where(adj_mat > 0)
            batch_now = reg.shape[0]
            if self.args.dataset == 'TACOS':
                proposals = proposals.expand(batch_now, 800, 2)#1400,800
            elif self.args.dataset == 'Charades':
                proposals = proposals.expand(batch_now, 800, 2)#1400,800
            else:
                proposals = proposals.expand(batch_now, 1400, 2)#1400,800
            predict_box = proposals[indices]  # [nb, 2]
            predict_reg = reg[indices]  # [nb, 2]
            refine_box = predict_box + predict_reg
            if self.args.dataset == 'TACOS':
                gt = gt.expand(800, batch_now, 2).transpose(0, 1).contiguous()
            elif self.args.dataset == 'Charades':
                gt = gt.expand(800, batch_now, 2).transpose(0, 1).contiguous()
            else:         
                gt = gt.expand(1400, batch_now, 2).transpose(0, 1).contiguous()
            gt = gt[indices]
            reg_loss = self.criterion2(refine_box, gt.float())
            if self.args.dataset in ['TACOS', 'Charades']:
                loss = cls_loss + 5e-3 * reg_loss #1e-3 5e-3
            else:
                loss = cls_loss + 1e-3 * reg_loss #1e-3 5e-3
        # predict_box = proposals[indices]  # [nb, 2]
        # predict_reg = reg[range(reg.size(0)), indices]  # [nb, 2]
        # refine_box = predict_box + predict_reg
        # reg_loss = self.criterion2(refine_box, gt.float())
        # loss = cls_loss + 1e-3 * reg_loss #1e-3 5e-3
        # if detail:
        #     return refine_box, loss, predict_flatten, reg, proposals
        return refine_box, loss, predict_flatten, None, None
