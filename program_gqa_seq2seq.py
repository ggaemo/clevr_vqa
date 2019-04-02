from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ops import ConvBlock, Coordinate, Attention

from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p, max_length):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class Model(torch.nn.Module):
    def __init__(self, g_theta_layer,
                 f_phi_layer,
                 embedding_dim, rnn_dim,
                 input_dim,
                 answer_vocab_size, fixed_embed, **kwargs):
        super(Model, self).__init__()

        self.encode = FCReLUBlock(2048 + 4, [2048, 1024])

        prev_channel = 1024

        with open('gqadata/glove_300d_gqa.pkl', 'rb') as f:
            weight = pickle.load(f)
            self.embedding = nn.Embedding(*weight.shape)
            self.embedding.load_state_dict({'weight': torch.Tensor(weight)})

        if fixed_embed:
            self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=rnn_dim,
                          bidirectional=True)

        self.g_theta_layer = list()

        prev_channel = (prev_channel) + rnn_dim * 2

        self.g_theta = FCReLUBlock(prev_channel, g_theta_layer)

        # FCReLUResBlock

        self.f_phi_layer = list()

        prev_channel = g_theta_layer[-1]

        for layer_num, channel in enumerate(f_phi_layer):
            # self.f_phi_layer.append(nn.Linear(prev_channel, channel))
            self.f_phi_layer.append(nn.utils.weight_norm(nn.Linear(prev_channel, channel)))
            self.f_phi_layer.append(nn.Dropout(0.5))
            self.f_phi_layer.append(nn.ReLU())
            prev_channel = channel

        self.f_phi = nn.Sequential(*self.f_phi_layer)

        self.classifier = nn.Sequential(*[
            nn.Linear(prev_channel, answer_vocab_size,
                      bias=False)])

    def forward(self, x):
        image_embed, question_padded, q_mask_padded, lengths = x
        image_embed = self.encode(image_embed)

        # image_embed = image_embed / (image_embed.norm(p=2, dim=1, keepdim=True).expand_as(
        #     image_embed) + 1e-8)

        embedded = self.embedding(question_padded)
        embedded = pack_padded_sequence(embedded, lengths, batch_first=True)

        self.gru.flatten_parameters()
        _, question_embed = self.gru(embedded) # (num_layers *
        # question_embed_t, _ = self.gru(embedded) # (num_layers *
        # num_directions, batch, hidden_size):

        # question_embed_t, lengths = pad_packed_sequence(question_embed_t)
        #
        # question_embed_t = question_embed_t.permute(1, 2, 0) #batch, num_directions
        # # *hidden_size, seq_len
        # question_embed_t = question_embed_t.unsqueeze(3)  # batch, num_directions
        # # *hidden_size, seq_len, 1
        #
        # q_att = self.q_att(question_embed_t)
        # question_embed = torch.sum(torch.mul(q_att, question_embed_t), (2, 3))

        question_embed = question_embed.permute(1, 0, 2) # b, 2, channel
        question_embed = question_embed.flatten(1)  # b, 2 *channel
        question_embed = question_embed.unsqueeze(1)
        question_embed = question_embed.expand(-1, 100, -1)  # (batch_size, num_obj,
        # channel * 2)

        image_question = torch.cat((image_embed, question_embed), dim=2)

        feature = self.g_theta(image_question)

        feature_agg = torch.sum(feature, 1)

        feature_agg = self.f_phi(feature_agg)

        logit = self.classifier(feature_agg)

        return logit