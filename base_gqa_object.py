import pickle

import torch

from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ops import ConvBlock, Coordinate, Attention, FCReLUBlock



class Model(torch.nn.Module):
    def __init__(self, g_theta_layer,
                 f_phi_layer,
                 embedding_dim, rnn_dim,
                 input_dim,
                 answer_vocab_size, fixed_embed, **kwargs):
        super(Model, self).__init__()

        # self.encode = FCReLUBlock(2048 + 4, [2048, 1024])

        prev_channel = 2048 + 4

        with open('gqadata/glove_300d_gqa_all.pkl', 'rb') as f:
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
        # image_embed = self.encode(image_embed)

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