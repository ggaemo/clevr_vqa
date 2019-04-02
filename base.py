import pickle

import torch

from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ops import ConvBlock, Coordinate, Attention



class Model(torch.nn.Module):
    def __init__(self, g_theta_layer,
                 f_phi_layer,
                 embedding_dim, rnn_dim,
                 input_dim,
                 answer_vocab_size, fixed_embed, **kwargs):
        super(Model, self).__init__()

        kernel_size = 3
        stride = 1
        self.encoder_layer = list()
        pad = 1
        #
        self.encoder_layer.append(nn.Conv2d(1024, 256, kernel_size,
                                            stride, pad,
                                            bias=False))
        self.encoder_layer.append(nn.BatchNorm2d(256))
        self.encoder_layer.append(nn.ReLU())
        self.encoder_layer.append(nn.Conv2d(256, 256, kernel_size,
                                            stride, pad,
                                            bias=False))
        self.encoder_layer.append(nn.BatchNorm2d(256))
        self.encoder_layer.append(nn.ReLU())

        self.encode = nn.Sequential(*self.encoder_layer)

        if input_dim == 128:
            self.reduced_dim = (14, 14)
        elif input_dim == 320:
            self.reduced_dim = (20, 30)
        elif input_dim == 2048:
            self.reduced_dim = (7, 7)
        prev_channel = 256

        self.grid_coord = Coordinate(self.reduced_dim)


        with open('glove_60b_{}.pkl'.format(embedding_dim), 'rb') as f:
            weight = pickle.load(f)
            self.embedding = nn.Embedding(*weight.shape)
            self.embedding.load_state_dict({'weight': torch.Tensor(weight)})

        if fixed_embed:
            self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=rnn_dim,
                          bidirectional=True)

        self.g_theta_layer = list()

        prev_channel = (prev_channel + 2) + rnn_dim * 2

        # self.q_att = Attention(rnn_dim * 2, rnn_dim)

        self.g_theta = ConvBlock(1, 1, 0, prev_channel, g_theta_layer)
        # self.g_theta = ConvBlock(1, 1, 0, prev_channel, g_theta_layer)

        self.f_phi_layer = list()

        prev_channel = g_theta_layer[-1]

        for layer_num, channel in enumerate(f_phi_layer):
            # self.f_phi_layer.append(nn.Linear(prev_channel, channel))
            self.f_phi_layer.append(nn.utils.weight_norm(nn.Linear(prev_channel, channel)))
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
        image_embed = self.grid_coord(image_embed)

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
        question_embed = question_embed.flatten(1)  # (batch_size, channel * 2)
        question_embed = question_embed.unsqueeze(2).unsqueeze(3)  # (batch_size, channel * 2, 1, 1)

        question_embed = question_embed.expand(-1, -1, self.reduced_dim[0],
                                               self.reduced_dim[1])

        image_question = torch.cat((image_embed, question_embed), dim=1)  # b, c, h, w

        feature = self.g_theta(image_question)

        feature_agg = torch.sum(feature, (2, 3))

        feature_agg = self.f_phi(feature_agg)

        logit = self.classifier(feature_agg)

        return logit