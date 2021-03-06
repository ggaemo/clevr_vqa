import pickle

import torch

from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence

from ops import ConvBlock, Coordinate, Attention, FC_ReLU

class Model(torch.nn.Module):
    def __init__(self, g_theta_layer,
                 f_phi_layer,
                 input_dim,
                 embedding_dim, rnn_dim, img_att_layer,
                 answer_vocab_size, fixed_embed, **kwargs):
        super(Model, self).__init__()

        kernel_size = 1
        stride = 1
        self.encoder_layer = list()
        pad = 0


        self.encoder_layer.append(nn.Conv2d(1024, 512, kernel_size,
                                            stride, pad,
                                            bias=False))
        self.encoder_layer.append(nn.BatchNorm2d(512))
        self.encoder_layer.append(nn.ReLU())
        self.encoder_layer.append(nn.Conv2d(512, 512, kernel_size,
                                            stride, pad,
                                            bias=False))
        self.encoder_layer.append(nn.BatchNorm2d(512))
        self.encoder_layer.append(nn.ReLU())

        self.encode = nn.Sequential(*self.encoder_layer)

        if input_dim == 128:
            self.reduced_dim = (14, 14)
        elif input_dim == 320:
            self.reduced_dim = (20, 30)
        img_channel = 512

        self.grid_coord = Coordinate(self.reduced_dim)


        with open('glove_60b_{}.pkl'.format(embedding_dim), 'rb') as f:
            weight = pickle.load(f)
            self.embedding = nn.Embedding(*weight.shape)
            self.embedding.load_state_dict({'weight': torch.Tensor(weight)})

        if fixed_embed:
            self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=rnn_dim,
                          bidirectional=True)

        text_channel = rnn_dim * 2

        self.g_theta_layer = list()

        prev_channel = (img_channel + 2) + text_channel

        common_embedding_channel = 512

        self.attention = Attention(prev_channel, img_att_layer)

        # self.visual_feature = FC_ReLU(img_channel + 2, common_embedding_channel)
        # self.text_feature = FC_ReLU(text_channel, common_embedding_channel)
        self.visual_feature = nn.Linear(img_channel + 2, common_embedding_channel)
        self.text_feature = nn.Linear(text_channel, common_embedding_channel)

        prev_channel = common_embedding_channel
        self.f_phi_layer = list()
        for layer_num, channel in enumerate(f_phi_layer):
            self.f_phi_layer.append(nn.Linear(prev_channel, channel))
            # self.f_phi_layer.append(nn.utils.weight_norm(nn.Linear(prev_channel, channel)))
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
        _, question_embed = self.gru(embedded)

        question_embed = question_embed.permute(1, 0, 2) # b, 2, channel
        question_embed = question_embed.flatten(1)  # (batch_size, channel * 2)
        question_embed_expanded = question_embed.unsqueeze(2).unsqueeze(3)  # (batch_size,
        # channel * 2, 1, 1)

        question_embed_expanded = question_embed_expanded.expand(-1, -1,
                                                                 self.reduced_dim[0],
                                                                 self.reduced_dim[0])

        image_question = torch.cat((image_embed, question_embed_expanded), dim=1)  # b, c, h, w

        attention = self.attention(image_question)

        img_att = torch.sum(torch.mul(attention, image_embed), (2, 3))

        visual_feature = self.visual_feature(img_att)

        text_feature = self.text_feature(question_embed)

        multimodal_feature = torch.mul(visual_feature, text_feature)

        feature = self.f_phi(multimodal_feature)

        logit = self.classifier(feature)

        return logit