import torch

from torch import nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ops import ConvBlock, Coordinate, Attention, FC_ReLU

from pytorch_pretrained_bert import BertModel


class Model(torch.nn.Module):
    def __init__(self, g_theta_layer,
                 f_phi_layer,
                 input_dim,
                 answer_vocab_size, fixed_embed, rnn_dim, **kwargs):
        super(Model, self).__init__()

        kernel_size = 1
        stride = 1
        self.encoder_layer = list()
        pad = 0
        #
        self.encoder_layer.append(nn.Conv2d(1024, 1024, kernel_size,
                                            stride, pad,
                                            bias=False))
        self.encoder_layer.append(nn.BatchNorm2d(1024))
        self.encoder_layer.append(nn.ReLU())
        self.encoder_layer.append(nn.Conv2d(1024, 1024, kernel_size,
                                            stride, pad,
                                            bias=False))
        self.encoder_layer.append(nn.BatchNorm2d(1024))
        self.encoder_layer.append(nn.ReLU())

        self.encode = nn.Sequential(*self.encoder_layer)

        if input_dim == 128:
            self.reduced_dim = (14, 14)
        elif input_dim == 320:
            self.reduced_dim = (20, 30)

        prev_channel = 1024

        self.grid_coord = Coordinate(self.reduced_dim)

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        if fixed_embed:
            self.bert.eval()

        bert_channel = 768
        text_channel = rnn_dim
        # self.text_encoder =  FC_ReLU(bert_channel, text_channel)
        self.text_encoder = nn.Linear(bert_channel, text_channel)

        # self.gru = nn.GRU(input_size=768, hidden_size=text_channel // 2,
        #                   bidirectional=True)

        self.gru = nn.GRU(input_size=768, hidden_size=text_channel,
                          bidirectional=True)
        self.g_theta_layer = list()

        prev_channel = (prev_channel + 2) + text_channel * 2

        # self.q_att = Attention(rnn_dim * 2, rnn_dim)

        # self.g_theta = ConvBlock(1, 1, 0, prev_channel, g_theta_layer).cuda()
        self.g_theta = ConvBlock(1, 1, 0, prev_channel, g_theta_layer)


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

        encoded_layers, _ = self.bert(question_padded, attention_mask=q_mask_padded)

        question_embed_t = encoded_layers[-1]  # batch_size, seq_len, channel last_layer

        # question_embed_t = question_embed_t[:, :, :]

        # lengths = lengths - 2
        question_embed_t = pack_padded_sequence(question_embed_t, lengths ,
                                                batch_first=True)

        self.gru.flatten_parameters()
        _, question_embed = self.gru(question_embed_t)
        # question_embed = question_embed_t[:, 0, :] #batch_size, channel
        # question_embed = self.text_encoder(question_embed)

        # question_embed_t = question_embed_t.permute(0, 2, 1) # batch, channel, seq_len
        question_embed = question_embed.permute(1, 0, 2)  # b, 2, channel
        question_embed = question_embed.flatten(1)  # (batch_size, channel * 2)
        question_embed = question_embed.unsqueeze(2).unsqueeze(3)  # (batch_size,
        # channel , 1, 1)

        question_embed = question_embed.expand(-1, -1, self.reduced_dim[0],
                                               self.reduced_dim[1])

        image_question = torch.cat((image_embed, question_embed), dim=1)  # b, c, h, w

        feature = self.g_theta(image_question)

        feature_agg = torch.sum(feature, (2, 3))

        feature_agg = self.f_phi(feature_agg)

        logit = self.classifier(feature_agg)

        return logit
