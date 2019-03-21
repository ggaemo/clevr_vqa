import torch
from torch import nn

class PrintLayer(nn.Module):
    def __init__(self, name):
        super(PrintLayer, self).__init__()
        self.name = name

    def forward(self, x):
        # Do your print / debug stuff here
        print(self.name, x.size())
        return x


class Coordinate(torch.nn.Module):
    def __init__(self, reduced_dim):
        super(Coordinate, self).__init__()
        coord_x = torch.linspace(-1, 1, steps=reduced_dim[0])
        coord_y = torch.linspace(-1, 1, steps=reduced_dim[1])
        grid_x, grid_y = torch.meshgrid(coord_x, coord_y)
        grid_coord = torch.stack([grid_x, grid_y], 0)
        self.grid_coord = grid_coord.unsqueeze(0)

    def forward(self, x):
        grid_coord = self.grid_coord.cuda()
        # grid_coord = self.grid_coord
        grid_coord = grid_coord.expand(x.size()[0], -1, -1, -1)

        x = torch.cat([x, grid_coord], 1)
        return x


class LowerMask(torch.nn.Module):
    def __init__(self, num_obj, num_pair, num_channel):
        self.num_channel = num_channel
        self.num_pair = num_pair

        super(LowerMask, self).__init__()

        # self.lower_mask = torch.tril(torch.ones((num_obj, num_obj))).unsqueeze(
        #     0).unsqueeze(3)

        self.lower_mask = torch.tril(torch.ones((num_obj, num_obj))).unsqueeze(
            0).unsqueeze(3).byte()

    def forward(self, x):
        lower_mask = self.lower_mask.cuda()
        # x = torch.mul(x, lower_mask)
        # x = x.permute(0, 2, 3, 1)
        x = torch.masked_select(x, lower_mask)
        x = x.view(-1, self.num_pair, self.num_channel)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class Attention(nn.Module):
    def __init__(self, input_channel_num, channel_list):
        super(Attention, self).__init__()



        self.attention = ConvBlock(1, 1, 0, input_channel_num, channel_list)
        self.logit = nn.Conv2d(channel_list[-1],1,1,1,0, bias=True)
        # self.attention = nn.Sequential(
        #     nn.Conv2d(input_channel_num, output_channel_num, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(output_channel_num, output_channel_num//2, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(output_channel_num//2, 1, 1)
        # )

    def forward(self, x):
        out = self.attention(x)
        out = self.logit(out)
        out = nn.Softmax(1)(out.view(out.size()[0], -1)).view_as(out)
        return out

class QueryAttention(nn.Module):
    def __init__(self, ):
        super(QueryAttention, self).__init__()


        # self.attention = nn.Sequential(
        #     nn.Conv2d(input_channel_num, output_channel_num, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(output_channel_num, output_channel_num//2, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(output_channel_num//2, 1, 1)
        # )

    def forward(self, query, value):
        query = query.unsqueeze(2)
        similarity = torch.matmul(value, query)
        similarity = similarity.squeeze(2)

        out = nn.Softmax(1)(similarity)
        # out = nn.Softmax(1)(out.view(out.size()[0], -1)).view_as(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, kernel_size, stride, pad, prev_channel, channel_list):
        super(ConvBlock, self).__init__()

        layer_list = list()

        for layer_num, channel in enumerate(channel_list):
            layer_list.extend([
                nn.Conv2d(prev_channel,channel,kernel_size,stride,
                                         pad, bias=True),
                               # nn.BatchNorm2d(channel),
                               nn.ReLU(inplace=True)])
            prev_channel = channel

        self.conv_block = nn.Sequential(*layer_list)


    def forward(self, x):
        out = self.conv_block(x)
        return out

class FC_ReLU(nn.Module):
    def __init__(self, input_channel_num, output_channel_num):
        super(FC_ReLU, self).__init__()
        self.fc_relu = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(input_channel_num, output_channel_num)),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.fc_relu(x)
        return out



class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):

        q # bs, seq_len, d_model

        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.k_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2) #bs h seq_len, d_k
        q = q.transpose(1, 2)  # bs h seq_len, d_k
        v = v.transpose(1, 2)  # bs h seq_len, d_k

        scores = selfattention(q, k, v, self.d_k)

        concat = scores.tranpose(1,2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)
        return output



def selfattention(q, k, v, d_k):

    scores = torch.matmul(q, k.tranpose(-2, -1)) / torch.sqrt(d_k)

    scores = nn.Softmax(-1)

    output = torch.matmul(scores, v)

