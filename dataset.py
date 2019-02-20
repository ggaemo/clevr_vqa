import pickle
import os
import h5py

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

import torch

import numpy as np

class PadSequence():
    def __call__(self, batch):
        batch = sorted(batch, key=lambda x: x[2], reverse=True)

        image_list = list()
        q_list = list()
        q_len_list = list()
        q_type_list = list()
        a_list = list()
        idx_list = list()


        for image, q, q_len, q_type, a, idx in  batch:
            image_list.append(image)
            q_list.append(q)
            q_len_list.append(q_len)
            q_type_list.append(q_type)
            a_list.append(a)
            idx_list.append(idx)

        image = torch.stack(image_list)
        q = [torch.LongTensor(x) for x in q_list]
        a = torch.LongTensor(a_list)
        q_type = torch.LongTensor(q_type_list)
        idx_list = torch.LongTensor(idx_list)

        q_padded = torch.nn.utils.rnn.pad_sequence(q, batch_first=True)
        lengths = torch.LongTensor(q_len_list)

        return image, q_padded, lengths, q_type, a, idx_list


class ImageDuplicate():
    def __call__(self, batch):

        image_list = list()
        q_color_list = list()
        q_type_list = list()
        q_subtype_list = list()
        a_list = list()

        for x in batch:
            q_color_list.extend(x[1])
            q_type_list.extend(x[2])
            q_subtype_list.extend(x[3])
            a_list.extend(x[4])
            num_q = len(x[1])
            image_list.extend(x[0].unsqueeze(0).expand(num_q, -1, -1, -1))

        image_list = torch.stack(image_list)
        q_color_list = torch.LongTensor(q_color_list)
        q_type_list = torch.LongTensor(q_type_list)
        q_subtype_list = torch.LongTensor(q_subtype_list)
        a_list = torch.LongTensor(a_list)

        return image_list, q_color_list, q_type_list, q_subtype_list, a_list


class CLEVRDataset(Dataset):

    def __init__(self, img_dir, qa_dir, train, transform=None):
        print('@@@@@@@@@@@@@@@@@@@@2 no EOS BOS token@@@@@@@@@@@@@@@@@@@@@@@@')

        self.img_dir = img_dir

        if train:
            self.data_type = 'train'
        else:
            self.data_type = 'val'

        with open(os.path.join(qa_dir, '{}_data.pkl'.format(self.data_type)), 'rb') as f:
            self.meta_data = pickle.load(f)

        with open(os.path.join(qa_dir, 'idx_word_dict.pkl'), 'rb') as f:
            idx_to_word_dict = pickle.load(f)
            idx_to_question = idx_to_word_dict['idx_to_question']
            idx_to_question_type = idx_to_word_dict['idx_to_question_type']
            idx_to_answer = idx_to_word_dict['idx_to_answer']

        idx_to_question[0] = '_'  # padded value
        self.question_vocab = len(idx_to_question)
        self.answer_vocab = len(idx_to_answer)
        self.START_TOKEN = self.question_vocab
        self.END_TOKEN = self.question_vocab + 1
        #
        # idx_to_question[self.START_TOKEN] = 'START_TOKEN'
        # idx_to_question[self.END_TOKEN] = 'END_TOKEN'

        self.transform = transform

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):

        row = self.meta_data[idx]

        img_index = row['image_index']

        img_name = os.path.join(self.img_dir, self.data_type,
                                'CLEVR_{}_{:06d}.png'.format(self.data_type, img_index))
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # q = [self.START_TOKEN] + row['question'] + [self.END_TOKEN]

        q = row['question']
        q_type = row['question_type']
        a = row['answer']

        return image, q, len(q), q_type, a

class CLEVRresnet101Dataset(Dataset):

    def __init__(self, qa_dir, is_train, transform=None):

        if is_train:
            self.data_type = 'train'
        else:
            self.data_type='val'

        self.feature_data = h5py.File('data/CLEVR_preprocessed/CLEVR_resnet101/{}.h5'.format(
                self.data_type), 'r')['features']


        with open(os.path.join(qa_dir, '{}_data.pkl'.format(self.data_type)), 'rb') as f:
            self.meta_data = pickle.load(f)

        with open(os.path.join(qa_dir, 'idx_word_dict.pkl'), 'rb') as f:
            idx_to_word_dict = pickle.load(f)
            idx_to_question = idx_to_word_dict['idx_to_question']
            idx_to_question_type = idx_to_word_dict['idx_to_question_type']
            idx_to_answer = idx_to_word_dict['idx_to_answer']

        idx_to_question[0] = '_'  # padded value
        self.question_vocab = len(idx_to_question)
        self.answer_vocab = len(idx_to_answer)
        self.START_TOKEN = self.question_vocab
        self.END_TOKEN = self.question_vocab + 1
        #
        # idx_to_question[self.START_TOKEN] = 'START_TOKEN'
        # idx_to_question[self.END_TOKEN] = 'END_TOKEN'

        self.transform = transform

    def __len__(self):
        return len(self.meta_data)


    def __getitem__(self, idx):
        row = self.meta_data[idx]

        img_index = row['image_index']

        img_feature = self.feature_data[img_index]

        if self.transform:
            img_feature = self.transform(img_feature)

        # q = [self.START_TOKEN] + row['question'] + [self.END_TOKEN]
        q = row['question']
        q_type = row['question_type']
        a = row['answer']

        return img_feature, q, len(q), q_type, a, idx


class SCLEVRDataset(Dataset):

    def __init__(self, data_dir, train, transform=None):

        if train:
            self.data_type = 'train'
        else:
            self.data_type = 'val'

        with open(os.path.join(data_dir, 'sort-of-clevr.pickle'), 'rb') as f:
            train_data, test_data = pickle.load(f)

        if self.data_type == 'train':
            data = train_data
        else:
            data = test_data

        self.image_list = list()
        self.q_color_list = list()
        self.q_type_list = list()
        self.q_subtype_list = list()
        self.a_list = list()

        total_len = 0
        for val in data:
            image, relation_qa, nonrelation_qa = val
            image = Image.fromarray(image.astype('uint8'), 'RGB')
            relation_q, relation_a = relation_qa
            nonrelation_q, nonrelation_a = nonrelation_qa
            ##6 for one-hot vector of color, 2 for question type, 3 for question subtype
            relation_q_list = [np.where(x)[0] for x in relation_q]
            nonrelation_q_list = [np.where(x)[0] for x in nonrelation_q]

            q_list = relation_q_list + nonrelation_q_list
            q_list = [(x[0], x[1] - 6, x[2] - 8) for x in q_list]

            self.q_color_list.extend([x[0] for x in q_list])
            self.q_type_list.extend([x[1] for x in q_list])
            self.q_subtype_list.extend([x[2] for x in q_list])
            self.a_list.extend(relation_a + nonrelation_a)
            self.image_list.extend([image] * len(q_list))
            total_len += len(q_list)



        self.data = data
        self.transform = transform
        self.total_len = total_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):

        image = self.image_list[idx]
        q_color_list = self.q_color_list[idx]
        q_type_list = self.q_type_list[idx]
        q_subtype_list = self.q_subtype_list[idx]
        a_list = self.a_list[idx]

        if self.transform:
            image = self.transform(image)

        return image, q_color_list, q_type_list, q_subtype_list, a_list

def load_data(dataname, batch_size, input_dim, num_gpu, pretrained=False):

    if dataname == 'CLEVR':

        img_dir = '/home/jinwon/Relational_Network/data/CLEVR_v1.0/images'
        qa_dir =  '/home/jinwon/Relational_Network/data/CLEVR_v1.0' \
                  '/processed_data'

        if pretrained:
            transform_list = [
            transforms.Resize((input_dim, input_dim)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ]
            test_transform_list = [
                transforms.Resize((input_dim, input_dim)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        else:
            transform_list = [
                transforms.Resize((input_dim, input_dim)),
                transforms.Pad((8, 8), padding_mode='edge'),
                transforms.RandomCrop((input_dim, input_dim)),
                transforms.RandomRotation((-2.8, 2.8)),
                transforms.ToTensor()
            ]

            test_transform_list = [
                    transforms.Resize((input_dim, input_dim)),
                    transforms.ToTensor()
                ]

        data_transform = transforms.Compose(transform_list)
        test_data_transform = transforms.Compose(test_transform_list)

        train = CLEVRDataset(img_dir, qa_dir, True, transform=data_transform)
        val = CLEVRDataset(img_dir, qa_dir, False, transform=test_data_transform)

        trn_kwargs = {'num_workers': 55, 'pin_memory': True, 'collate_fn': PadSequence()}
        test_kwargs = {'num_workers': 10, 'pin_memory': True, 'collate_fn': PadSequence()}
    elif dataname == 'CLEVRresnet101':

        qa_dir = '/home/jinwon/Relational_Network/data/CLEVR_v1.0' \
                 '/processed_data'

        train = CLEVRresnet101Dataset(qa_dir, True, transform=torch.Tensor)
        val = CLEVRresnet101Dataset(qa_dir, False, transform=torch.Tensor)
        trn_kwargs = {'num_workers': 20, 'pin_memory': True, 'collate_fn': PadSequence()}
        test_kwargs = {'num_workers': 20, 'pin_memory': True, 'collate_fn': PadSequence()}

    elif dataname == 'SCLEVR':
        data_dir = '/home/jinwon/Relational_Network/data/Sort-of-CLEVR/raw_data/shape_2_color_shape'

        data_transform_list = [
            # transforms.Resize((75, 75)),
            transforms.ToTensor()
        ]
        test_transform_list = [
            # transforms.Resize((75, 75)),
            transforms.ToTensor()
        ]

        data_transform = transforms.Compose(data_transform_list)
        test_data_transform = transforms.Compose(test_transform_list)

        train = SCLEVRDataset(data_dir, 'train', data_transform)
        val = SCLEVRDataset(data_dir, 'test', test_data_transform)
        collate_fn = None

        trn_kwargs = {'num_workers': 45, 'pin_memory': True}
        test_kwargs = {'num_workers': 2, 'pin_memory': True}



    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, **trn_kwargs)



    test_loader = DataLoader(val, batch_size=batch_size, shuffle=False, **test_kwargs)

    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(val, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, input_dim


if __name__ == '__main__':

    import matplotlib.pyplot as plt


    input_dim = 128
    # transform_list = [
    #     transforms.Resize((input_dim, input_dim)),
    #     transforms.Pad((8, 8), padding_mode='edge'),
    #     transforms.RandomCrop((input_dim, input_dim)),
    #     # transforms.RandomRotation((-2.8, 2.8), resample=PIL.Image.BILINEAR),
    #     transforms.ToTensor()
    # ]
    #
    # data_transform = transforms.Compose(transform_list)
    # train = CLEVRDataset('/home/jinwon/Relational_Network/data/CLEVR_v1.0/images',
    #              '/home/jinwon/Relational_Network/data/CLEVR_v1.0/processed_data',
    #              False, transform=data_transform)
    #
    #
    # train = CLEVRDataset('/home/jinwon/Relational_Network/data/CLEVR_v1.0/images',
    #                      '/home/jinwon/Relational_Network/data/CLEVR_v1.0/processed_data',
    #                      False, transform=data_transform)

    train = CLEVRresnet101Dataset('/home/jinwon/Relational_Network/data/CLEVR_v1.0'
                           '/processed_data', True, torch.Tensor)
    dataloader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=PadSequence())

    # data_dir = '/home/jinwon/Relational_Network/data/Sort-of-CLEVR/raw_data' \
    #            '/shape_2_color_shape_order_2_shape'
    #
    # data_transform_list = [
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor()
    # ]
    #
    # data_transform = transforms.Compose(data_transform_list)
    #
    # train = SCLEVRDataset(data_dir, 'train', data_transform)
    # with open(os.path.join(data_dir, 'ans_color_qst_dict.pickle'), 'rb') as f:
    #     answer_dict, color_dict, question_subtype_dict = pickle.load(f)

    dataloader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=PadSequence())

    for i_batch, b in enumerate(dataloader):
        image, q, len_q, q_type, a = b
        print(image.shape)



