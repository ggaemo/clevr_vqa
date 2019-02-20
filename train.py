import torch
from torch import nn
import numpy as np



def warmup_lr_scheduler(optimizer, epoch, init_lr, max_lr):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = min(init_lr + (max_lr - init_lr) / 5.0 * (epoch - 1), max_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def loss_function(target, logit, q_type, q_type_dict):
    loss = nn.CrossEntropyLoss()

    output = loss(logit, target)

    pred = torch.argmax(logit, dim=1)
    right_wrong = torch.eq(pred, target)

    acc = float(torch.sum(right_wrong)) / float(target.size()[0])

    correct_list = torch.zeros(len(q_type_dict.keys()))
    num_q_type_list = torch.zeros(len(q_type_dict.keys()))
    for q, val in q_type_dict.items():
        mask = torch.eq(q_type, q)
        num_q = torch.sum(mask)
        right_wrong_tmp = right_wrong.masked_select(mask)
        num_q_type_list[q] = num_q
        tmp_correct = float(torch.sum(right_wrong_tmp))
        correct_list[q] = tmp_correct

    return output, acc, correct_list, num_q_type_list


def train(model, train_loader, optimizer, q_type_dict, epoch, device):
    model.train()
    train_loss = 0
    train_acc = 0
    train_correct_list = np.zeros(len(q_type_dict.keys()), dtype=np.float32)
    train_num_q_list = np.zeros(len(q_type_dict.keys()), dtype=np.float32)

    acc_list = list()

    for batch_idx, (image, question_padded, lengths, q_type, a, idx) in enumerate(
            train_loader, start=1):
        image = image.to(device)
        question_padded = question_padded.to(device)
        lengths = lengths.to(device)
        a = a.to(device)
        q_type = q_type.to(device)
        optimizer.zero_grad()
        logit = model((image, question_padded, lengths))
        loss, acc, correct_list, num_q_type_list = loss_function(a, logit, q_type,
                                                                 q_type_dict)
        loss.backward()
        train_loss += loss.item()
        train_acc += acc
        train_correct_list += correct_list.data.numpy()
        train_num_q_list += num_q_type_list.data.numpy()
        optimizer.step()


        acc_list.append(acc)
        if batch_idx % 500 == 0 or batch_idx == 10:
            print(batch_idx)
            print('train', epoch, train_acc/batch_idx, 'batch acc', acc)
            print(acc_list[:20], acc_list[-20:])
            print(np.sum(train_correct_list)/np.sum(train_num_q_list), train_acc/batch_idx)

    train_loss /= batch_idx
    train_acc /= batch_idx
    train_acc_list = train_correct_list / train_num_q_list

    print('====> Train Epoch: {} loss: {:.4f}, acc {:.4f}'.format(
        epoch, train_loss, train_acc))

    return train_loss, train_acc, train_acc_list


def test(model, test_loader, q_type_dict, epoch, device):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_correct_list = np.zeros(len(q_type_dict.keys()), dtype=np.float32)
    test_num_q_list = np.zeros(len(q_type_dict.keys()), dtype=np.float32)

    acc_list = list()
    with torch.no_grad():
        for batch_idx, (image, question_padded, lengths, q_type, a, idx) in enumerate(
                test_loader, start=1):
            image = image.to(device)
            question_padded = question_padded.to(device)
            lengths = lengths.to(device)
            a = a.to(device)
            q_type = q_type.to(device)

            logit = model((image, question_padded, lengths))
            loss, acc, correct_list, num_q_type_list = loss_function(a, logit, q_type, q_type_dict)
            test_loss += loss.item()
            test_acc += acc
            test_correct_list += correct_list.data.numpy()
            test_num_q_list += num_q_type_list.data.numpy()

            acc_list.append(acc)
            if batch_idx % 100 == 0 or batch_idx == 10:
                print(batch_idx)
                print('test', epoch, test_acc / batch_idx, 'batch acc', acc)
                print(acc_list[:20], acc_list[-20:])
                print(np.sum(test_correct_list)/np.sum(test_num_q_list), test_acc/batch_idx)

            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch[:n]])
            #     # dummy_img = torch.rand(32, 3, 64, 64)
            #     # comparison = torchvision.utils.make_grid(dummy_img, normalize=True,
            #     #                                  scale_each=True)
            #     # comparison = torchvision.utils.make_grid(comparison, nrow=2)
            #     writer.add_image('recon', comparison, epoch)

    test_loss /= batch_idx
    test_acc /= batch_idx
    test_acc_list = test_correct_list / test_num_q_list

    print('====> Test Epoch: {} loss: {:.4f}, acc {:.4f}'.format(
        epoch, test_loss, test_acc))
    return test_loss, test_acc, test_acc_list
