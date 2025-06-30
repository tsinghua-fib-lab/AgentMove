from .load import max_len, process_traj
import time
import random
from torch import optim
import torch.utils.data as data
from .models import *
import argparse

from tqdm import tqdm
from .config import LR, MAX_EPOCH, BATCH_SIZE, SAVE_DIR, PROMPT_NUM, CHECKPOINT_DIR

def calculate_mrr_and_ndcg(probs, labels, k=5):
    """
    Calculate MRR@k and NDCG@k.
    :param probs: The probabilities from the model (N, num_classes)
    :param labels: The ground truth labels (N,)
    :param k: Top k predictions to consider
    :return: MRR and NDCG scores
    """
    batch_size = probs.shape[0]
    mrr = 0
    ndcg = 0
    
    _, topk_indices = torch.topk(probs, k=k, dim=1)
    labels = labels.view(-1, 1).expand_as(topk_indices)
    
    # Calculate MRR
    matches = topk_indices == labels
    correct_indices = matches.nonzero(as_tuple=True)
    if len(correct_indices[0]) > 0:
        ranks = correct_indices[1] + 1  # Convert indices to ranks (1-indexed)
        mrr = torch.sum(1.0 / ranks.float()).item() / batch_size

    # Calculate NDCG
    for i in range(batch_size):
        if matches[i].any():
            rank = (matches[i].nonzero(as_tuple=True)[0][0] + 1).float()
            dcg = (2 ** 1 - 1) / torch.log2(rank + 1)
            idcg = (2 ** 1 - 1) / torch.log2(torch.tensor(1.) + 1)
            ndcg += (dcg / idcg).item()
    
    ndcg /= batch_size
    return mrr, ndcg

def calculate_acc(prob, label):
    # log_prob (N, L), label (N), batch_size [*M]
    acc_train = [0, 0, 0, 0]
    for i, k in enumerate([1, 5, 10, 20]):
        # topk_batch (N, k)
        _, topk_predict_batch = torch.topk(prob, k=k)
        for j, topk_predict in enumerate(to_npy(topk_predict_batch)):
            # topk_predict (k)
            if to_npy(label)[j] in topk_predict:
                acc_train[i] += 1

    return np.array(acc_train)


def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1]-1  # prob (N, L)
    label = label.view(-1)  # label (N)
    init_label = np.linspace(0, num_label-1, num_label)  # (N), [0 -- num_label-1]
    init_prob = torch.zeros(size=(num_label, num_neg+len(label)))  # (N, num_neg+num_label)

    random_ig = random.sample(range(1, l_m+1), num_neg)  # (num_neg) from (1 -- l_max)
    while len([lab for lab in label if lab in random_ig]) != 0:  # no intersection
        random_ig = random.sample(range(1, l_m+1), num_neg)

    global global_seed
    random.seed(global_seed)
    global_seed += 1

    # place the pos labels ahead and neg samples in the end
    for k in range(num_label):
        for i in range(num_neg + len(label)):
            if i < len(label):
                init_prob[k, i] = prob[k, label[i]]
            else:
                init_prob[k, i] = prob[k, random_ig[i-len(label)]]

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label)  # (N, num_neg+num_label), (N)


class DataSet(data.Dataset):
    def __init__(self, traj, m1, v, label, length):
        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M), (NUM), (NUM)
        self.traj, self.mat1, self.vec, self.label, self.length = traj, m1, v, label, length

    def __getitem__(self, index):
        traj = self.traj[index].to(device)
        mats1 = self.mat1[index].to(device)
        vector = self.vec[index].to(device)
        label = self.label[index].to(device)
        length = self.length[index].to(device)
        return traj, mats1, vector, label, length

    def __len__(self):  # no use
        return len(self.traj)


class Trainer:
    def __init__(self, model, record):
        # load other parameters
        self.model = model.to(device)
        self.records = record
        self.start_epoch = record['epoch'][-1] if load else 1
        self.num_neg = 10
        self.interval = 1000
        self.batch_size = BATCH_SIZE # N = 1
        self.learning_rate = LR
        self.num_epoch = MAX_EPOCH
        self.threshold = np.mean(record['acc_valid'][-1]) if load else 0  # 0 if not update

        # (NUM, M, 3), (NUM, M, M, 2), (L, L), (NUM, M, M), (NUM, M), (NUM) i.e. [*M]
        self.traj, self.mat1, self.mat2s, self.mat2t, self.label, self.len = \
            trajs, mat1, mat2s, mat2t, labels, lens
        # nn.cross_entropy_loss counts target from 0 to C - 1, so we minus 1 here.
        self.dataset = DataSet(self.traj, self.mat1, self.mat2t, self.label-1, self.len)
        self.data_loader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

        self.sorted_user_ids = sorted_user_ids
        self.testing_user_ids = test_user_ids

    def train(self):
        # set optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)

        for t in range(self.num_epoch):
            # settings or validation and test
            valid_size, test_size, test_size2 = 0, 0, 0
            acc_valid, acc_test, acc_test2 = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
            mrr_valid, ndcg_valid = 0, 0
            mrr_test, ndcg_test, mrr_test2, ndcg_test2 = 0, 0, 0, 0

            bar = tqdm(total=part)
            for step, item in enumerate(self.data_loader):
                user_id_int = self.sorted_user_ids[step]
                test_traj_id, sample_id, traj_len = self.testing_user_ids[str(user_id_int)]
                # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
                person_input, person_m1, person_m2t, person_label, person_traj_len = item

                # first, try batch_size = 1 and mini_batch = 1
                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(device)
                for mask_len in range(1, person_traj_len[0]+1):  # from 1 -> len
                    # if mask_len != person_traj_len[0]:
                    #     continue
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]  # (N)
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len)  # (N, L)

                    # 减去测试轨迹的全部长度做训练，只保留最后一个点进行测试
                    if mask_len <= person_traj_len[0] - 2:  # only training
                        # nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                        prob_sample, label_sample = sampling_prob(prob, train_label, self.num_neg)
                        loss_train = F.cross_entropy(prob_sample, label_sample)
                        loss_train.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    elif mask_len == person_traj_len[0] - 1:  # only validation
                        valid_size += person_input.shape[0]
                        # v_prob_sample, v_label_sample = sampling_prob(prob_valid, valid_label, self.num_neg)
                        # loss_valid += F.cross_entropy(v_prob_sample, v_label_sample, reduction='sum')
                        acc_valid += calculate_acc(prob, train_label)
                        mrr, ndcg = calculate_mrr_and_ndcg(prob, train_label, k=5)
                        mrr_valid += mrr
                        ndcg_valid += ndcg

                    elif mask_len == person_traj_len[0]:  # only test
                        test_size2 += person_input.shape[0]
                        acc_test2 += calculate_acc(prob, train_label)
                        mrr, ndcg = calculate_mrr_and_ndcg(prob, train_label, k=5)
                        mrr_test2 += mrr
                        ndcg_test2 += ndcg

                        if sample_id <= PROMPT_NUM:
                            test_size += person_input.shape[0] 
                            # v_prob_sample, v_label_sample = sampling_prob(prob_valid, valid_label, self.num_neg)
                            # loss_valid += F.cross_entropy(v_prob_sample, v_label_sample, reduction='sum')
                            acc_test += calculate_acc(prob, train_label)                            
                            mrr_test += mrr
                            ndcg_test += ndcg

                bar.update(self.batch_size)
            bar.close()

            acc_valid = np.array(acc_valid) / valid_size
            print('epoch:{}, time:{}, valid_acc:{}'.format(self.start_epoch + t, time.time() - start, acc_valid))

            acc_test = np.array(acc_test) / test_size
            mrr_test = np.array(mrr_test) / test_size
            ndcg_test = np.array(ndcg_test) / test_size
            acc_test2 = np.array(acc_test2) / test_size2
            mrr_test2 = np.array(mrr_test2) / test_size2
            ndcg_test2 = np.array(ndcg_test2) / test_size2
            print('epoch:{}, time:{}, test_acc:{}, test_acc_raw:{}, test_mrr:{}, test_mrr_raw:{}, test_ndcg:{}, test_ndcg_raw:{}'.format(self.start_epoch + t, time.time() - start, acc_test, acc_test2, mrr_test, mrr_test2, ndcg_test, ndcg_test2))

            self.records['acc_valid'].append(acc_valid)
            self.records['acc_test'].append(acc_test)
            self.records['epoch'].append(self.start_epoch + t)

            if self.threshold < np.mean(acc_valid):
                self.threshold = np.mean(acc_valid)
                # save the model
                torch.save({'state_dict': self.model.state_dict(),
                            'records': self.records,
                            'time': time.time() - start},
                           os.path.join(CHECKPOINT_DIR, 'best_stan_ndcg_mrr_' + dname + "_" + str(self.num_epoch) + "_" + str(self.batch_size) + '.pth'))

    def inference(self):
        user_ids = []
        for t in range(self.num_epoch):
            # settings or validation and test
            valid_size, test_size = 0, 0
            acc_valid, acc_test = [0, 0, 0, 0], [0, 0, 0, 0]
            cum_valid, cum_test = [0, 0, 0, 0], [0, 0, 0, 0]

            for step, item in enumerate(self.data_loader):
                # get batch data, (N, M, 3), (N, M, M, 2), (N, M, M), (N, M), (N)
                person_input, person_m1, person_m2t, person_label, person_traj_len = item

                # first, try batch_size = 1 and mini_batch = 1

                input_mask = torch.zeros((self.batch_size, max_len, 3), dtype=torch.long).to(device)
                m1_mask = torch.zeros((self.batch_size, max_len, max_len, 2), dtype=torch.float32).to(device)
                for mask_len in range(1, person_traj_len[0] + 1):  # from 1 -> len
                    # if mask_len != person_traj_len[0]:
                    #     continue
                    input_mask[:, :mask_len] = 1.
                    m1_mask[:, :mask_len, :mask_len] = 1.

                    train_input = person_input * input_mask
                    train_m1 = person_m1 * m1_mask
                    train_m2t = person_m2t[:, mask_len - 1]
                    train_label = person_label[:, mask_len - 1]  # (N)
                    train_len = torch.zeros(size=(self.batch_size,), dtype=torch.long).to(device) + mask_len

                    prob = self.model(train_input, train_m1, self.mat2s, train_m2t, train_len)  # (N, L)

                    if mask_len <= person_traj_len[0] - 2:  # only training
                        continue

                    elif mask_len == person_traj_len[0] - 1:  # only validation
                        acc_valid = calculate_acc(prob, train_label)
                        cum_valid += calculate_acc(prob, train_label)

                    elif mask_len == person_traj_len[0]:  # only test
                        acc_test = calculate_acc(prob, train_label)
                        cum_test += calculate_acc(prob, train_label)

                print(step, acc_valid, acc_test)

                if acc_valid.sum() == 0 and acc_test.sum() == 0:
                    user_ids.append(step)


if __name__ == '__main__':
    # load data
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_name', type=str, default='NYC')
    args = parser.parse_args()
    dname = args.city_name
    file = open(os.path.join(SAVE_DIR, dname + '_data.pkl'), 'rb')
    file_data = joblib.load(file)
    # tensor(NUM, M, 3), np(NUM, M, M, 2), np(L, L), np(NUM, M, M), tensor(NUM, M), np(NUM)
    # self.traj:person_input, self.mat1:person_m1, self.mat2t:person_m2t, self.label-1:person_label, self.len:person_traj_len
    [trajs, mat1, mat2s, mat2t, labels, lens, u_max, l_max] = file_data
    mat1, mat2s, mat2t, lens = torch.FloatTensor(mat1), torch.FloatTensor(mat2s).to(device), \
                               torch.FloatTensor(mat2t), torch.LongTensor(lens)

    # load中对轨迹重新排序后的用户id顺序 load.py
    sorted_user_ids = json.load(open(os.path.join(SAVE_DIR, dname + '_sorted_user_ids.json')))
    # 原始轨迹构造过程中的用户ID及相关信息 agentmove_data.py
    test_user_ids = json.load(open(os.path.join(SAVE_DIR, dname + '_testing_set.json')))

    # the run speed is very flow due to the use of location matrix (also huge memory cost)
    # please use a partition of the data (recommended)
    part = 100
    trajs, mat1, mat2t, labels, lens = \
        trajs[:part], mat1[:part], mat2t[:part], labels[:part], lens[:part]

    ex = mat1[:, :, :, 0].max(), mat1[:, :, :, 0].min(), mat1[:, :, :, 1].max(), mat1[:, :, :, 1].min()

    stan = Model(t_dim=hours+1, l_dim=l_max+1, u_dim=u_max+1, embed_dim=50, ex=ex, dropout=0)
    num_params = 0

    for name in stan.state_dict():
        print(name)

    for param in stan.parameters():
        num_params += param.numel()
    print('num of params', num_params)

    load = False

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if load:
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR,'best_stan_win_' + dname + '.pth'))
        stan.load_state_dict(checkpoint['state_dict'])
        start = time.time() - checkpoint['time']
        records = checkpoint['records']
    else:
        records = {'epoch': [], 'acc_valid': [], 'acc_test': []}
        start = time.time()

    trainer = Trainer(stan, records)
    trainer.train()
    # trainer.inference()

