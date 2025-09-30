import numpy as np
import scipy
from utils.load_configs import get_link_prediction_args
from utils.preprocess import load_graphs
from utils.utilities import set_random_seed
from models.trainer import Trainer
import torch


if __name__ == '__main__':
    args = get_link_prediction_args()
    print(args)

    with open('auc_result.txt', 'a+') as f:
        f.write(str(args) + '\n')
    with open('ap_result.txt', 'a+') as f:
        f.write(str(args) + '\n')

    data_snapshots_num = {'Enron': 16,
                          'UCI': 13,
                          'BitCoinOTC': 17,
                          'BitCoinAlpha': 14,
                          'ML-10M': 7,
                          'HepPh': 28}

    graphs, adjs = load_graphs(args.dataset)

    if args.featureless == True:
        feats = [
            scipy.sparse.identity(adjs[data_snapshots_num[args.dataset] - 1].shape[0]).tocsr()[range(0, x.shape[0]),
            :]
            for x in
            adjs if
            x.shape[0] <= adjs[data_snapshots_num[args.dataset] - 1].shape[0]]

    assert data_snapshots_num[args.dataset] <= len(adjs), "Time steps is illegal"

    device = torch.device("cuda:0")

    all_test_auc = []
    all_test_ap = []

    for i in range(args.num_runs):
        print('Run {:<3}'.format(i))
        set_random_seed(seed=args.seed + i)

        trainer = Trainer(graphs, adjs, feats, data_snapshots_num, device, args)
        test_auc, test_ap = trainer.run(device)

        all_test_auc.append(test_auc)
        all_test_ap.append(test_ap)

        print("Best Test AUC = {}".format(test_auc))
        print("Best Test AP = {}".format(test_ap))
        with open('auc_result.txt', 'a+') as f:
            f.write(str(test_auc) + '\n')
        with open('ap_result.txt', 'a+') as f:
            f.write(str(test_ap) + '\n')

        del trainer
        torch.cuda.empty_cache()

    print('{:<6} run average auc = {}'.format(args.num_runs, np.mean(all_test_auc)))
    print('{:<6} run std auc = {}'.format(args.num_runs, np.std(all_test_auc)))

    print('{:<6} run average ap = {}'.format(args.num_runs, np.mean(all_test_ap)))
    print('{:<6} run std ap = {}'.format(args.num_runs, np.std(all_test_ap)))

    with open('auc_result.txt', 'a+') as f:
        f.write(str(np.mean(all_test_auc)) + '\n')
        f.write(str(np.std(all_test_auc)) + '\n')

    with open('ap_result.txt', 'a+') as f:
        f.write(str(np.mean(all_test_ap)) + '\n')
        f.write(str(np.std(all_test_ap)) + '\n')
