import copy
import numpy as np
import torch
import random

def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def fixed_unigram_candidate_sampler(true_clasees,
                                    num_true,
                                    num_sampled,
                                    unique,
                                    distortion,
                                    unigrams):
    # TODO: implementate distortion to unigrams
    assert true_clasees.shape[1] == num_true
    samples = []
    for i in range(true_clasees.shape[0]):
        dist = copy.deepcopy(unigrams)
        candidate = list(range(len(dist)))
        taboo = true_clasees[i].cpu().tolist()
        for tabo in sorted(taboo, reverse=True):
            candidate.remove(tabo)
            dist.pop(tabo)
        sample = np.random.choice(candidate, size=num_sampled, replace=unique, p=dist / np.sum(dist))
        samples.append(sample)
    return samples

def to_device_graphs(graphs, adjs, delta_times, device):
    # to device
    graphs = [g.to(device) for g in graphs]
    adjs = [adj.to(device) for adj in adjs]
    delta_times = [delta_time.to(device) for delta_time in delta_times]

    return graphs, adjs, delta_times
