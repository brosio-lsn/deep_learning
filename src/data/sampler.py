from collections import defaultdict
import torch
from torch.utils.data import Sampler

class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Build buckets: signature → list of indices
        buckets = defaultdict(list)
        for idx in range(len(dataset)):
            sig = dataset.__getitem__(idx)['this_step_start']
            buckets[sig].append(idx)

        # Create batches
        self.batches = []
        for sig, indices in buckets.items():
            if shuffle:
                indices = torch.tensor(indices)
                perm = torch.randperm(len(indices))
                indices = indices[perm].tolist()

            for i in range(0, len(indices), batch_size):
                self.batches.append(indices[i:i+batch_size])

        # Shuffle batches themselves
        if shuffle:
            perm = torch.randperm(len(self.batches))
            self.batches = [self.batches[i] for i in perm]

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch
