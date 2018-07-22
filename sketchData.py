import numpy as np
from os.path import isfile, join, isdir
from torch.utils.data import Dataset, DataLoader
from utils import augment_strokes, to_big_strokes

class SketchDataset(Dataset):
    """Class for loading data."""

    def __init__(self,
                 strokes,
                 max_seq_length=250,
                 scale_factor=1.0,
                 random_scale_factor=0.0,
                 augment_stroke_prob=0.0,
                 batch_size=32,
                 limit=1000):
        self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
        self.scale_factor = scale_factor  # divide offsets by this factor
        self.random_scale_factor = random_scale_factor  # data augmentation method
        # Removes large gaps in the data. x and y offsets are clamped to have
        # absolute value no greater than this limit.
        self.limit = limit
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
        self.start_stroke_token = [0, 0, 1, 0, 0]  # S_0 in sketch-rnn paper
        # sets self.strokes (list of ndarrays, one per sketch, in stroke-3 format,
        # sorted by size)
        self.strokes = []
        self.strokeLen = []
        self.preprocess(strokes)

        self.N = len(self.strokes)

        # randomize the batch
        self.batch_size = batch_size
        self.batch_num = int(self.N)/int(batch_size)
        self.batch_ind = 0
        self.batch_idx = []

    def preprocess(self, strokes):
        """Remove entries from strokes having > max_seq_length points.
           self.strokes stores big-strokes
        """
        raw_data = []
        seq_len = []
        count_data = 0

        for i in range(len(strokes)):
            data = strokes[i]
            if len(data) <= (self.max_seq_length):
                count_data += 1
                # removes large gaps from the data
                data = np.minimum(data, self.limit)
                data = np.maximum(data, -self.limit)
                data = np.array(data, dtype=np.float32)
                data[:, 0:2] /= self.scale_factor
                data_big = to_big_strokes(data)
                raw_data.append(data_big)
                seq_len.append(len(data))
        seq_len = np.array(seq_len)  # nstrokes for each sketch
        idx = np.argsort(seq_len)
        for i in range(len(seq_len)):
            self.strokes.append(raw_data[idx[i]])
            self.strokeLen.append(seq_len[idx[i]])
        print("total images <= max_seq_len is %d" % count_data)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.strokes[idx]

    def random_sample(self):
        """Return a random sample, in stroke-3 format as used by draw_strokes."""
        sample = np.copy(random.choice(self.strokes))
        return sample

    def random_scale(self, data):
        """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
        x_scale_factor = (
                np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (
                np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result

    def calculate_normalizing_scale_factor(self):
        """Calculate the normalizing factor explained in appendix of sketch-rnn."""
        data = []
        for i in range(len(self.strokes)):
            if len(self.strokes[i]) > self.max_seq_length:
                continue
            for j in range(len(self.strokes[i])):
                data.append(self.strokes[i][j, 0])
                data.append(self.strokes[i][j, 1])
        data = np.array(data)
        return float(np.std(data))

    def normalize(self, scale_factor=None):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        if scale_factor is None:
            scale_factor = self.calculate_normalizing_scale_factor()
        self.scale_factor = scale_factor
        for i in range(len(self.strokes)):
            self.strokes[i][:, 0:2] /= self.scale_factor

    def denormalize(self, stroke):
        """Denormalize one stroke usually for visualization"""
        stroke[:, 0:2] *= self.scale_factor
        return stroke

    def _get_batch_from_indices(self, indices):
        """Given a list of indices, return the potentially augmented batch.
           Returned strokes are sorted by length"""
        x_batch = []
        seq_len = []
        # sort the indices to make sure the data in accending sequence
        indices[::-1].sort() # = np.sort(indices, descending=True)
        for idx in range(len(indices)):
            i = indices[idx]
            data = self.random_scale(self.strokes[i])
            data_copy = np.copy(data)
            # if self.augment_stroke_prob > 0: # TODO: compatible with big stroke
            #     data_copy = augment_strokes(data_copy, self.augment_stroke_prob)
            x_batch.append(data_copy)
            length = len(data_copy)
            seq_len.append(length)
        seq_len = np.array(seq_len, dtype=int)
        # We return three things: stroke-3 format, stroke-5 format, list of seq_len.
        return self.pad_batch(x_batch), seq_len

    def random_batch(self):
        """Return a randomised portion of the training data."""
        if self.batch_ind <= 0: # resort the strokes every epoch
            idx = np.random.permutation(range(0, self.N))
            self.batch_idx = idx[0 : self.batch_num * self.batch_size].reshape(self.batch_num, self.batch_size)
            self.batch_ind = self.batch_num 
        self.batch_ind -= 1
        return self._get_batch_from_indices(self.batch_idx[self.batch_ind,:])

    def get_batch(self, idx):
        """Get the idx'th batch from the dataset."""
        return self._get_batch_from_indices(idx)

    def pad_batch(self, batch):
        """Pad the batch to be stroke-5 bigger format
           strokes in batch should in descending order by length, required by pack
           """
        max_len = len(batch[0])
        result = np.zeros((max_len + 1, self.batch_size, 5), dtype=np.float32)
        assert len(batch) == self.batch_size

        strokeTypeTarget = np.ones((max_len,self.batch_size), dtype=int) * -1 # used as classification target

        total_len = 0
        for i in range(self.batch_size):
            l = len(batch[i])
            assert l <= max_len
            # put in the first token, as described in sketch-rnn methodology
            result[0, i, :] = self.start_stroke_token
            result[1:l+1, i, :] = batch[i]
            total_len += l

            strokeTypeTarget[0:l, i] = batch[i][:, 3]
            strokeTypeTarget[l-1, i] = 2

        strokeTypeTarget = strokeTypeTarget.reshape(-1)
        strokeTypeTarget = strokeTypeTarget[strokeTypeTarget>=0]

        return result, strokeTypeTarget

if __name__=='__main__':
    from utils import drawFig, strokes_to_lines, to_normal_strokes
    import matplotlib.pyplot as plt
    import torch.nn as nn
    import torch
    datapath = '/home/wenshan/datasets/quickdraw'

    filecat = 'sketchrnn_cat.npz'

    with np.load(join(datapath, filecat)) as cat_data:
        train_cat, val_cat, test_cat = cat_data['train'], cat_data['valid'], cat_data['test']

    dataset = SketchDataset(train_cat, batch_size=3)
    dataset.normalize()

    print len(dataset)

    # for k in range(100):
    #     # import ipdb;ipdb.set_trace()
    #     sample = dataset[k]
    #     # print 'sample',sample
    #     sample_denorm = dataset.denormalize(sample)
    #     # print 'denorm', sample_denorm
    #     drawFig(sample_denorm)

    for k in range(100):
        (sample, target), seq_len = dataset.random_batch()
        print sample.shape, seq_len
        small_stroke = to_normal_strokes(sample[:,0,:])
        sample_denorm = dataset.denormalize(small_stroke)
        drawFig(sample_denorm)
        pack = nn.utils.rnn.pack_padded_sequence(torch.FloatTensor(sample), seq_len.tolist(), batch_first=False)
        print pack
        import ipdb;ipdb.set_trace()
