import numpy as np
from os.path import isfile, join, isdir
from torch.utils.data import Dataset

class SketchDatasetHierarchy(Dataset):
    """Class for loading data.
    Returns (strokes, lineSeq, lineNum)
    strokes(lines) in one sketch, 20 x 30 x 2
    each line is padded to allow using dataload
    lineSeq is also padded 
    """

    def __init__(self,
                 strokes,
                 max_line_number=20, # statistic: 
                 max_line_length=30,
                 scale_factor=1.0,
                 random_scale_factor=0.0,
                 augment_stroke_prob=0.0,
                 limit=1000):

        processed_file = 'data_processed.npz'
        if isfile(processed_file): # load preprocessed file to save time
            outfile = open(processed_file, 'r')
            processed_data = np.load(outfile)
            self.sketchLineNum = processed_data['sketchLineNum']
            self.sketchLineLength = processed_data['sketchLineLength']
            self.strokePaddedLines = processed_data['strokePaddedLines']
            self.scale_factor = float(processed_data['scale_factor'])
            self.N = len(self.sketchLineNum)
            outfile.close()
            return 

        self.strokes = strokes
        self.max_line_number = max_line_number  # max number of strokes allowed in one sketch
        self.max_line_length = max_line_length 
        self.scale_factor = scale_factor  # divide offsets by this factor
        self.random_scale_factor = random_scale_factor  # data augmentation method
        # Removes large gaps in the data. x and y offsets are clamped to have
        # absolute value no greater than this limit.
        self.limit = limit
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method

        self.normalize()

        self.sketches = [] # all sketches in stroke-3 format
        # self.sketchLen = [] # list of sketch length
        self.sketchLineNum = [] # list of line number
        self.sketchLineLength = [] # strokeNum x max_line_number
        # strokeNum x max_line_number x max_line_length x 2
        self.strokePaddedLines = [] # all sketches in padded-line format
        self.preprocess(self.strokes)

        self.N = len(self.sketchLineNum)

        outfile = open(processed_file, 'w')
        np.savez(outfile, 
            sketchLineNum=self.sketchLineNum, 
            sketchLineLength=self.sketchLineLength,
            strokePaddedLines=self.strokePaddedLines,
            scale_factor=self.scale_factor)
        outfile.close()

    def preprocess(self, sketches):
        """Remove entries from strokes having > max_seq_length points.
           self.strokes storkes big-strokes
        """
        count_data = 0

        for i in range(len(self.strokes)):
            data = self.strokes[i]
            lines = self.strokes_to_lines(data)
            linenum = len(lines)
            if linenum > self.max_line_number: # too many strokes in the sketch
                continue
            lineLenList = np.zeros(self.max_line_number)
            for line_ind, line in enumerate(lines):
                lineLenList[line_ind] = len(line)

            if np.max(lineLenList) > self.max_line_length: # too many points in one stroke
                continue

            self.sketchLineLength.append(lineLenList)
            self.sketches.append(data)
            self.sketchLineNum.append(linenum)

            padded_lines = self.pad_lines(lines, self.max_line_number, self.max_line_length)
            self.strokePaddedLines.append(padded_lines)
            count_data += 1

        self.sketchLineLength = np.array(self.sketchLineLength)
        self.sketchLineNum = np.array(self.sketchLineNum)
        self.strokePaddedLines = np.array(self.strokePaddedLines)
        print("total images is %d" % count_data)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.strokePaddedLines[idx],self.sketchLineLength[idx], self.sketchLineNum[idx]

    def strokes_to_lines(self, strokes):
        """
        cut the strokes into lines without changing to global frame (still in delta_x, delta_y)
        """
        lines = []
        line = []
        for i in range(len(strokes)):
            if strokes[i, 2] == 1:
                line.append(strokes[i, 0:2])
                lines.append(line)
                line = []
            else:
                line.append(strokes[i, 0:2])
        return lines

    def pad_lines(self, lines, maxLineNum, maxLineLen):
        """
        return maxLineNum x maxLineLen x 2 numpy array
        """
        padded_array = np.zeros((maxLineNum, maxLineLen, 2))
        for lineInd,line in enumerate(lines):
            padded_array[lineInd,:len(line),:] = line
        return padded_array

    def unpad_lines(self, padded_lines):
        unpaded_array = []
        for line in padded_lines:
            lineLen = len(line)
            for k in line[::-1]:
                if k[0]==0 and k[1]==0:
                    lineLen -= 1
                else:
                    break
            if lineLen>0:
                unpaded_array.append(line[:lineLen,:])
            else:
                break
        return unpaded_array

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
            # if len(self.strokes[i]) > self.max_seq_length:
            #     continue
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
            self.strokes[i] = np.array(self.strokes[i], dtype=np.float32)
            self.strokes[i][:, 0:2] /= self.scale_factor

    def denormalize(self, stroke):
        """Denormalize one stroke usually for visualization"""
        stroke *= self.scale_factor
        return stroke


    def paddedLineDrawFig(self, paddedLine, line_len, line_num):
        paddedLine = self.denormalize(paddedLine)
        lines = self.unpad_lines(paddedLine)
        lines_draw = []
        x, y = 0, 0
        for line in lines:
            line_draw = []
            for i in range(len(line)):
                x += float(line[i, 0])
                y += float(line[i, 1])
                line_draw.append([x, y])
            lines_draw.append(np.array(line_draw))

        for line in lines_draw:
            line_np = np.array(line)
        #     print cat0_np.shape, cat0_np, cat0_np[:,0], cat0_np[:,1]
            plt.plot(line_np[:,0],0-line_np[:,1])
        plt.show()


if __name__=='__main__':
    import matplotlib.pyplot as plt
    import torch.nn as nn
    import torch
    from torch.utils.data import Dataset, DataLoader

    datapath = '/home/wenshan/datasets/quickdraw'

    filecat = 'sketchrnn_cat.npz'

    with np.load(join(datapath, filecat)) as cat_data:
        train_cat, val_cat, test_cat = cat_data['train'], cat_data['valid'], cat_data['test']

    dataset = SketchDatasetHierarchy(train_cat)
    print len(dataset)

    # for k in range(100):
    #     padded_lines, line_len, line_num = dataset[k]
    #     dataset.paddedLineDrawFig(padded_lines,line_len, line_num)
    
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=2)
    dataiter = iter(dataloader)

    while True:
        sample = dataiter.next()
        dataset.paddedLineDrawFig(sample[0][0,:].numpy(),sample[1][0,:].numpy(), sample[2][0].numpy())
        # import ipdb; ipdb.set_trace()
        # print sample.shape, seq_len
        # small_stroke = to_normal_strokes(sample[:,0,:])
        # sample_denorm = dataset.denormalize(small_stroke)
        # drawFig(sample_denorm)
        # pack = nn.utils.rnn.pack_padded_sequence(torch.FloatTensor(sample), seq_len.tolist(), batch_first=False)
        # print pack
        # import ipdb;ipdb.set_trace()
