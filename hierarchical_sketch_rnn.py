import torch 
import torch.nn as nn
from utils import output_to_strokes
import numpy as np

class StrokeRnn(nn.Module): 
    '''
    Input: (delta_x, delta_y)
    Encode: VAE
    Decode: (delta_x, delta_y), (0, 0) means end of the stroke
    '''
    def __init__(self, inputNum, hiddenNum, outputNum):
        super(StrokeRnn, self).__init__()
        self.hiddenNum = hiddenNum
        self.inputNum = inputNum
        self.encoder = nn.LSTM(inputNum, hiddenNum)
        self.logstd = nn.Linear(hiddenNum, hiddenNum)
        self.mean = nn.Linear(hiddenNum, hiddenNum)

        self.decoderH = nn.Linear(hiddenNum, hiddenNum)
        self.decoderC = nn.Linear(hiddenNum, hiddenNum)
        self.decoder = nn.LSTM(inputNum+hiddenNum, hiddenNum) # the input is combined with the coding
        self.output = nn.Linear(hiddenNum, outputNum)

    def forward(self, x, seq_len):
        (seqNum, batchNum, inputNum) = x.size()  
        code, meanVar, logstdVar = self.encode(x, seq_len, batchNum)
        outputVar = self.decode(code, batchNum, seqNum)

        return outputVar, meanVar, logstdVar


    def encode(self, x, seq_len, batchNum):
        # import ipdb;ipdb.set_trace()
        # x_pack = nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=False)
        # outputVar: max_seq_len x batchNum(linenum) x hidden
        outputVar, _ = self.encoder(x, self.init_hidden(batchNum))
        
        hn = torch.zeros((0)).cuda()
        for ind, slen in enumerate(seq_len):
            hn = torch.cat((hn, outputVar[slen-1,ind,:]))

        meanVar = self.mean(hn.view(batchNum, -1)) # batch x hidden
        logstdVar = self.logstd(hn.view(batchNum, -1)) # batch x hidden

        # sample from the mean and logstd
        epsilon = torch.normal(mean=torch.zeros_like(meanVar))
        sample = meanVar + (logstdVar/2).exp() * epsilon # batch x hidden

        return sample, meanVar, logstdVar

    def decode(self, code, batchNum, seqNum):
        '''
        seqNum: longest sequence length in one batch
        '''
        hn = torch.tanh(self.decoderH(code)).view(1,batchNum,-1) 
        cn = torch.tanh(self.decoderC(code)).view(1,batchNum,-1) 

        outlist = []
        decoderInput = torch.zeros((1, batchNum, self.inputNum)).cuda()
        decoderInput = torch.cat((decoderInput, code.view((1, batchNum, self.hiddenNum))), dim=2) # 1 x batch x hidden
        hidden = (hn, cn)
        # if testing, input the data in sequence one by one
        for k in range(seqNum):
            outputVar, hidden = self.decoder(decoderInput, hidden)
            outputVar = self.output(outputVar.view(-1, self.hiddenNum))

            outlist.append(outputVar)
            decoderInput = torch.cat((outputVar.view((1, batchNum, -1)), code.view((1, batchNum, self.hiddenNum))), dim=2)

        outputVar = torch.cat(outlist, dim=0) # (seq x batch) x ouput

        return outputVar.view(seqNum, batchNum, -1)

    def init_hidden(self, batchNum):
        return torch.zeros(1, batchNum, self.hiddenNum).cuda(), \
                torch.zeros(1, batchNum, self.hiddenNum).cuda()


class SketchRnn(nn.Module):
    def __init__(self, inputNum, lineHiddenNum, sketchHiddenNum, outputNum):
        super(SketchRnn, self).__init__()
        self.lineHiddenNum = lineHiddenNum
        self.lineRnn = StrokeRnn(inputNum, lineHiddenNum, outputNum)
        self.sketchRnn = StrokeRnn(lineHiddenNum, sketchHiddenNum, lineHiddenNum)

    def forward(self, x, line_len, sketch_len):
        (seqNum, lineBatch, inputNum) = x.size()  
        sketchBatch = len(sketch_len)
        lineCode, LineMeanVar, LineLogstdVar = self.lineRnn.encode(x, line_len, lineBatch)

        # reshape and pad the lineCode
        maxlen = max(sketch_len) + 1 # pad one zero-sequence as eof
        sketchInput = torch.zeros((0)).cuda()
        ind = 0
        for ind, slen in enumerate(sketch_len):
            sketchInput = torch.cat((sketchInput, lineCode[ind:ind+slen,:]))
            ind += slen
            sketchInput = torch.cat((sketchInput, torch.zeros((maxlen-slen,self.lineHiddenNum)).cuda())) # padding
        sketchInput = sketchInput.view(sketchBatch, maxlen, -1)
        sketchInput = torch.transpose(sketchInput, 0, 1)

        sketch_len_eof = list(np.array(sketch_len) + 1) # add one because of eof
        lineCodeRecons, meanVar, logstdVar = self.sketchRnn(sketchInput, sketch_len_eof)

        # cat the lines in different batch 
        lineDecodeInput = torch.zeros((0)).cuda() # should be same size with lineCode
        endStrokeCode = torch.zeros((0)).cuda() # use for train eof of sketch
        for ind, slen in enumerate(sketch_len):
            lineDecodeInput = torch.cat((lineDecodeInput, lineCodeRecons[0:slen,ind,:]))
            endStrokeCode = torch.cat((endStrokeCode, lineCodeRecons[slen:slen+1, ind, :]))

        # import ipdb; ipdb.set_trace()
        outputVar = self.lineRnn.decode(lineDecodeInput, lineBatch, seqNum)

        return outputVar, endStrokeCode, LineMeanVar, LineLogstdVar, meanVar, logstdVar

    def get_high_params(self):
        return self.sketchRnn.parameters()

    def load_line_model(self, lineModelName):
        preTrainDict = torch.load(lineModelName)
        model_dict = self.state_dict()
        load_dict = {}
        for k,v in preTrainDict.items():
            load_dict['lineRnn.'+k] = v
        for item in load_dict:
            print('  Load pretrained layer: ',item )
        model_dict.update(load_dict)
        self.load_state_dict(model_dict)


if __name__ == '__main__':
    batchNum = 7
    inputNum = 5
    hiddenNum = 128
    hiddenNum2 = 64
    outputNum = 5
    batchNum = 3
    lineNum = [2, 5, 3]
    lineLen = range(15,15-sum(lineNum),-1)
    # rnn = StrokeRnn(inputNum, hiddenNum, outputNum)
    # rnn.cuda()
    # inputVar = torch.randn(seqNum, batchNum, inputNum).cuda()
    # outputVar = rnn(inputVar, range(seqNum,seqNum-batchNum,-1))

    # # test SketchRNN
    # rnn = SketchRnn(inputNum, hiddenNum, hiddenNum2, outputNum)
    # rnn.cuda()
    # maxLineLen = 20
    # inputVar = torch.randn(maxLineLen, sum(lineNum), inputNum).cuda()
    # outputVar, endStrokeCode, LineMeanVar, LineLogstdVar, meanVar, logstdVar = rnn(inputVar, lineLen, lineNum)


    # test load_line
    InputNum = 2
    HiddenNumLine = 512
    HiddenNumSketch = 256
    OutputNum = 2
    LineModel = 'models/2_4_sketchrnn_40000.pkl'
    sketchnet = SketchRnn(InputNum, HiddenNumLine, HiddenNumSketch, OutputNum)
    import ipdb; ipdb.set_trace() 
    sketchnet.load_line_model(LineModel)
    import ipdb; ipdb.set_trace() 

