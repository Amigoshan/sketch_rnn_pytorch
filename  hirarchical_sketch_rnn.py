import torch 
import torch.nn as nn
from utils import output_to_strokes

class SketchRnn(nn.Module):
    def __init__(self, inputNum, hiddenNum, outputNum, bidir = False):
        super(SketchRnn, self).__init__()
        self.hiddenNum = hiddenNum
        self.bidir = bidir
        if bidir:
            self.bidirScale = 2
        else:
            self.bidirScale = 1
        self.encoder = nn.LSTM(inputNum, hiddenNum, bidirectional=bidir)
        self.mean = nn.Linear(self.bidirScale*hiddenNum, hiddenNum)
        self.logstd = nn.Linear(self.bidirScale*hiddenNum, hiddenNum)
        self.decoderH = nn.Linear(hiddenNum, hiddenNum)
        self.decoderC = nn.Linear(hiddenNum, hiddenNum)
        self.decoder = nn.LSTM(inputNum+hiddenNum, hiddenNum) # the input is combined with the coding
        self.output = nn.Linear(hiddenNum, outputNum)

    def forward(self, x, seq_len, testing=False, use_gt=False):
        """
        x:       network input, should be in shape (seq, batch, inputNum)
        seq_len: a list indicating the length of each sequence in the batch
        testing: the decoder use previous output as next input
        """
        (seqNum, batchNum, inputNum) = x.size()    
        seqNum -= 1     
        # import ipdb; ipdb.set_trace()

        x_pack = nn.utils.rnn.pack_padded_sequence(x[1:,:,:], seq_len, batch_first=False) # first in sequence is S0 used by decoder

        _, (hn, _) = self.encoder(x_pack, self.init_hidden(batchNum)) # hn: 1 x batch x hidden
        if self.bidir:
            hn = torch.cat((hn[0,:,:], hn[1,:,:]), dim=1)
        meanVar = self.mean(hn.view(batchNum, -1)) # batch x hidden
        logstdVar = self.logstd(hn.view(batchNum, -1)) # batch x hidden

        # sample from the mean and logstd
        epsilon = torch.normal(mean=torch.zeros_like(meanVar))
        sample = meanVar + (logstdVar/2).exp() * epsilon # batch x hidden

        # hiddenInput = torch.tanh(self.aftersample(sample))
        hn = torch.tanh(self.decoderH(sample)).view(1,batchNum,-1) 
        cn = torch.tanh(self.decoderC(sample)).view(1,batchNum,-1) 
        if not testing:
            # shift the input one time step behind and concate with the coding sample
            decoderInput = x[0:-1,:,:]
            decoderInput = torch.cat((decoderInput, sample.detach().expand((seqNum, batchNum, self.hiddenNum))), dim=2)
            dx_pack = nn.utils.rnn.pack_padded_sequence(decoderInput, seq_len, batch_first=False) 
            outputVar, _ = self.decoder(dx_pack, (hn, cn))
            outputVar = self.output(outputVar.data) # outputVar is in same order with x_pack

        else: # if testing, the outputVar is already converted into one-hot stroke, and can not be backpropagated!
            # import ipdb; ipdb.set_trace()
            outlist = []
            decoderInput = x[0:1,:,:]
            decoderInput = torch.cat((decoderInput, sample.view((1, batchNum, self.hiddenNum))), dim=2) # 1 x batch x hidden
            hidden = (hn, cn)
            # if testing, input the data in sequence one by one
            for k in range(seqNum):
                outputVar, hidden = self.decoder(decoderInput, hidden)
                outputVar = self.output(outputVar.view(-1, self.hiddenNum))
                # calculate the stroke type using last three column 
                outStroke = output_to_strokes(outputVar.detach().cpu().numpy())
                outStroke = torch.from_numpy(outStroke).cuda()

                outlist.append(outStroke)
                if not use_gt:
                    decoderInput = torch.cat((outStroke.view((1, batchNum, 5)), sample.view((1, batchNum, self.hiddenNum))), dim=2)
                else: 
                    decoderInput = torch.cat((x[k+1:k+2,:,:].view((1, batchNum, 5)), sample.view((1, batchNum, self.hiddenNum))), dim=2)
            outputVar = torch.cat(outlist, dim=0) # (seq x batch) x ouput

        # import ipdb; ipdb.set_trace()
        return meanVar, logstdVar, outputVar


    def init_hidden(self, batchNum):
        return torch.zeros(self.bidirScale, batchNum, self.hiddenNum).cuda(), \
                torch.zeros(self.bidirScale, batchNum, self.hiddenNum).cuda()


    def calc_loss(outputVar, targetVar):
        pass


if __name__ == '__main__':
    batchNum = 7
    inputNum = 5
    hiddenNum = 128
    outputNum = 5
    seqNum = 10
    rnn = SketchRnn(inputNum, hiddenNum, outputNum)
    inputVar = torch.randn(seqNum+1, batchNum, inputNum)
    mean, logstd, outputVar = rnn(inputVar, range(seqNum,seqNum-batchNum,-1), testing=True)

    # testing multiple batch using pack_padded_sequence
    inputVar = torch.randn(seqNum, 10, inputNum)

    
    # import ipdb; ipdb.set_trace()