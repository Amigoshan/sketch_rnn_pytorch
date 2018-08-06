import torch 
import torch.nn as nn
from utils import output_to_strokes

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
    def __init__(self, inputNum, hiddenNum, outputNum):
        super(SketchRnn, self).__init__()
        self.hiddenNum = hiddenNum
        self.encoder = nn.LSTM(inputNum, hiddenNum)
        self.mean = nn.Linear(hiddenNum, hiddenNum)
        self.logstd = nn.Linear(hiddenNum, hiddenNum)
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
    rnn = StrokeRnn(inputNum, hiddenNum, outputNum)
    rnn.cuda()
    inputVar = torch.randn(seqNum, batchNum, inputNum).cuda()
    outputVar = rnn(inputVar, range(seqNum,seqNum-batchNum,-1))

    # testing multiple batch using pack_padded_sequence
    # inputVar = torch.randn(seqNum, 10, inputNum)

    
    import ipdb; ipdb.set_trace() 