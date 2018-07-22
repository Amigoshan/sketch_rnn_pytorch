import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from os.path import join
import time

from sketchRNN import SketchRnn 
from sketchData import SketchDataset
from utils import loadPretrain, drawFig, to_normal_strokes, output_to_strokes

exp_prefix = '1_1_'

Lr = 0.0001
Batch = 1
Trainstep = 50000
Showiter = 1
Snapshot = 10000

InputNum = 5
HiddenNum = 512
OutputNum = 5

exp_name = exp_prefix+'sketchrnn'
paramName = 'models/'+ exp_name
modelname = 'models/1_3_sketchrnn_100000.pkl'
LoadPretrain = True

datapath = '/home/wenshan/datasets/quickdraw'
filecat = 'sketchrnn_cat.npz'
imgoutdir = 'resimg'
datadir = 'logdata'

with np.load(join(datapath, filecat)) as cat_data:
    train_cat, val_cat, test_cat = cat_data['train'], cat_data['valid'], cat_data['test']

dataset = SketchDataset(train_cat, batch_size=Batch)
dataset.normalize()

sketchnet = SketchRnn(InputNum, HiddenNum, OutputNum)
if LoadPretrain:
    sketchnet = loadPretrain(sketchnet, modelname)
sketchnet.cuda()

criterion_mse = nn.MSELoss(size_average=False)
criterion_ce = nn.CrossEntropyLoss(weight=torch.Tensor([1,10,100]).cuda(), size_average=False)

count = 0
lossplot_cons = []
lossplot_kl = []
lossplot = []
running_loss_cons = 0.0
running_loss_kl = 0.0
running_loss = 0.0

while True:
    count += 1

    (sample, targetStroke), seq_len = dataset.random_batch()
    inputVar = torch.from_numpy(sample)
    mean, logstd, outputVar = sketchnet(inputVar.cuda(), seq_len.tolist(), testing=True)#, use_gt=True)

    # import ipdb; ipdb.set_trace()

    outStroke = output_to_strokes(outputVar.detach().cpu().numpy())
    # outStroke: seq x batch x outputNum
    outStroke = outStroke.reshape((seq_len[0], Batch, OutputNum))

    targetVar = nn.utils.rnn.pack_padded_sequence(inputVar[1:,:,:], seq_len, batch_first=False) # first in sequence is S0 used by decoder
    targetVar = targetVar.data

    # import ipdb; ipdb.set_trace()
    loss_cons = criterion_mse(outputVar[:,0:2], targetVar[:,0:2].cuda())   
    loss_stroke = criterion_ce(outputVar[:,2:5], torch.LongTensor(targetStroke).cuda())  
    # loss_kl = ((std*std+mean*mean)/2 - std.log() - 0.5).sum()
    loss_kl = (logstd.exp()+mean.pow(2) - logstd - 1).sum()/2.0

    print 'loss-cons:', loss_cons.item(), 'loss-stroke:', loss_stroke.item(), 'loss-kl:', loss_kl.item()
    # # zero the parameter gradients
    # targetVar = nn.utils.rnn.pack_padded_sequence(inputVar[1:,:,:], seq_len, batch_first=False) # first in sequence is S0 used by decoder
    # targetVar = targetVar.data

    # # import ipdb;ipdb.set_trace()

    # loss_cons = criterion_mse(outputVar, targetVar.cuda()) 
    # loss_kl = (logstd.exp()+mean.pow(2) - logstd - 1).sum()/2.0
    # loss =  loss_cons + loss_kl  #

    # visualize the output
    small_stroke = to_normal_strokes(sample[:,0,:])
    sample_denorm = dataset.denormalize(small_stroke)
    drawFig(sample_denorm)

    small_stroke = to_normal_strokes(outStroke[:,0,:])
    sample_denorm = dataset.denormalize(small_stroke)
    small_stroke[-1,-1] = 1
    drawFig(sample_denorm)

    # running_loss_cons += loss_cons.item()
    # running_loss_kl += loss_kl.item()
    # running_loss += loss.item()
    # if count % Showiter == 0:    # print every 20 mini-batches
    #     timestr = time.strftime('%m/%d %H:%M:%S',time.localtime())
    #     print(exp_prefix[0:-1] + ' [%d %s] loss: %.5f, cons_loss: %.5f, kl_loss: %.5f, lr: %f' %
    #     (count , timestr, running_loss / Showiter, running_loss_cons / Showiter, 
    #         running_loss_kl / Showiter, Lr))
    #     running_loss = 0.0
    #     running_loss_cons = 0.0
    #     running_loss_kl = 0.0
    # lossplot.append(loss.item())
    # lossplot_kl.append(loss_kl.item())
    # lossplot_cons.append(loss_cons.item())

    if count>=Trainstep:
        break

