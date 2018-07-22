import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from os.path import join
import time

from sketchRNN import SketchRnn 
from sketchData import SketchDataset
from utils import to_normal_strokes, output_to_strokes, drawFig

exp_prefix = '1_3_'

Lr = 0.0001
Batch = 100
Trainstep = 200000
Showiter = 10
Snapshot = 10000
Visiter = 2000

InputNum = 5
HiddenNum = 512
OutputNum = 5
ClipNorm = 10.0

exp_name = exp_prefix+'sketchrnn'
paramName = 'models/'+ exp_name

datapath = '/home/wenshan/datasets/quickdraw'
filecat = 'sketchrnn_cat.npz'
imgoutdir = 'resimg'
datadir = 'logdata'

with np.load(join(datapath, filecat)) as cat_data:
    train_cat, val_cat, test_cat = cat_data['train'], cat_data['valid'], cat_data['test']

dataset = SketchDataset(train_cat, batch_size=Batch)
dataset.normalize()

sketchnet = SketchRnn(InputNum, HiddenNum, OutputNum)
sketchnet.cuda()

criterion_mse = nn.MSELoss(size_average=False)
criterion_ce = nn.CrossEntropyLoss(weight=torch.Tensor([1,10,100]).cuda(), size_average=False)
optimizer = optim.Adam(sketchnet.parameters(), lr = Lr) #,weight_decay=1e-5)

count = 0
lossplot_cons = []
lossplot_stroke = []
lossplot_kl = []
lossplot = []
running_loss_cons = 0.0
running_loss_stroke = 0.0
running_loss_kl = 0.0
running_loss = 0.0

while True:
    count += 1

    (sample, targetStroke), seq_len = dataset.random_batch()
    # (sample, targetStroke), seq_len = dataset.get_batch(range(Batch))
    inputVar = torch.from_numpy(sample)
    mean, logstd, outputVar = sketchnet(inputVar.cuda(), seq_len.tolist())

    # zero the parameter gradients
    optimizer.zero_grad()
    targetVar = nn.utils.rnn.pack_padded_sequence(inputVar[1:,:,:], seq_len, batch_first=False) # first in sequence is S0 used by decoder
    targetVar = targetVar.data

    # import ipdb; ipdb.set_trace()
    loss_cons = criterion_mse(outputVar[:,0:2], targetVar[:,0:2].cuda())   
    loss_stroke = criterion_ce(outputVar[:,2:5], torch.LongTensor(targetStroke).cuda())  
    # loss_kl = ((std*std+mean*mean)/2 - std.log() - 0.5).sum()
    loss_kl = (logstd.exp()+mean.pow(2) - logstd - 1).sum()/2.0
    # loss_kl = (std.log()+(1+mean*mean)/(2*std*std) - 0.5).mean()
    loss =  loss_cons + loss_stroke + loss_kl  # 
    loss.backward()

    # torch.nn.utils.clip_grad_norm(sketchnet.parameters(), ClipNorm)
    for param in sketchnet.parameters():
        param.grad.clamp_(-ClipNorm, ClipNorm) 

    optimizer.step()

    running_loss_cons += loss_cons.item()
    running_loss_stroke += loss_stroke.item()
    running_loss_kl += loss_kl.item()
    running_loss += loss.item()
    if count % Showiter == 0:    # print every 20 mini-batches
        timestr = time.strftime('%m/%d %H:%M:%S',time.localtime())
        print(exp_prefix[0:-1] + ' [%d %s] loss: %.5f, cons_loss: %.5f %.5f, kl_loss: %.5f, lr: %f' %
        (count , timestr, running_loss / Showiter, running_loss_cons / Showiter, running_loss_stroke / Showiter,
            running_loss_kl / Showiter, Lr))
        running_loss = 0.0
        running_loss_cons = 0.0
        running_loss_stroke = 0.0
        running_loss_kl = 0.0

        # print '  ',
        # for param in sketchnet.parameters():
        #     print '%.1f %.1f' % (torch.mean(param.grad).item(), torch.std(param.grad).item()),
        # print ''

    # if count % Visiter == 0:
    #     # import ipdb; ipdb.set_trace()
    #     # viualize the input output
    #     small_stroke = to_normal_strokes(sample[:,0,:])
    #     sample_denorm = dataset.denormalize(small_stroke)
    #     drawFig(sample_denorm)

    #     outStroke = output_to_strokes(outputVar.detach().cpu().numpy())
    #     outStroke = outStroke.reshape((seq_len[0], Batch, OutputNum)) # TODO: can not handle batch with various seq length
    #     small_stroke = to_normal_strokes(outStroke[:,0,:])
    #     sample_denorm = dataset.denormalize(small_stroke)
    #     small_stroke[-1,-1] = 1
    #     drawFig(sample_denorm)

    #     # viualize the input output
    #     small_stroke = to_normal_strokes(sample[:,10,:])
    #     sample_denorm = dataset.denormalize(small_stroke)
    #     drawFig(sample_denorm)

    #     small_stroke = to_normal_strokes(outStroke[:,10,:])
    #     sample_denorm = dataset.denormalize(small_stroke)
    #     small_stroke[-1,-1] = 1
    #     drawFig(sample_denorm)

    lossplot.append(loss.item())
    lossplot_kl.append(loss_kl.item())
    lossplot_cons.append(loss_cons.item())
    lossplot_stroke.append(loss_stroke.item())

    if (count)%Snapshot==0:
        torch.save(sketchnet.state_dict(), paramName+'_'+str(count)+'.pkl')
        np.save(join(datadir,exp_prefix+'lossklplot.npy'), lossplot_kl)
        np.save(join(datadir,exp_prefix+'lossconsplot.npy'), lossplot_cons)
        np.save(join(datadir,exp_prefix+'lossstrokeplot.npy'), lossplot_stroke)
        np.save(join(datadir,exp_prefix+'lossplot.npy'), lossplot)


    if count>=Trainstep:
        break

import matplotlib.pyplot as plt
group = 10
lossplot = np.array(lossplot)
lossplot_cons = np.array(lossplot_cons)
lossplot_kl = np.array(lossplot_kl)
if len(lossplot)%group>0:
    lossplot = lossplot[0:len(lossplot)/group*group]
    lossplot_cons = lossplot_cons[0:len(lossplot_cons)/group*group]
    lossplot_kl = lossplot_kl[0:len(lossplot_kl)/group*group]
lossplot = lossplot.reshape((-1,group))
lossplot = lossplot.mean(axis=1)
lossplot_cons = lossplot_cons.reshape((-1,group))
lossplot_cons = lossplot_cons.mean(axis=1)
lossplot_kl = lossplot_kl.reshape((-1,group))
lossplot_kl = lossplot_kl.mean(axis=1)
plt.plot(lossplot)
plt.plot(lossplot_cons)
plt.plot(lossplot_kl)

plt.grid()
plt.savefig(join(imgoutdir, exp_name+'.png'))
# plt.ylim([0,0.04])
plt.show()
import ipdb; ipdb.set_trace()

