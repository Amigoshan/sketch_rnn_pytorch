import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from os.path import join
import time
from utils import loadPretrain
import matplotlib.pyplot as plt

from hierarchical_sketch_rnn import StrokeRnn
from sketchDataHierarchy import SketchDatasetHierarchy
from torch.utils.data import Dataset, DataLoader
# from utils import to_normal_strokes, output_to_strokes, drawFig, loadPretrain

import visdom

exp_prefix = '2_3_'

Lr = 0.001
Batch = 32
Trainstep = 40000
Showiter = 10
Snapshot = 10000
Visiter = 2000
Bidirection = True

InputNum = 2
HiddenNum = 512
OutputNum = 2
ClipNorm = 0.1
LoadPretrain = True
modelname = 'models/2_2_sketchrnn_40000.pkl'

exp_name = exp_prefix+'sketchrnn'
paramName = 'models/'+ exp_name

datapath = './data'
filecat = 'sketchrnn_cat.npz'
imgoutdir = 'resimg'
datadir = 'logdata'


with np.load(join(datapath, filecat)) as cat_data:
    train_cat, val_cat, test_cat = cat_data['train'], cat_data['valid'], cat_data['test']

dataset = SketchDatasetHierarchy(test_cat)
# dataloader = DataLoader(dataset, batch_size=Batch, shuffle=True, num_workers=2)
# dataiter = iter(dataloader)

sketchnet = StrokeRnn(InputNum, HiddenNum, OutputNum)
if LoadPretrain:
    sketchnet = loadPretrain(sketchnet, modelname)
sketchnet.cuda()

criterion_mse = nn.MSELoss()
# criterion_ce = nn.CrossEntropyLoss(weight=torch.Tensor([1,10,100]).cuda(), size_average=Bidirection)
optimizer = optim.Adam(sketchnet.parameters(), lr = Lr) #,weight_decay=1e-5)

#initialize visualization
vis = visdom.Visdom(env=exp_name, server='http://localhost', port=8097)
loss_win = vis.line(X=np.array([-1]), Y=np.array([0]),
                         opts=dict(xlabel='steps', ylabel='loss', title=exp_prefix[0:-1]+'loss'))
loss_cons_win = vis.line(X=np.array([-1]), Y=np.array([0]),
                        opts=dict(xlabel='steps', ylabel='loss', title=exp_prefix[0:-1]+'cons loss'))
loss_loc_win = vis.line(X=np.array([-1]), Y=np.array([0]),
                        opts=dict(xlabel='steps', ylabel='loss', title=exp_prefix[0:-1]+'loc loss'))
loss_kl_win = vis.line(X=np.array([-1]), Y=np.array([0]),
                        opts=dict(xlabel='steps', ylabel='loss', title=exp_prefix[0:-1]+'kl loss'))

count = 0
lossplot_cons = []
lossplot_loc = []
lossplot_kl = []
lossplot = []
running_loss_cons = 0.0
running_loss_loc = 0.0
running_loss_kl = 0.0
running_loss = 0.0

while True:
    count += 1
    # import ipdb; ipdb.set_trace()
    sketchLines, sketchLinelen, sketchLinenum, sketchLinelenFlat = dataset.get_random_batch(Batch)
    # strokePadded, sketchLineLength, sketchLineNum = dataiter.next()


    # inputVar = strokePadded[0, 0:sketchLineNum[0],:].cuda()
    inputVar = torch.transpose(torch.from_numpy(sketchLines), 0, 1)

    # sketchLinelen = [item for sublist in sketchLinelen for item in sublist]
    outputVar, mean, logstd= sketchnet(inputVar.cuda(), sketchLinelenFlat)

    # zero the parameter gradients
    optimizer.zero_grad()
    # TODO: there could be too much zero that bias the training
    targetVar = torch.transpose(torch.from_numpy(sketchLines), 0, 1)

    # import ipdb; ipdb.set_trace()
    loss_cons = criterion_mse(outputVar, targetVar.cuda())   
    # loss_kl = ((std*std+mean*mean)/2 - std.log() - 0.5).sum()
    loss_kl = (logstd.exp()+mean.pow(2) - logstd - 1).mean()/2.0

    loss_loc = criterion_mse(outputVar[0,:,:], targetVar[0,:,:].cuda()) * 10

    loss =  loss_cons + loss_kl + loss_loc  # 
    loss.backward()

    # torch.nn.utils.clip_grad_norm(sketchnet.parameters(), ClipNorm)
    for param in sketchnet.parameters():
        param.grad.clamp_(-ClipNorm, ClipNorm) 

    optimizer.step()

    # visualize the trained lines:
    for k in range(inputVar.size(0)):

        inputLine = inputVar[:,k,:].cpu().numpy()
        print inputLine
        line1 = dataset.returnPaddedLine(inputLine)
        outputLine = outputVar[:,k,:].detach().cpu().numpy()
        print outputLine
        line2 = dataset.returnPaddedLine(outputLine)

        fig = plt.figure(1, (20, 5))
        axis = fig.subplots(1, 2)

        axis[0].plot(line1[:,0],0-line1[:,1],'o-')
        axis[0].set_ylim(-200, 200);axis[0].set_xlim(-400, 400)
        axis[0].grid()
        axis[1].plot(line2[:,0],0-line2[:,1],'o-')
        axis[1].set_ylim(-200, 200);axis[1].set_xlim(-400, 400)
        axis[1].grid()
        plt.show()


    running_loss_cons += loss_cons.item()
    running_loss_loc += loss_loc.item()
    running_loss_kl += loss_kl.item()
    running_loss += loss.item()
    if count % Showiter == 0:    # print every 20 mini-batches
        timestr = time.strftime('%m/%d %H:%M:%S',time.localtime())
        print(exp_prefix[0:-1] + ' [%d %s] loss: %.5f, cons_loss: %.5f %5f, kl_loss: %.5f, lr: %f' %
        (count , timestr, running_loss / Showiter, running_loss_cons / Showiter, running_loss_loc / Showiter, 
            running_loss_kl / Showiter, Lr))
        running_loss = 0.0
        running_loss_cons = 0.0
        running_loss_loc = 0.0
        running_loss_kl = 0.0

    lossplot.append(loss.item())
    lossplot_kl.append(loss_kl.item())
    lossplot_cons.append(loss_cons.item())
    lossplot_loc.append(loss_loc.item())

    vis.line(X=np.array([count]), Y=np.array([loss.item()]), win=loss_win, update='append')
    vis.line(X=np.array([count]), Y=np.array([loss_cons.item()]), win=loss_cons_win, update='append')
    vis.line(X=np.array([count]), Y=np.array([loss_loc.item()]), win=loss_loc_win, update='append')
    vis.line(X=np.array([count]), Y=np.array([loss_kl.item()]), win=loss_kl_win, update='append')

    if (count)%Snapshot==0:
        torch.save(sketchnet.state_dict(), paramName+'_'+str(count)+'.pkl')
        np.save(join(datadir,exp_prefix+'lossklplot.npy'), lossplot_kl)
        np.save(join(datadir,exp_prefix+'lossconsplot.npy'), lossplot_cons)
        np.save(join(datadir,exp_prefix+'losslocplot.npy'), lossplot_loc)
        np.save(join(datadir,exp_prefix+'lossplot.npy'), lossplot)


    if count>=Trainstep:
        break

    # update Learning Rate
    if count==30000 or count==37000:
        Lr = Lr*0.2
        for param_group in optimizer.param_groups:
            param_group['lr'] = Lr

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

