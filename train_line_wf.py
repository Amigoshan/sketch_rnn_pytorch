import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np 
from os.path import join
import time
import matplotlib.pyplot as plt

from hierarchical_sketch_rnn import StrokeRnn
from sketchDataHierarchy import SketchDatasetHierarchy

import visdom
import sys
sys.path.append('../WorkFlow')
from workflow import WorkFlow

exp_prefix = '2_7_2_'

Lr = 0.001
Batch = 32
TestBatch = 5
Trainstep = 10000
Showiter = 10
Snapshot = 10000
# Visiter = 2000
# Bidirection = False

InputNum = 2
HiddenNum = 256
OutputNum = 2
ClipNorm = 0.1
LoadPretrain = True
modelname = 'models/2_7_sketch_line_30000.pkl'

lambda_loc = 1.0
lambda_kl = 0.3

datapath = './data'
filecat = 'sketchrnn_cat.npz'
saveModelName = 'sketch_line'


# Template for custom WorkFlow object.
class MyWF(WorkFlow.WorkFlow):
    def __init__(self, workingDir, prefix = "", suffix = ""):
        super(MyWF, self).__init__(workingDir, prefix, suffix)

        # === Custom member variables. ===
        self.countTrain = 0
        # self.countTest  = 0
        with np.load(join(datapath, filecat)) as cat_data:
            train_cat, val_cat, test_cat = cat_data['train'], cat_data['valid'], cat_data['test']

        self.dataset = SketchDatasetHierarchy(train_cat)
        self.valset = SketchDatasetHierarchy(val_cat)

        self.sketchnet = StrokeRnn(InputNum, HiddenNum, OutputNum)
        if LoadPretrain:
            self.sketchnet = self.load_model(self.sketchnet, modelname)
        self.sketchnet.cuda()

        self.criterion_mse = nn.MSELoss(size_average=True)
        # self.criterion_ce = nn.CrossEntropyLoss(weight=torch.Tensor([1,10,100]).cuda(), size_average=Bidirection)
        self.optimizer = optim.Adam(self.sketchnet.parameters(), lr = Lr) #,weight_decay=1e-5)

        # === Create the AccumulatedObjects. ===
        self.AV['loss'].avgWidth =  100
        self.add_accumulated_value("loss_cons", 100)
        self.add_accumulated_value("loss_kl", 100)
        self.add_accumulated_value("loss_loc", 100)
        self.add_accumulated_value("test_loss")

        # === Create a AccumulatedValuePlotter object for ploting. ===
        self.AVP.append(WorkFlow.VisdomLinePlotter(\
                "train_test_loss", self.AV, ['loss', 'test_loss'], [True, False]))

        self.AVP.append(WorkFlow.VisdomLinePlotter(\
                "loss_cons", self.AV, ["loss_cons", "loss_kl"], [True, True]))

        self.AVP.append(WorkFlow.VisdomLinePlotter(\
                "loss_cons", self.AV, ["loss_loc"], [True]))

    # Overload the function initialize().
    def initialize(self):
        super(MyWF, self).initialize()

        # === Custom code. ===
        self.logger.info("Initialized.")

    # Overload the function train().
    def train(self):
        super(MyWF, self).train()

        # === Custom code. ===
        self.countTrain += 1

        sketchLines, sketchLinelen, sketchLinenum, sketchLinelenFlat = self.dataset.get_random_batch(Batch)
        inputVar = torch.transpose(torch.from_numpy(sketchLines), 0, 1)
        # sketchLinelen = [item for sublist in sketchLinelen for item in sublist]
        outputVar, mean, logstd= self.sketchnet(inputVar.cuda(), sketchLinelenFlat)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # TODO: there could be too much zero that bias the training
        targetVar = torch.transpose(torch.from_numpy(sketchLines), 0, 1)

        loss_cons = self.criterion_mse(outputVar, targetVar.cuda())   
        loss_kl = (logstd.exp()+mean.pow(2) - logstd - 1).mean()/2.0
        loss_loc = self.criterion_mse(outputVar[0,:,:], targetVar[0,:,:].cuda()) * lambda_loc

        loss =  loss_cons + loss_kl * lambda_kl + loss_loc  # 
        loss.backward()

        # torch.nn.utils.clip_grad_norm(sketchnet.parameters(), ClipNorm)
        for param in self.sketchnet.parameters():
            param.grad.clamp_(-ClipNorm, ClipNorm) 

        self.optimizer.step()


        self.AV["loss"].push_back(loss.item())
        self.AV["loss_cons"].push_back(loss_cons.item())
        self.AV["loss_kl"].push_back(loss_kl.item())
        self.AV["loss_loc"].push_back(loss_loc.item())


        if ( self.countTrain % Snapshot == 0 ):
            self.write_accumulated_values()
            self.save_model(self.sketchnet, saveModelName+'_'+str(self.countTrain))

        # Plot accumulated values.
        self.plot_accumulated_values()

    # Overload the function test().
    def test(self):
        super(MyWF, self).test()

        # === Custom code. ===
        sketchLines, sketchLinelen, sketchLinenum, sketchLinelenFlat = self.valset.get_random_batch(TestBatch)
        inputVar = torch.transpose(torch.from_numpy(sketchLines), 0, 1)
        outputVar, mean, logstd= self.sketchnet(inputVar.cuda(), sketchLinelenFlat)

        targetVar = torch.transpose(torch.from_numpy(sketchLines), 0, 1)

        loss_cons = self.criterion_mse(outputVar, targetVar.cuda())   
        loss_kl = (logstd.exp()+mean.pow(2) - logstd - 1).mean()/2.0
        loss_loc = self.criterion_mse(outputVar[0,:,:], targetVar[0,:,:].cuda()) * lambda_loc

        loss =  loss_cons + loss_kl * lambda_kl + loss_loc  # 

        self.AV["test_loss"].push_back(loss.item(), self.countTrain)

        losslogstr = self.get_log_str()
        self.logger.info("%s #%d - %s lr: %.6f" % (exp_prefix[:-1], self.countTrain, losslogstr, Lr))


    # Overload the function finalize().
    def finalize(self):
        super(MyWF, self).finalize()

        # === Custom code. ===
        self.logger.info("Finalized.")

if __name__ == "__main__":
    print("Hello WorkFlow.")

    # wf.print_delimeter(title = "Before initialization.")

    try:
        # Instantiate an object for MyWF.
        wf = MyWF("./", prefix = exp_prefix)

        # Initialization.
        # wf.print_delimeter(title = "Initialize.")
        wf.initialize()

        # Training loop.
        # wf.print_delimeter(title = "Loop.")

        while True:
            wf.train()
            if wf.countTrain%Showiter==0:
                wf.test()

            if (wf.countTrain>Trainstep):
                break

            # update Learning Rate
            if wf.countTrain==1 or wf.countTrain==7000:
                Lr = Lr*0.2
                for param_group in wf.optimizer.param_groups:
                    param_group['lr'] = Lr

        # Test and finalize.
        # wf.print_delimeter(title = "Test and finalize.")

        # wf.test()
        wf.finalize()

        # # Show the accululated values.
        # self.print_delimeter(title = "Accumulated values.")
        # wf.AV["loss"].show_raw_data()

        # self.print_delimeter()
        # wf.AV["lossLeap"].show_raw_data()

        # self.print_delimeter()
        # wf.AV["lossTest"].show_raw_data()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")

