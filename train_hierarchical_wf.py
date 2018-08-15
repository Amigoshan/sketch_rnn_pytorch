import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np 
from os.path import join
import time
import matplotlib.pyplot as plt

from hierarchical_sketch_rnn import SketchRnn
from sketchDataHierarchy import SketchDatasetHierarchy

import visdom
import sys
sys.path.append('../WorkFlow')
from workflow import WorkFlow

exp_prefix = '4_2_'

Lr = 0.0002
LrDecrease = [40000]
Batch = 64
TestBatch = 5
Trainstep = 50000
Showiter = 10
Snapshot = 10000

# lambda_eof = 0.1
lambda_kl = 0.02
lambda_kl_line = 0.1

InputNum = 2
HiddenNumLine = 512
HiddenNumSketch = 512
OutputNum = 2
ClipNorm = 0.1
LoadPretrain = True
modelname = 'models/3_8_2_hierarchical_sketch_60000.pkl'
LoadLineModel = False
LineModel = 'models/2_4_sketchrnn_40000.pkl'
Visualization = False

datapath = './data'
filecat = 'sketchrnn_cat.npz'
saveModelName = 'hierarchical_sketch'


# TODO: 
# log params into log file
# save image snapshot

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

        self.sketchnet = SketchRnn(InputNum, HiddenNumLine, HiddenNumSketch, OutputNum)
        if LoadPretrain:
            self.sketchnet = self.load_model(self.sketchnet, modelname)
        if LoadLineModel:
            self.sketchnet.load_line_model(LineModel)

        self.sketchnet.cuda()

        self.criterion_mse = nn.MSELoss()
        self.optimizer = optim.Adam(self.sketchnet.parameters(), lr = Lr) #get_high_params(), lr = Lr) #,weight_decay=1e-5)

        # === Create the AccumulatedObjects. ===
        self.AV['loss'].avgWidth =  100
        self.add_accumulated_value("loss_cons", 100)
        self.add_accumulated_value("test_loss_cons")
        self.add_accumulated_value("loss_kl", 100)
        self.add_accumulated_value("loss_kl_line", 100)
        self.add_accumulated_value("loss_cons_high", 100)
        # self.add_accumulated_value("loss_eof", 100)
        self.add_accumulated_value("test_loss")

        # === Create a AccumulatedValuePlotter object for ploting. ===
        self.AVP.append(WorkFlow.VisdomLinePlotter(\
                "train_test_loss", self.AV, ['loss', 'test_loss'], [True, False]))

        self.AVP.append(WorkFlow.VisdomLinePlotter(\
                "loss_kl", self.AV, ["loss_kl", "loss_kl_line"], [True, True]))

        self.AVP.append(WorkFlow.VisdomLinePlotter(\
                "loss_cons", self.AV, ["loss_cons", "test_loss_cons"], [True, False]))

        self.AVP.append(WorkFlow.VisdomLinePlotter(\
                "loss_cons_high", self.AV, ["loss_cons_high"], [True]))

        # self.AVP.append(WorkFlow.VisdomLinePlotter(\
        #         "loss_eof", self.AV, ["loss_eof"], [True]))

    # Overload the function initialize().
    def initialize(self):
        super(MyWF, self).initialize()

        # === Custom code. ===
        self.logger.info("Initialized.")


    def compare_strokes(self, inputStroke, outputStroke, dataset):
        # import ipdb; ipdb.set_trace()
        fig = plt.figure(1, (20, 7))
        axis = fig.subplots(1, 2)
        start1, start2 = [0,0] , [0,0]
        for k in range(inputStroke.shape[1]):
        # visualize the trained lines:
            line1, start1 = dataset.returnPaddedLine(inputStroke[:,k,:], start1)
            line2, start2 = dataset.returnPaddedLine(outputStroke[:,k,:], start2)
            print inputStroke[:,k,:]
            print outputStroke[:,k,:]
            axis[0].plot(line1[:,0],0-line1[:,1],'o-')
            axis[1].plot(line2[:,0],0-line2[:,1],'o-')
        axis[0].set_ylim(-300, 300);axis[0].set_xlim(-400, 400)
        axis[0].grid()
        axis[1].set_ylim(-300, 300);axis[1].set_xlim(-400, 400)
        axis[1].grid()
        plt.show()

    # Overload the function train().
    def train(self):
        super(MyWF, self).train()

        # === Custom code. ===
        self.countTrain += 1

        sketchLines, sketchLinelen, sketchLinenum, sketchLinelenFlat = self.dataset.get_random_batch(Batch)
        inputVar = torch.transpose(torch.from_numpy(sketchLines), 0, 1)

        outputVar, endStrokeCode, linemean, linelogstd, mean, logstd, sketchInput, lineCodeRecons = self.sketchnet(inputVar.cuda(), sketchLinelenFlat, sketchLinenum)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # TODO: there could be too much zero that bias the training
        targetVar = torch.transpose(torch.from_numpy(sketchLines), 0, 1)

        loss_cons = self.criterion_mse(outputVar, targetVar.cuda())   
        loss_kl_line = (linelogstd.exp()+linemean.pow(2) - linelogstd - 1).mean()/2.0
        loss_cons_high = self.criterion_mse(lineCodeRecons, sketchInput.detach())
        loss_kl = (logstd.exp()+mean.pow(2) - logstd - 1).mean()/2.0
        # loss_eof = self.criterion_mse(endStrokeCode, torch.zeros_like(endStrokeCode).cuda()) 

        loss =  loss_cons_high + loss_kl * lambda_kl + loss_cons + loss_kl_line * lambda_kl_line #+ loss_eof * lambda_eof # 
        loss.backward()

        # torch.nn.utils.clip_grad_norm(sketchnet.parameters(), ClipNorm)
        for param in self.sketchnet.parameters(): #get_high_params():
            param.grad.clamp_(-ClipNorm, ClipNorm) 

        # import ipdb; ipdb.set_trace()
        self.optimizer.step()

        if Visualization:
            # visualization
            strokeInd = 0
            lineNum = sketchLinenum[strokeInd]
            self.compare_strokes(inputVar[:,0:lineNum,:].detach().cpu().numpy(), outputVar[:,0:lineNum,:].detach().cpu().numpy(), self.dataset)

        self.AV["loss"].push_back(loss.item())
        self.AV["loss_cons"].push_back(loss_cons.item())
        self.AV["loss_kl_line"].push_back(loss_kl_line.item())
        self.AV["loss_cons_high"].push_back(loss_cons_high.item())
        self.AV["loss_kl"].push_back(loss_kl.item())
        # self.AV["loss_eof"].push_back(loss_eof.item())

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

        outputVar, endStrokeCode, _, _, mean, logstd, sketchInput, lineCodeRecons = self.sketchnet(inputVar.cuda(), sketchLinelenFlat, sketchLinenum)

        targetVar = torch.transpose(torch.from_numpy(sketchLines), 0, 1)

        loss_cons = self.criterion_mse(outputVar, targetVar.cuda())   
        loss_kl = (logstd.exp()+mean.pow(2) - logstd - 1).mean()/2.0
        # loss_eof = self.criterion_mse(endStrokeCode, torch.zeros_like(endStrokeCode).cuda()) 
        loss_cons_high = self.criterion_mse(lineCodeRecons, sketchInput.detach())
        # loss =  loss_cons + loss_kl * lambda_kl + loss_eof * lambda_eof # 
        loss =  loss_cons_high + loss_kl * lambda_kl

        self.AV["test_loss_cons"].push_back(loss_cons.item(), self.countTrain)
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
            if wf.countTrain in LrDecrease:
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

