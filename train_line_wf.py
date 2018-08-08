from workflow import WorkFlow
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
modelname = 'models/2_3_sketchrnn_40000.pkl'

exp_name = exp_prefix+'sketchrnn'
paramName = 'models/'+ exp_name

datapath = './data'
filecat = 'sketchrnn_cat.npz'
imgoutdir = 'resimg'
datadir = 'logdata'


# Template for custom WorkFlow object.
class MyWF(WorkFlow.WorkFlow):
    def __init__(self, workingDir):
        super(MyWF, self).__init__(workingDir)

        # === Custom member variables. ===
        # self.countTrain = 0
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
        self.criterion_ce = nn.CrossEntropyLoss(weight=torch.Tensor([1,10,100]).cuda(), size_average=Bidirection)
        self.optimizer = optim.Adam(self.sketchnet.parameters(), lr = Lr) #,weight_decay=1e-5)

        # === Create the AccumulatedObjects. ===
        self.AV['loss'].avgWidth =  100
        self.add_accumulated_value("loss_cons", 100)
        self.add_accumulated_value("loss_kl")
        self.add_accumulated_value("test_loss")

        # === Create a AccumulatedValuePlotter object for ploting. ===
        self.AVP.append(WorkFlow.VisdomLinePlotter(\
                "train_test_loss", self.AV, ['loss', 'test_loss'], [True, False]))

        self.AVP.append(WorkFlow.VisdomLinePlotter(\
                "loss_cons", self.AV, ["loss_cons"], [True]))

        self.AVP.append(WorkFlow.VisdomLinePlotter(\
                "loss_kl", self.AV, ["loss_kl"], [True]))

    # Overload the function initialize().
    def initialize(self):
        super(MyWF, self).initialize()

        # === Custom code. ===
        self.logger.info("Initialized.")

    # Overload the function train().
    def train(self):
        super(MyWF, self).train()

        # === Custom code. ===
        sketchLines, sketchLinelen, sketchLinenum, sketchLinelenFlat = self.dataset.get_random_batch(Batch)
        inputVar = torch.transpose(torch.from_numpy(sketchLines), 0, 1)
        # sketchLinelen = [item for sublist in sketchLinelen for item in sublist]
        outputVar, mean, logstd= self.sketchnet(inputVar.cuda(), sketchLinelenFlat)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # TODO: there could be too much zero that bias the training
        targetVar = torch.transpose(torch.from_numpy(sketchLines), 0, 1)

        loss_cons = criterion_mse(outputVar, targetVar.cuda())   
        loss_kl = (logstd.exp()+mean.pow(2) - logstd - 1).mean()/2.0
        loss_loc = criterion_mse(outputVar[0,:,:], targetVar[0,:,:].cuda()) * 2

        loss =  loss_cons + loss_kl + loss_loc  # 
        loss.backward()



        self.AV["loss"].push_back(math.sin( self.countTrain*0.1 ), self.countTrain*0.1)
        self.AV["loss_cons"].push_back(math.cos( self.countTrain*0.1 ), self.countTrain*0.1)
        self.AV["loss_kl"].push_back(math.cos( self.countTrain*0.1 ), self.countTrain*0.1)
        self.AV["test_loss"].push_back(math.cos( self.countTrain*0.1 ), self.countTrain*0.1)


        if ( self.countTrain % Snapshot == 0 ):
            self.write_accumulated_values()
            self.save_model()

        self.countTrain += 1

        # Plot accumulated values.
        self.plot_accumulated_values()


    # Overload the function test().
    def test(self):
        super(MyWF, self).test()

        # === Custom code. ===

        self.logger.info("Train #%d - loss: %.5f, cons_loss: %.5f , kl_loss: %.5f, lr: %f" \
            % self.countTrain, self.AV['lost'].last(), self.AV['loss_cons'].last, 
            self.AV['kl_loss'].last(), Lr )


        # # Test the existance of an AccumulatedValue object.
        # if ( True == self.have_accumulated_value("lossTest") ):
        #     self.AV["lossTest"].push_back(0.01, self.countTest)
        # else:
        #     self.logger.info("Could not find \"lossTest\"")

        # self.logger.info("Tested.")

    # Overload the function finalize().
    def finalize(self):
        super(MyWF, self).finalize()

        # === Custom code. ===
        self.logger.info("Finalized.")

if __name__ == "__main__":
    print("Hello WorkFlow.")

    self.print_delimeter(title = "Before initialization.")

    try:
        # Instantiate an object for MyWF.
        wf = MyWF("./")
        wf.verbose = True

        # Initialization.
        self.print_delimeter(title = "Initialize.")
        wf.initialize()

        # Training loop.
        self.print_delimeter(title = "Loop.")

        for i in range(100):
            wf.train()

        # Test and finalize.
        self.print_delimeter(title = "Test and finalize.")

        wf.test()
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

