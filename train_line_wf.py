from workflow import WorkFlow
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
from os.path import join
import time

from hierarchical_sketch_rnn import StrokeRnn 
from sketchDataHierarchy import SketchDatasetHierarchy
from torch.utils.data import Dataset, DataLoader
from utils import loadPretrain

Lr = 0.001
Batch = 1
Trainstep = 170000
Showiter = 10
Snapshot = 10000
Visiter = 2000
Bidirection = True

InputNum = 2
HiddenNum = 512
OutputNum = 2
ClipNorm = 0.5
LoadPretrain = False


datapath = '/home/wenshan/datasets/quickdraw'
filecat = 'sketchrnn_cat.npz'
imgoutdir = 'resimg'
datadir = 'logdata'

def print_delimeter(c = "=", n = 20, title = "", leading = "\n", ending = "\n"):
    d = [c for i in range(n/2)]

    if ( 0 == len(title) ):
        s = "".join(d) + "".join(d)
    else:
        s = "".join(d) + " " + title + " " + "".join(d)

    print("%s%s%s" % (leading, s, ending))



# Template for custom WorkFlow object.
class MyWF(WorkFlow.WorkFlow):
    def __init__(self, workingDir):
        super(MyWF, self).__init__(workingDir)

        # === Create the AccumulatedObjects. ===
        self.AV['loss'].avgWidth =  100
        self.add_accumulated_value("loss_cons", 100)
        self.add_accumulated_value("loss_kl")
        self.add_accumulated_value("test_loss")
        # This should raise an exception.
        # self.add_accumulated_value("loss")

        # === Create a AccumulatedValuePlotter object for ploting. ===
        # avNameList    = ["loss", "loss2", "lossLeap"]
        # avAvgFlagList = [  True,   False,      True ]
        self.AVP.append(WorkFlow.VisdomLinePlotter(\
                "train_test_loss", self.AV, ['loss', 'test_loss'], [True, False]))

        self.AVP.append(WorkFlow.VisdomLinePlotter(\
                "loss_cons", self.AV, ["loss_cons"], [True]))

        self.AVP.append(WorkFlow.VisdomLinePlotter(\
                "loss_kl", self.AV, ["loss_kl"], [True]))

        # === Custom member variables. ===
        # self.countTrain = 0
        # self.countTest  = 0
		with np.load(join(datapath, filecat)) as cat_data:
		    self.train_cat, self.val_cat, self.test_cat = cat_data['train'], cat_data['valid'], cat_data['test']

		self.dataset = SketchDatasetHierarchy(train_cat)
		# dataloader = DataLoader(dataset, batch_size=Batch, shuffle=True, num_workers=2)
		# dataiter = iter(dataloader)

		self.sketchnet = StrokeRnn(InputNum, HiddenNum, OutputNum)
		if LoadPretrain:
		    self.sketchnet = loadPretrain(self.sketchnet, modelname)
		self.sketchnet.cuda()

		self.criterion_mse = nn.MSELoss(size_average=True)
		self.criterion_ce = nn.CrossEntropyLoss(weight=torch.Tensor([1,10,100]).cuda(), size_average=Bidirection)
		self.optimizer = optim.Adam(self.sketchnet.parameters(), lr = Lr) #,weight_decay=1e-5)



    # Overload the function initialize().
    def initialize(self):
        super(MyWF, self).initialize()

        # === Custom code. ===

        self.logger.info("Initialized.")

    # Overload the function train().
    def train(self):
        super(MyWF, self).train()

        # === Custom code. ===
        self.logger.info("Train loop #%d" % self.countTrain)

        # Test the existance of an AccumulatedValue object.
        if ( True == self.have_accumulated_value("loss") ):
            self.AV["loss"].push_back(math.sin( self.countTrain*0.1 ), self.countTrain*0.1)
        else:
            self.logger.info("Could not find \"loss\"")

        # Directly access "loss2" without existance test.
        self.AV["loss2"].push_back(math.cos( self.countTrain*0.1 ), self.countTrain*0.1)

        # lossLeap.
        if ( self.countTrain % 10 == 0 ):
            self.AV["lossLeap"].push_back(\
                math.sin( self.countTrain*0.1 + 0.25*math.pi ),\
                self.countTrain*0.1)

        # testAvg.
        self.AV["testAvg1"].push_back( 0.5, self.countTrain )

        if ( self.countTrain < 50 ):
            self.AV["testAvg2"].push_back( self.countTrain, self.countTrain )
        else:
            self.AV["testAvg2"].push_back( 50, self.countTrain )

        if ( self.countTrain % 10 == 0 ):
            self.write_accumulated_values()

        self.countTrain += 1

        # Plot accumulated values.
        self.plot_accumulated_values()

        self.logger.info("Trained.")

        time.sleep(0.05)

    # Overload the function test().
    def test(self):
        super(MyWF, self).test()

        # === Custom code. ===
        # Test the existance of an AccumulatedValue object.
        if ( True == self.have_accumulated_value("lossTest") ):
            self.AV["lossTest"].push_back(0.01, self.countTest)
        else:
            self.logger.info("Could not find \"lossTest\"")

        self.logger.info("Tested.")

    # Overload the function finalize().
    def finalize(self):
        super(MyWF, self).finalize()

        # === Custom code. ===
        self.logger.info("Finalized.")

if __name__ == "__main__":
    print("Hello WorkFlow.")

    print_delimeter(title = "Before initialization.")

    try:
        # Instantiate an object for MyWF.
        wf = MyWF("/tmp/WorkFlowDir")
        wf.verbose = True

        # Initialization.
        print_delimeter(title = "Initialize.")
        wf.initialize()

        # Training loop.
        print_delimeter(title = "Loop.")

        for i in range(100):
            wf.train()

        # Test and finalize.
        print_delimeter(title = "Test and finalize.")

        wf.test()
        wf.finalize()

        # # Show the accululated values.
        # print_delimeter(title = "Accumulated values.")
        # wf.AV["loss"].show_raw_data()

        # print_delimeter()
        # wf.AV["lossLeap"].show_raw_data()

        # print_delimeter()
        # wf.AV["lossTest"].show_raw_data()
    except WorkFlow.WFException as e:
        print( e.describe() )

    print("Done.")

