import networkCalculator
from sklearn.metrics import confusion_matrix
import itertools
import pickle
import datetime
import time
import argparse
import pickle

# create new stuff
parser = argparse.ArgumentParser("Startup training script")
parser.add_argument('-f','--file', type=argparse.FileType('rb'), help="name of file that contains the protocol",required=True)
parser.add_argument('-v','--verbose', action="store_true", help="verbosity") # TODO implement
args = parser.parse_args()

data = pickle.load(args.file)

start = time.time()
trainer = networkCalculator.Trainer(**data['init_args'])


for i,element in enumerate(data['trainlist']):

    end = time.time()
    conf = networkCalculator.structure_config()
    conf = element['struct_conf']
    data_conf = element['data_conf']
    outputF_name = element['outputName']
    train_args = element['train_args']

    with open(outputF_name,"wb") as f:
        pickle.dump(str(datetime.datetime.now().strftime("%Y%m%d-%H%M")),f)

    print(conf.convFilterDict["conv"]["input_shape"])
    print(conf.convFilterDict["conv"]["kernel_size"])

    trainer.setDataConf(data_conf)
    trainer.prepareData()
    out = trainer.training(conf=conf,**train_args)

    trainer.saveNetworks(conf)

    kb = (conf.convFilterDict["conv"]["input_shape"],conf.convFilterDict["conv"]["kernel_size"])
    with open(outputF_name,"a+b") as f:
        pickle.dump((out[0],out[1],kb),f)

    print('----------------------------------------------------------------------------------------------')
    print('')
    print('Done with ' + str(element))
    print('')
    print('----------------------------------------------------------------------------------------------')
