from helperfunctions import *
import networkCalculator
import copy
import pickle

config_file_name = "config_std_norm.conf"

data_conf = networkCalculator.data_conf()
struct_conf = networkCalculator.structure_config()
dataList =  ['tobias','jascha','jan','daniel1','chris1']
outputFile_name = "networkperformance_log_is{0}_fr{1}_fn{2}_{3}.p".format(16,8,100,'std')

init_args = {"path":['grip_detection','datasets','ipa_emg','one_handed_grasps'],"crossvalType":'interdata',"norm":(normalization_std,('empty'))}

ll = list()
train_args = {"epos":400, "delta":0.005, "patRatio":0.1}
ll.append({'struct_conf':copy.deepcopy(struct_conf),'data_conf': copy.deepcopy(data_conf),'train_args': copy.deepcopy(train_args), 'outputName': outputFile_name})
# TODO: description
ll[-1]['struct_conf'].convFilterDict["conv"]["input_shape"] = (1, 16, 1)
ll[-1]['struct_conf'].convFilterDict["conv"]["kernel_size"] = (1, 8)
ll[-1]['struct_conf'].convFilterDict["conv"]["filters"] = 100
ll[-1]['data_conf'].data_persons = dataList

gather_dict = {'init_args': init_args, 'trainlist': ll}

with open(config_file_name,"wb") as f:
    pickle.dump(gather_dict,f)
