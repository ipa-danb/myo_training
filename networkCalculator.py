class structure_config:
    def __init__(self):
        self.model_params = { 'model':'CNN' # or CNN

                        }
        self.convFilterDict = { "conv":  {
                                    "filters":       30,
                                    "kernel_size":   (1,8),    # size of the filters: [3-8]
                                    "input_shape":   (1,8,1)  # input shape adapted
                                    },
                           "maxpool":{"pool_size":(1,5)},  # calculated automatically
                           "structure":{
                                    'hidden_layer':      1,
                                    'neurons_per_layer': 100,
                                    'filter_layer':      100
                                                 }
                         }
        self.augmentation = { "shift_list": [-1,1,3], # [start, end, number of points] list of values for shifting
                         "noisesigma": 0.2 # sigma value of gaussian noise // currently not used
                       }

        self.networkfile_path = '/home/myo/network_tests/stuff220170930-125105_final' # legacy, not used anymore

        self.output_classes = 4
        self.activation = 'relu'

class data_conf:
    def __init__(self):
        self.data_categories = ['empty','screwdriver','hammer','weightP']
        self.data_persons = ['daniel1','tobias','jan','jascha','chris1']


def updateprinter(current,maximum,message):
    print("\n\n############################################################################")
    print("## {0:^70} ##".format(""))
    print("## {0:^70} ##".format("<{0}/{1}> {2} ".format(current,maximum,message)) )
    print("## {0:^70} ##".format(""))
    print("############################################################################\n\n")


from helperfunctions import *
from sklearn.metrics import confusion_matrix

class Trainer:

    def __init__(self,path=['emg_data','ipa_emg','severalObjects'],data_conf=data_conf(),norm=(normalization_none,()),crossvalType='interdata'):
        """
        CrossvalTypes: {'interdata','intersample'}
        """
        self.dataDict = {'interdata':['interdata','interperson'] , 'intersample':['sample','intersample'] , 'none':['none'] }
        self.path = path
        self.data_conf = data_conf
        self.norm = norm
        self.normalization = norm[0]
        self.args = norm[1]
        self.parseCrossval(crossvalType)

    def parseCrossval(self,cvType):
        if not type(cvType) is tuple:
            crossvalType = cvType
        else:
            crossvalType = cvType[0]

        dList = [element for element in self.dataDict if crossvalType in self.dataDict[element]]
        if len(dList) > 0:
            self.crossvalType = dList[0]
        else:
            self.crossvalType = 'none'

    def prepareData(self):
        self.loadData()
        self.prepareCrossvalidation()

    def setDataConf(self,data_conf):
        self.data_conf = data_conf

    def loadData(self):
        updateprinter(1,5,"Loading Data")

        dataList = glob.glob(os.path.join(os.path.expanduser('~'),*self.path,'**'),recursive=True)

        # create a dictionary with filepaths respective to persons and categories
        fileNameDict = dict()
        categories =  self.data_conf.data_categories
        persons = self.data_conf.data_persons

        for p in persons:
            fileNameDict[p] = dict()
            for c in categories:
                fileNameDict[p][c] = selectFileNames(selectFileNames(dataList,p),c)

        # read data from files and put it into a respective data dictionary as pandas frame
        fileDict = dict()
        for p in fileNameDict:
            fileDict[p] = dict()
            for f in fileNameDict[p]:
                tmpList = list()
                for k in fileNameDict[p][f]:
                    tmpList.append(pd.read_csv(os.path.join(os.path.expanduser('~'),*self.path,k), skiprows=1, header=None, delim_whitespace=True))
                fileDict[p][f] = np.array(pd.concat(tmpList)[[1,2,3,4,5,6,7,8]])

        self.fileDict = fileDict

        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(fileNameDict)

        # normalize the data
        self.fileDict = self.normalization(self.fileDict,self.args)

    def prepareCrossvalidation(self):

        fold_dict = dict()
        fileDict = self.fileDict
        cats = self.data_conf.data_categories

        if self.crossvalType == 'interdata':
            updateprinter(2,5,"Preparing data for interdata crossvalidation Data")
            for mr,p in enumerate(fileDict):
                tmpX = list() # training input
                tmpY = list() # training output
                tmpXTest = list() # test input
                tmpYTest = list() # test output

                print("[{0}/{1}] start with interperson dataset creation leaving out {2}".format(mr+1,len(fileDict),p))

                for p2 in fileDict:
                    if p2 != p:
                        print("----- adding {0:7} to {1:5} set".format(p2,'train'))
                        for j,f in enumerate(fileDict[p2]):
                            tmpX.append( np.array(fileDict[p2][f]) )
                            tmpY.append(cats.index(f)*np.ones((len(fileDict[p2][f]),1) ) )

                    else:
                        print("----- adding {0:7} to {1:5} set".format(p2,'test'))
                        for j,f in enumerate(fileDict[p2]):
                            tmpXTest.append( np.array(fileDict[p2][f]) )
                            tmpYTest.append(cats.index(f)*np.ones((len(fileDict[p2][f]),1) ) )


                tmpXTest = np.vstack(tmpXTest)
                tmpYTest = np.vstack(tmpYTest)
                tmpX = np.vstack(tmpX)
                tmpY = np.vstack(tmpY)

                fold_dict[p] = dict()
                fold_dict[p]['train'] = [tmpX,tmpY]
                fold_dict[p]['test']  = [tmpXTest,tmpYTest]
                print("----------------------------------------------------")

        else:
            tmpx = list()
            tmpy = list()

            for person in fileDict:
                for label in fileDict[person]:
                    # create
                    tmpx.append(np.array(fileDict[person][label]))
                    tmpy.append(cats.index(label)*np.ones((len(fileDict[person][label]),1) ) )

            tmpx_ar = np.vstack(tmpx)
            tmpy_ar = np.vstack(tmpy)

            if self.crossvalType == 'intersample':

                updateprinter(2,5,"Preparing data for intersample crossvalidation Data")

                from sklearn.model_selection import KFold

                kf = KFold(n_splits=self.folds,shuffle=True)
                i=0
                for train_index, test_index in kf.split(tmpx_ar):
                    fold_dict[i] = dict()
                    print("----- adding {0:7} to {1:5} set".format(i,'train'))
                    fold_dict[i]['train'] = [tmpx_ar[train_index], tmpy_ar[train_index]]
                    print("----- adding {0:7} to {1:5} set".format(i,'test'))
                    fold_dict[i]['test']  = [tmpx_ar[test_index], tmpy_ar[test_index]]
                    i += 1

            else:

                updateprinter(2,5,"Preparing data for no crossvalidation")

                fold_dict["total"] = dict()
                fold_dict["total"]['train'] = [tmpx_ar, tmpy_ar]
                fold_dict['total']['test']  = [tmpx_ar, tmpy_ar]

        self.fold_dict = fold_dict

    def createTemplateModel(self,config):

        from keras.utils.np_utils import to_categorical
        from keras.layers.convolutional import Conv1D
        import keras
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Flatten, LocallyConnected2D
        from keras.layers import Conv2D, MaxPooling2D, LSTM, Conv1D, MaxPooling1D, ZeroPadding2D
        from keras.layers.normalization import BatchNormalization
        from keras.utils.vis_utils import model_to_dot
        from keras.callbacks import EarlyStopping

        #hidden_layers = max(0,int(self.arch[0]))
        #neurons_per_layer = max(1,int(self.arch[1]))
        #kernel_size = config.convFilterDict["conv"]["kernel_size"]
        #filter_layer = max(1,int(self.arch[3]))

        print("Architecture: ")
        print("Hidden Layers: ", config.convFilterDict["structure"]["hidden_layer"])
        print("Neurons:       ", config.convFilterDict["structure"]["neurons_per_layer"])
        print("ConvFilters:   ", config.convFilterDict["conv"]["kernel_size"])
        print("FilterLayer:   ", config.convFilterDict["conv"]["filters"])
        print("Inputshape:    ", config.convFilterDict["conv"]["input_shape"])

        config.convFilterDict["maxpool"]["pool_size"] = (1, config.convFilterDict["conv"]["input_shape"][1] - config.convFilterDict["conv"]["kernel_size"][1] + 1)

        # Network architecture
        output_classes    = config.output_classes
        activation        = config.activation

        model = Sequential()
        if config.model_params['model'] == 'CNN':
            model.add(Conv2D(**config.convFilterDict["conv"]))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(**config.convFilterDict["maxpool"]))
        elif config.model_params['model'] == 'MLP':
            model.add(Dense(neurons_per_layer,input_shape=config.convFilterDict["conv"]["input_shape"]))
        model.add(Activation(config.activation))
        model.add(Dropout(0.1))

        model.add(Flatten())

        for i in range(0,config.convFilterDict["structure"]["hidden_layer"]):
            model.add(Dense(config.convFilterDict["structure"]["neurons_per_layer"]))
            model.add(BatchNormalization())
            model.add(Activation(config.activation))
            model.add(Dropout(0.1))

        model.add(Dense(config.output_classes,activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], class_mode="sparse" )
        model.summary()
        return model

    def loadArchitecture(self,conf):
        """ legacy
        """
        import pickle

        try:
            architecture_list = pickle.load(open(conf.networkfile_path,'rb'))
            self.arch = architecture_list[0]
        except:
            self.arch = None
            print("using standard architecture")

    def saveNetworks(self,conf):
        import os
        import datetime

        # create directory
        dir_path,_ = os.path.split(os.path.realpath(__file__))
        p = os.path.join(dir_path,datetime.datetime.now().strftime("%Y%m%d%H%M"))
        os.makedirs(p,exist_ok=True)

        keylist = list(self.fold_dict.keys())

        for i,element in enumerate(self.histo):
            filePath = os.path.join(p,str(keylist[i]))
            element[2].save(filePath)

        with open(os.path.join(p,"config.txt"),"w") as f:
            s = self.createStringFromConfig(conf)
            f.write(s)

    def createStringFromConfig(self,conf):
        str = "model : {0}\nconvFilter Dict: \n filters:{1}, \n kernelsize: {2} \n inputshape: {3} \n maxpool: {4} \naugmentation: \n shiftlist: {5} \n noisesigma: {6} \nnetworkfile: {7} \noutput_classes: {8} \nactivation: {9} \n Datalist: {10}".format(conf.model_params['model'],conf.convFilterDict["conv"]["filters"],conf.convFilterDict["conv"]["kernel_size"],conf.convFilterDict["conv"]["input_shape"],conf.convFilterDict["maxpool"]["pool_size"],conf.augmentation["shift_list"],conf.augmentation["noisesigma"],conf.networkfile_path, conf.output_classes,conf.activation,self.data_conf.data_persons)
        return str

    def training(self,conf,epos=500,delta=0.005,patRatio=0.1):

        from keras.utils.np_utils import to_categorical
        from keras.layers.convolutional import Conv1D
        import keras
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Flatten, LocallyConnected2D
        from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, ZeroPadding2D
        from keras.layers.normalization import BatchNormalization
        from keras.utils.vis_utils import model_to_dot
        from keras.callbacks import EarlyStopping
        import copy

        class AccuracyHistory(keras.callbacks.Callback):
            def on_train_begin(self,logs={}):
                self.accuracy = []
            def on_epoch_end(self,epoch,logs={}):
                self.accuracy.append(logs.get('acc'))

        # ... in these training period
        pat = int(patRatio*epos)

        updateprinter(3,5,"Start with data perperation")

        inputlength = conf.convFilterDict["conv"]["input_shape"][1] # length of input vectors
        shiftvec = list(np.linspace(*conf.augmentation["shift_list"]))    # values for shift - list
        outlength = conf.output_classes


        working_fold_dict = copy.deepcopy(self.fold_dict)

        for c in self.fold_dict:
            # prepare training data
            tt = self.fold_dict[c]['train']
            x,y = augmentData2(tt[0],tt[1],shiftvec,inputlength)
            y = to_categorical(y,num_classes=outlength)
            working_fold_dict[c]['train'] = [expandVectors(x),y]

            # prepare test data
            tt2 = self.fold_dict[c]['test']
            x2,y2 = augmentData2(tt2[0],tt2[1],[0],inputlength)
            working_fold_dict[c]['test'] = [expandVectors(x2), to_categorical(y2,num_classes=outlength)]
            print("done with preparing data of " + str(c))

        #updateprinter(2,4,"Try to load network architecture")

        #self.loadArchitecture(conf)

        updateprinter(4,5,"Start training networks")

        histo = list()
        for c in working_fold_dict:
            x_train = working_fold_dict[c]['train'][0]
            y_train = working_fold_dict[c]['train'][1]
            x_test = working_fold_dict[c]['test'][0]
            y_test = working_fold_dict[c]['test'][1]

            early_stopping = EarlyStopping(monitor='val_loss',patience=pat,min_delta=delta)
            history = AccuracyHistory()
            model = self.createTemplateModel(conf)

            # dont change batch_size as this is limited by graphics card memory
            model.fit(x_train,y_train,epochs = epos, batch_size = 2000, validation_split=0.2, shuffle=True, callbacks = [early_stopping,history], verbose=1)
            sc = model.evaluate(x_test,y_test, batch_size=2000, verbose=0)

            histo.append( [sc,history,model] )
            print(" Done with " + str(c) + " with accuracy of " + str(sc[1]))

        updateprinter(5,5,"Evaluate network performance")

        y0 = list()
        y1 = list()
        for i,c in enumerate(working_fold_dict):
            y0.append(histo[i][2].predict(working_fold_dict[c]['test'][0],batch_size=2000))
            y1.append(working_fold_dict[c]['test'][1])

        y0_stacked = np.vstack(y0)
        y1_stacked = np.vstack(y1)

        rr_pred  = np.argmax(y0_stacked,axis=1)
        rr_true  = np.argmax(y1_stacked,axis=1)
        rr_pred2 = np.copy(rr_pred)
        rr_true2 = np.copy(rr_true)

        rr_pred[rr_pred == 3] = 1
        rr_true[rr_true == 3] = 1
        rr_true[rr_true == 2] = 1
        rr_pred[rr_pred == 2] = 1

        aa  = confusion_matrix(rr_true,rr_pred)
        aa2 = confusion_matrix(rr_true2,rr_pred2)

        print("Performance: {:.2%}".format( sum( [i[0][1] for i in histo] )/len(histo)  ) )
        self.histo = histo
        return [aa,aa2,histo]
