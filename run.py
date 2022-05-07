# -*- coding: utf-8 -*-
from params import Params
from dataset import classification as dataset
from tools import units
from tools.save import save_experiment
from My_Model import My_Model
from My_model_for_QA import My_model_for_QA
import itertools
import argparse
import keras.backend as K
import time
import numpy as np

gpu_count = len(units.get_available_gpus())
dir_path, global_logger = units.getLogger()

def run(params, reader):
    now = int(time.time())
    timeArray = time.localtime(now)
    timeStamp = time.strftime("%Y%m%d_%H.%M.%S", timeArray)
    result_path = 'result/' + params.dataset_name + '/' + timeStamp + '.txt'
    f = open(result_path, 'a')
    f.write(params.to_string())
    f.close()

    params = dataset.process_embedding(reader, params)
    qdnn = My_Model(params)
    model = qdnn.getModel()

    # history = model.fit(x=train_x, y=train_y, batch_size=params.batch_size, epochs=params.epochs,
    #                     validation_data=(test_x, test_y))
    #
    # evaluation = model.evaluate(x=val_x, y=val_y)

    model.compile(loss='categorical_crossentropy',
                  optimizer=units.getOptimizer(name=params.optimizer, lr=params.lr),
                  metrics=['accuracy'])

    model.summary()
    (train_x, train_y), (test_x, test_y), (val_x, val_y) = reader.get_processed_data()

    # pretrain_x, pretrain_y = dataset.get_sentiment_dic_training_data(reader,params)
    # model.fit(x=pretrain_x, y = [pretrain_y,pretrain_y], batch_size = params.batch_size, epochs= 3,validation_data= (test_x, [test_y,test_y]))

    history = model.fit(x=train_x, y=train_y, batch_size=params.batch_size, epochs=params.epochs,
                        validation_data=(test_x, test_y))

    evaluation = model.evaluate(x=test_x,y=test_y)
    t=model.predict(x=test_x)
    np.save("sst_test",t)
    save_experiment(model, params, evaluation, history, reader)
    return history,evaluation

grid_parameters = {
    "dataset_name": ["MR"],
    "wordvec_path": ["glove/glove.6B.50d.txt"],
    # "glove/glove.6B.300d.txt"],"glove/normalized_vectors.txt","glove/glove.6B.50d.txt","glove/glove.6B.100d.txt",
    "loss": ["categorical_crossentropy"],  # "mean_squared_error"],,"categorical_hinge"
    "optimizer": ["rmsprop"],  # "adagrad","adamax","nadam"],,"adadelta","adam"
    "batch_size": [16],  # ,32
    "nums_states":[5],
    "activation": ["sigmoid"],
    "embedding_l2": [0],  # 0.0000005,0.0000001,
    "dense_l2": [0],  # 0.0001,0.00001,0],
    "epochs": [1],
    "lr": [1],  # ,1,0.01
    "nb_classes": [2],
    "dropout_rate_embedding": [0],  # 0.5,0.75,0.8,0.9,1],
    "ablation": [1],
}

if __name__ == "__main__":

    # import argparse
    parser = argparse.ArgumentParser(description='running the complex embedding network')
    # parser.add_argument('-gpu_num', action = 'store', dest = 'gpu_num', help = 'please enter the gpu num.',default=gpu_count)
    parser.add_argument('-gpu_num', action='store', dest='gpu_num', help='please enter the gpu num.', default=1)
    parser.add_argument('-gpu', action='store', dest='gpu', help='please enter the gpu num.', default=0)
    args = parser.parse_args()

    parameters = [arg for index, arg in enumerate(itertools.product(*grid_parameters.values())) if
                  index % args.gpu_num == args.gpu]

    parameters = parameters[::-1]

    params = Params()
    config_file = 'config/qdnn.ini'  # define dataset in the config
    params.parse_config(config_file)
    for parameter in parameters:
        print(parameter)
        old_dataset = params.dataset_name
        params.setup(zip(grid_parameters.keys(), parameter))

        if old_dataset != params.dataset_name:
            print("switch {} to {}".format(old_dataset, params.dataset_name))
            reader = dataset.setup(params)
            params.reader = reader
        else:
            reader = dataset.setup(params)
            params.reader = reader
        #        params.print()
        #        dir_path,logger = units.getLogger()
        #        params.save(dir_path)
        history, evaluation = run(params, reader)
        # global_logger.info("{} : {:.4f} ".format( params.to_string() ,max(max=(history.history["val_loss1_acc"]),max=(history.history["val_loss2_acc"])) ))
        K.clear_session()


