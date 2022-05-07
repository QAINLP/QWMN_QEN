import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.layers import Input,Embedding,Activation,Flatten,Dense,Dropout,concatenate
from keras.models import Model
from My_layers.embedding import amplitude_embedding_layer
from My_layers.mixture import mixture
from My_layers.pure import pure
from My_layers.measurement import measurement
from My_layers.kronecker_product import kronecker_product
from My_layers.PartialTrace import PartialTrace
from My_layers.l2_normalization import L2Normalization
from My_layers.classical_mixture import classical_mixture
from My_layers.extract_diag import extract_diag
from My_layers.LRLT import LRLT
from keras import regularizers

class My_Model():
    def __init__(self,opt):
        self.opt=opt
        self.doc = Input(shape=(self.opt.max_sequence_length,), dtype='float32')
        self.embedding = amplitude_embedding_layer(np.transpose(self.opt.lookup_table),
                                                             None,
                                                             trainable=self.opt.embedding_trainable,
                                                             random_init=self.opt.random_init,
                                                             l2_reg=self.opt.embedding_l2)
        self.weight_embedding = Embedding(self.opt.lookup_table.shape[0], 1, trainable=True)
        self.l2_normalization=L2Normalization(axis=2)
        self.pure=classical_mixture()
        self.measurement=measurement(self.opt.nb_classes,self.opt.nums_states,return_state=False)
        self.kronecker=kronecker_product(self.opt.nb_classes)
        self.lrlt=LRLT(self.opt.nb_classes)
        self.partialtrace=PartialTrace(self.opt.nb_classes)
        self.dense1 = Dense(self.opt.nb_classes, activation=self.opt.activation,name='loss1', kernel_regularizer= regularizers.l2(self.opt.dense_l2))  # activation="sigmoid",
        self.dropout_embedding = Dropout(self.opt.dropout_rate_embedding)
        self.extract_diag=extract_diag()

    def build(self,state_type,pred_type):
        embedding=self.l2_normalization(self.embedding(self.doc))
        embedding=self.dropout_embedding(embedding)
        weight = Activation('softmax')(self.weight_embedding(self.doc))

        # mixture=self.pure([embedding,weight])
        # kronecker_product=self.kronecker(mixture)  #?,100,100

        #测试拟密度算子，RLT
        out_product_partial_transform=self.lrlt([embedding,weight])

        measured_state=self.measurement(out_product_partial_transform)
        subsystem=self.partialtrace(measured_state)
        #when using the PDP layer.
        # output=self.extract_diag(subsystem)
        #when using the FCL.
        #将subsystem进行faltten
        subsystem_flatten=Flatten()(subsystem)

        output=self.dense1(subsystem_flatten)

        model=Model(self.doc,output)
        return model

    def getModel(self):
        return self.build(state_type="sup",pred_type="PDP")

