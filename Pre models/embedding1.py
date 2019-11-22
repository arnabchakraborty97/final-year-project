#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:50:02 2019

@author: rcciit
"""

#importing all necessary modules
from keras.models import Model,load_model
from keras.layers import LSTM,Input,Dense,Embedding,Reshape,TimeDistributed,Dropout
from keras.layers.merge import concatenate
import numpy as np
from keras.utils.np_utils import to_categorical

vocab_size=len(ad_mat)  #vocabulary size, that is, number of unique nodes

'''def cat_enc(lists,n_classes):
    cat_mat=np.zeros(shape=(batch_size,n_classes))
    for i in range(len(lists)):
        cat_mat[i,np.array(lists[i])]=1
    return np.array(cat_mat)
'''        

'''Model Definition Step'''
#defining model components
visible1=Input(shape=(1,1))
visible2=Input(shape=(2,1))
visible3=Input(shape=(2,1))
emb=Embedding(vocab_size,128)
reshape_layer1=Reshape((1,128))
reshape_layer2=Reshape((1,256))
reshape_layer3=Reshape((1,256))
encoder=LSTM(256,return_state=True)
decoder1=LSTM(256,return_sequences=True)
decoder2=LSTM(256,return_sequences=True)
tdense1=TimeDistributed(Dense(500))
dense1=Dense(500)
dense2=Dense(7343,activation='softmax')

#defining model workflow
emb_out1=emb(visible1)
reshape_out1=reshape_layer1(emb_out1)
enc_out,state_h,state_c=encoder(reshape_out1)
states=[state_h,state_c]
emb_out2=emb(visible2)
reshape_out2=reshape_layer2(emb_out2)
dec_out1=decoder1(reshape_out2,initial_state=states)
drop_out1=Dropout(0.1)
emb_out3=emb(visible3)
reshape_out3=reshape_layer3(emb_out3)
dec_out2=decoder2(reshape_out3,initial_state=states)
drop_out2=Dropout(0.1)
tdense_out1=tdense1(dec_out1)
tdense_out2=tdense1(dec_out2)
merged_out=concatenate([tdense_out1,tdense_out2])
dense_out=dense1(merged_out)
output=dense2(dense_out)


#create the model
model=Model(input=[visible1,visible2,visible3],outputs=output)
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
print(model.summary())


'''Data Preparation Step'''
#input and context generation
current_input=[]
right_context=[]
left_context=[]
context_size=2

for i, value in enumerate(inputs):
    current_input.append(value)
    if i!=0:                    #if present input is not the first input
        from_pos=i-context_size
        if from_pos < 0:
            left_context.append(list(inputs[0:i]))
        else:
            left_context.append(list(inputs[from_pos:i]))
    else:
        left_context.append([])
    if i!=(len(inputs)-1):      #if present input is not the last input
        to_pos=i+context_size+1
        if to_pos > len(inputs):
            right_context.append(list(inputs[i+1:len(inputs)]))
        else:
            right_context.append(list(inputs[i+1:to_pos]))
    else:
        right_context.append([])

#removing the first two and last two inputs and contexts in order to keep uniform size        
current_input=current_input[context_size:len(current_input)-context_size]
left_context=left_context[context_size:len(left_context)-context_size]
right_context=right_context[context_size:len(right_context)-context_size]


'''Training Step'''

tmp_input=[]
tmp_left=[]
tmp_right=[]
tmp_merge=[]
n_output=7343
batch_size=len(current_input)
inc=0

for i in range(len(current_input)/batch_size):
        print(i)
        tmp_input=current_input[inc:inc+batch_size]
        tmp_left=left_context[inc:inc+batch_size]
        tmp_right=right_context[inc:inc+batch_size]
        tmp_merge=np.hstack((tmp_left,tmp_right))
        y = cat_enc(np.array(tmp_merge), n_output)
        y = np.reshape(y,(batch_size,1,n_output))
        tmp_input=np.reshape(tmp_input,(batch_size,1,1))
        tmp_left=np.reshape(tmp_left,(batch_size,2,1))
        tmp_right=np.reshape(tmp_right,(batch_size,2,1))
        inc=inc+batch_size
        model.fit([tmp_input,tmp_left,tmp_right],y,validation_split=0.1,epochs=15,batch_size=1000,verbose=1)

model.save('my_model.h5')

#loading the model
trained_model=load_model('my_model.h5')
emb_weights=trained_model.layers[2].get_weights()
lstm_weights1=trained_model.layers[6].get_weights()[0]
emb_weights.append(lstm_weights1)
lstm_weights2=trained_model.layers[6].get_weights()[1]
emb_weights.append(lstm_weights2)
lstm_weights3=trained_model.layers[6].get_weights()[2]
emb_weights.append(lstm_weights3)
#emb_weights=[embedding weights, lstm weights, lstm unit weights, bias]

'''Prediction model structure'''
visible=Input(shape=(1,1))
pred_emb=Embedding(vocab_size,128)
reshape_layer=Reshape((1,128))
lstm1=LSTM(256)

'''Prediction model workflow'''
emb_out=pred_emb(visible)
reshape_out=reshape_layer(emb_out)
lstm_out=lstm1(reshape_out)

'''Creating the prediction model'''
pred_model=Model(input=visible,outputs=lstm_out)
pred_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
pred_model.set_weights(emb_weights)
print(pred_model.summary())

'''Preparing prediction data'''
pred_data=np.arange(vocab_size)
pred_data=np.reshape(pred_data,(len(pred_data),1,1))

preds = pred_model.predict(pred_data)




'''
tmp_input=[]
tmp_left=[]
tmp_right=[]
y=[]
n_output=7343

for epoch in range(10):
        #print(i)
        tmp_input=current_input
        tmp_left=left_context
        tmp_right=right_context
        y = to_categorical(current_input, num_classes = n_output)
        y = np.reshape(y,(len(current_input),n_output))
        tmp_input=np.reshape(tmp_input,(240055,1,1))
        tmp_left=np.reshape(tmp_left,(240055,1,2))
        tmp_right=np.reshape(tmp_right,(240053,1,2))
        model.fit([tmp_input,left_context,right_context],y,epochs=1,batch_size=500,verbose=1)
'''