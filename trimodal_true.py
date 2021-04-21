# ===================================================================================================
# ============================================= original ============================================
# ===================================================================================================
import numpy as np, json
import pickle, sys, argparse
import keras
from keras.models import Model
from keras import backend as K
from keras import initializers
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
from keras.layers import *
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score
global seed
seed = 1337
np.random.seed(seed)
import math
import glob
import gc
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import itertools
#=============================================================
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#=============================================================
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
set_session(tf.Session(config=config))
#=============================================================
def sarcasm_classification_performance(prediction, test_label):

    true_label=[]
    predicted_label=[]

    for i in range(test_label.shape[0]):
        true_label.append(np.argmax(test_label[i]))
        predicted_label.append(np.argmax(prediction[i]))

    accuracy      = accuracy_score(true_label, predicted_label)
    prfs_weighted = precision_recall_fscore_support(true_label, predicted_label, average='weighted')

    return accuracy, prfs_weighted


def attention(x, y):
    m_dash = dot([x, y], axes=[2,2])
    m = Activation('softmax')(m_dash)
    h_dash = dot([m, y], axes=[2,1])
    return multiply([h_dash, x])

def divisorGen(n):
    factors = list(factorGenerator(n))
    nfactors = len(factors)
    f = [0] * nfactors
    while True:
        yield reduce(lambda x, y: x*y, [factors[x][0]**f[x] for x in range(nfactors)], 1)
        i = 0
        while True:
            f[i] += 1
            if f[i] <= factors[i][1]:
                break
            f[i] = 0
            i += 1
            if i >= nfactors:
                return

def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor

def featuresExtraction_fastext(foldNum, exMode):
    global train_cText, train_sentiment_cText_implicit, train_sentiment_cText_explicit, train_emotion_cText_implicit, train_emotion_cText_explicit, train_featureSpeaker_cText
    global test_cText, test_sentiment_cText_implicit, test_sentiment_cText_explicit, test_emotion_cText_implicit, test_emotion_cText_explicit, test_featureSpeaker_cText
    global train_uText, train_sentiment_uText_implicit, train_sentiment_uText_explicit, train_emotion_uText_implicit, train_emotion_uText_explicit, train_featureSpeaker_uText
    global test_uText, test_sentiment_uText_implicit, test_sentiment_uText_explicit, test_emotion_uText_implicit, test_emotion_uText_explicit, test_featureSpeaker_uText
    global train_length_cText, test_length_cText
    global train_mask_cText, test_mask_cText

    path = 'feature_extraction/dataset'+str(exMode)+'_fasttext/sarcasmDataset_speaker_dependent_'+str(exMode)+'_'+str(foldNum)+'.npz'
    data = np.load(path, mmap_mode='r')
    # =================================================================
    train_sentiment_cText_implicit = data['train_sentiment_cText_implicit']
    train_sentiment_cText_explicit = data['train_sentiment_cText_explicit']
    train_emotion_cText_implicit   = data['train_emotion_cText_implicit']
    train_emotion_cText_explicit   = data['train_emotion_cText_explicit']
    train_featureSpeaker_cText     = data['train_featureSpeaker_cText']
    # =================================================================
    test_emotion_cText_implicit    = data['test_emotion_cText_implicit']
    test_emotion_cText_explicit    = data['test_emotion_cText_explicit']
    test_sentiment_cText_implicit  = data['test_sentiment_cText_implicit']
    test_sentiment_cText_explicit  = data['test_sentiment_cText_explicit']
    test_featureSpeaker_cText      = data['test_featureSpeaker_cText']
    # =================================================================
    train_sentiment_uText_implicit = data['train_sentiment_uText_implicit']
    train_sentiment_uText_explicit = data['train_sentiment_uText_explicit']
    train_emotion_uText_implicit   = data['train_emotion_uText_implicit']
    train_emotion_uText_explicit   = data['train_emotion_uText_explicit']
    train_featureSpeaker_uText     = data['train_featureSpeaker_uText']
    # =================================================================
    test_emotion_uText_implicit    = data['test_emotion_uText_implicit']
    test_emotion_uText_explicit    = data['test_emotion_uText_explicit']
    test_sentiment_uText_implicit  = data['test_sentiment_uText_implicit']
    test_sentiment_uText_explicit  = data['test_sentiment_uText_explicit']
    test_featureSpeaker_uText      = data['test_featureSpeaker_uText']
    # =================================================================
    train_cText          = data['train_cText']
    train_cText          = np.array(train_cText)
    train_cText          = train_cText/np.max(abs(train_cText))
    # =================================================================
    train_uText          = data['train_uText']
    train_uText          = np.array(train_uText)
    train_uText          = train_uText/np.max(abs(train_uText))
    # =================================================================
    test_cText           = data['test_cText']
    test_cText           = np.array(test_cText)
    test_cText           = test_cText/np.max(abs(test_cText))
    # =================================================================
    test_uText           = data['test_uText']
    test_uText           = np.array(test_uText)
    test_uText           = test_uText/np.max(abs(test_uText))
    # =================================================================
    train_length_cText   = data['train_length_cText']
    test_length_cText    = data['test_length_cText']
    # ===========================================================================================
    train_mask_cText = np.zeros((train_cText.shape[0], train_cText.shape[1]), dtype='float16')
    test_mask_cText  = np.zeros((test_cText.shape[0], test_cText.shape[1]), dtype='float16')

    for i in range(len(train_length_cText)):
        train_mask_cText[i,:train_length_cText[i]] = 1.0

    for i in range(len(test_length_cText)):
        test_mask_cText[i,:test_length_cText[i]] = 1.0


def featuresExtraction_original(foldNum, exMode):
    global train_cText, train_uText, train_cVisual, train_uVisual, train_uAudio, train_length_CT, train_sarcasm_label, train_mask_CT
    global test_cText, test_uText, test_cVisual, test_uVisual, test_uAudio, test_length_CT, test_sarcasm_label, test_mask_CT
    global train_cText, train_uText, train_cVisual, train_uVisual, train_uAudio
    global test_cText, test_uText, test_cVisual, test_uVisual, test_uAudio

    sarcasm = np.load('feature_extraction/dataset'+str(exMode)+'_original/sarcasmDataset_speaker_dependent_'+ exMode +'.npz', mmap_mode='r', allow_pickle=True)
    train_cText       = sarcasm['feautesCT_train'][foldNum]
    train_cText       = np.array(train_cText)
    train_cText       = train_cText/np.max(abs(train_cText))
    # ======================================================
    train_uText       = sarcasm['feautesUT_train'][foldNum]
    train_uText       = np.array(train_uText)
    train_uText       = train_uText/np.max(abs(train_uText))
    # ======================================================
    train_uAudio      = sarcasm['feautesUA_train'][foldNum]
    train_uAudio      = np.array(train_uAudio)
    train_uAudio      = train_uAudio/np.max(abs(train_uAudio))
    # ======================================================
    train_cVisual     = sarcasm['feautesCV_train'][foldNum]
    train_cVisual     = np.array(train_cVisual)
    train_cVisual     = train_cVisual/np.max(abs(train_cVisual))
    # ======================================================
    train_uVisual     = sarcasm['feautesUV_train'][foldNum]
    train_uVisual     = np.array(train_uVisual)
    train_uVisual     = train_uVisual/np.max(abs(train_uVisual))
    # ======================================================
    test_cText        = sarcasm['feautesCT_test'][foldNum]
    test_cText        = np.array(test_cText)
    test_cText        = test_cText/np.max(abs(test_cText))
    # ======================================================
    test_uText        = sarcasm['feautesUT_test'][foldNum]
    test_uText        = np.array(test_uText)
    test_uText        = test_uText/np.max(abs(test_uText))
    # ======================================================
    test_uAudio       = sarcasm['feautesUA_test'][foldNum]
    test_uAudio       = np.array(test_uAudio)
    test_uAudio       = test_uAudio/np.max(abs(test_uAudio))
    # ======================================================
    test_cVisual      = sarcasm['feautesCV_test'][foldNum]
    test_cVisual      = np.array(test_cVisual)
    test_cVisual      = test_cVisual/np.max(abs(test_cVisual))
    # ======================================================
    test_uVisual      = sarcasm['feautesUV_test'][foldNum]
    test_uVisual      = np.array(test_uVisual)
    test_uVisual      = test_uVisual/np.max(abs(test_uVisual))
    # ======================================================

    train_length_CT = sarcasm['train_length_CT'][foldNum]
    test_length_CT  = sarcasm['test_length_CT'][foldNum]

    train_sarcasm_label = sarcasm['feautesLabel_train'][foldNum]
    test_sarcasm_label  = sarcasm['feautesLabel_test'][foldNum]

    train_mask_CT = np.zeros((train_cText.shape[0], train_cText.shape[1]), dtype='float')
    test_mask_CT  = np.zeros((test_cText.shape[0], test_cText.shape[1]), dtype='float')

    for i in range(len(train_length_CT)):
        train_mask_CT[i,:train_length_CT[i]] = 1.0

    for i in range(len(test_length_CT)):
        test_mask_CT[i,:test_length_CT[i]] = 1.0

def multiTask_multimodal(mode, filePath, drops=[0.7, 0.5, 0.5], r_units=300, td_units=100, numSplit=8, foldNum=0, exMode='True'):

    global tempAcc, tempP, tempR, tempF, tempAcc_is, tempP_is, tempR_is, tempF_is, tempAcc_es, tempP_es, tempR_es, tempF_es, tempAcc_ie, tempP_ie, tempR_ie, tempF_ie, tempAcc_ee, tempP_ee, tempR_ee, tempF_ee
    tempAcc =[]
    tempP   =[]
    tempR   =[]
    tempF   =[]

    runs = 1
    best_accuracy = 0
    drop0  = drops[0]
    drop1  = drops[1]
    r_drop = drops[2]

    for run in range(runs):
        # ===========================================================================================================================================
        in_uText        = Input(shape=(train_uText.shape[1], train_uText.shape[2]), name='in_uText')
        rnn_uText_T     = Bidirectional(GRU(r_units, return_sequences=True, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat', name='rnn_uText_T')(in_uText)
        td_uText_T      = Dropout(drop1)(TimeDistributed(Dense(td_units, activation='relu'))(rnn_uText_T))
        attn_uText      = attention(td_uText_T, td_uText_T)
        rnn_uText_F     = Bidirectional(GRU(r_units, return_sequences=False, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat', name='rnn_uText_F')(attn_uText)
        td_uText        = Dropout(drop1)(Dense(td_units, activation='relu')(rnn_uText_F))
        # ===========================================================================================================================================
        in_uVisual      = Input(shape=(train_uVisual.shape[1],), name='in_uVisual')
        td_uVisual      = Dropout(drop1)(Dense(td_units, activation='relu')(in_uVisual))
        # ===========================================================================================================================================
        in_uAudio       = Input(shape=(train_uAudio.shape[1],), name='in_uAudio')
        td_uAudio       = Dropout(drop1)(Dense(td_units, activation='relu')(in_uAudio))
        print('td_uText: ',td_uText.shape)

        # ===========================================================================================================================================
        # =================================== internal attention (multimodal attention) =============================================================
        # ===========================================================================================================================================
        if td_uVisual.shape[1]%numSplit == 0:
            td_text   = Lambda(lambda x: K.reshape(x, (-1, int(int(x.shape[1])/numSplit),numSplit)))(td_uText)
            td_visual = Lambda(lambda x: K.reshape(x, (-1, int(int(x.shape[1])/numSplit),numSplit)))(td_uVisual)
            td_audio  = Lambda(lambda x: K.reshape(x, (-1, int(int(x.shape[1])/numSplit),numSplit)))(td_uAudio)
            print('td_text: ',td_text.shape)
            print('td_visual: ',td_visual.shape)
            print('td_audio: ',td_audio.shape)

            intAttn_tv = attention(td_text, td_visual)
            intAttn_ta = attention(td_text, td_audio)
            intAttn_vt = attention(td_visual, td_text)
            intAttn_va = attention(td_visual, td_audio)
            intAttn_av = attention(td_audio, td_visual)
            intAttn_at = attention(td_audio, td_text)

            intAttn = concatenate([intAttn_tv, intAttn_ta, intAttn_vt, intAttn_va, intAttn_av, intAttn_at], axis=-1)
            print('intAttn: ', intAttn.shape)

        else:
            print('chose numSplit from '+ str(list(map(int, divisorGenerator(int(td_uVisual.shape[1])))))+'')
            return

        # ===========================================================================================================================================
        # =================================== external attention (self attention) ===================================================================
        # ===========================================================================================================================================
        extCat  = concatenate([td_text, td_visual, td_audio], axis=-1)
        extAttn = attention(extCat, extCat)
        print(extAttn.shape)
        # ===========================================================================================================================================
        merge_inAttn_extAttn  = concatenate([td_text, td_visual, td_audio, intAttn, extAttn], axis=-1)
        merge_inAttn_extAttn = Dropout(drop1)(Dense(td_units, activation='relu')(merge_inAttn_extAttn))
        print(merge_inAttn_extAttn.shape)
        # ===========================================================================================================================================
        merge_rnn  = Bidirectional(GRU(r_units, return_sequences=False, dropout=r_drop, recurrent_dropout=r_drop), merge_mode='concat', name='merged_rnn')(merge_inAttn_extAttn)
        merge_rnn = Dropout(drop1)(Dense(td_units, activation='relu')(merge_rnn))
        print(merge_rnn.shape)
        # ===========================================================================================================================================
        output_sarcasm = Dense(2, activation='softmax', name='output_sarcasm')(merge_rnn) # print('output_sarcasm: ',output_sarcasm.shape)
        # ===========================================================================================================================================
        output_senti_implicit = Dense(3, activation='softmax', name='output_senti_implicit')(merge_rnn) # print('output_senti_implicit: ',output_senti_implicit.shape)
        # ===========================================================================================================================================
        output_senti_explicit = Dense(3, activation='softmax', name='output_senti_explicit')(merge_rnn) # print('output_senti_explicit: ',output_senti_explicit.shape)
        # ===========================================================================================================================================
        output_emo_implicit = Dense(9, activation='sigmoid', name='output_emo_implicit')(merge_rnn) # print('output_emo_implicit: ',output_emo_implicit.shape)
        # ===========================================================================================================================================
        output_emo_explicit = Dense(9, activation='sigmoid', name='output_emo_explicit')(merge_rnn) # print('output_emo_explicit: ',output_emo_explicit.shape)
        # ===========================================================================================================================================
        model = Model(inputs=[in_uText, in_uAudio, in_uVisual],
                      outputs=[output_sarcasm, output_senti_implicit, output_senti_explicit, output_emo_implicit, output_emo_explicit])
        model.compile(loss={'output_sarcasm':'categorical_crossentropy',
                            'output_senti_implicit':'categorical_crossentropy',
                            'output_senti_explicit':'categorical_crossentropy',
                            'output_emo_implicit':'binary_crossentropy',
                            'output_emo_explicit':'binary_crossentropy'},
                      sample_weight_mode='None',
                      optimizer='adam',
                      metrics={'output_sarcasm':'accuracy',
                               'output_senti_implicit':'accuracy',
                               'output_senti_explicit':'accuracy',
                               'output_emo_implicit':'accuracy',
                               'output_emo_explicit':'accuracy'})
        print(model.summary())

        ###################### model training #######################
        np.random.seed(1)

        path = 'weights/sarcasm_speaker_dependent_wse_'+exMode+'_without_context_and_speaker_'+str(filePath)+', run: '+str(run)+'.hdf5'

        earlyStop_sarcasm = EarlyStopping(monitor='val_output_sarcasm_loss', patience=30)
        bestModel_sarcasm = ModelCheckpoint(path, monitor='val_output_sarcasm_acc', verbose=1, save_best_only=True, mode='max')

#         history = model.fit([train_uText, train_uAudio, train_uVisual], [train_sarcasm_label,train_sentiment_uText_implicit, train_sentiment_uText_explicit,train_emotion_uText_implicit, train_emotion_uText_explicit],
#                             epochs=200,
#                             batch_size=32,
#                             # sample_weight=train_mask_CT,
#                             shuffle=True,
#                             callbacks=[earlyStop_sarcasm, bestModel_sarcasm],
#                             validation_data=([test_uText, test_uAudio, test_uVisual], [test_sarcasm_label,test_sentiment_uText_implicit, test_sentiment_uText_explicit,test_emotion_uText_implicit, test_emotion_uText_explicit]),
#                             verbose=1)

        model.load_weights(path)
        prediction = model.predict([test_uText, test_uAudio, test_uVisual])
        # np.ndarray.dump(prediction[3], open('results/sarcasm_'+str(filePath)+'_'+str(run)+'.np', 'wb'))

        # ============================== sarcasm =========================================
        performance = sarcasm_classification_performance(prediction[0], test_sarcasm_label)

        tempAcc.append(performance[0])
        tempP.append(performance[1][0])
        tempR.append(performance[1][1])
        tempF.append(performance[1][2])

        open('results/sarcasm_speaker_dependent_wse_'+exMode+'_without_context_and_speaker.txt', 'a').write(path +'\n'+
                                                                                                        '=============== sarcasm ===============\n' +
                                                                                                        'loadAcc: '+ str(performance[0]) + '\n' +
                                                                                                        'prfs_weighted: '+ str(performance[1]) + '\n'*2)



        ################### release gpu memory ###################
        K.clear_session()
        del model
        gc.collect()


global globalAcc, globalP, globalR, globalF, globalAcc_is, globalP_is, globalR_is, globalF_is, globalAcc_es, globalP_es, globalR_es, globalF_es, globalAcc_ie, globalP_ie, globalR_ie, globalF_ie, globalAcc_ee, globalP_ee, globalR_ee, globalF_ee
exMode = 'True'  # execution mode
for drop in [0.3]:
    for rdrop in [0.3]:
        for r_units in [300]:
            for td_units in [50]:
                for numSplit in [50]:
                    if exMode == 'True':
                        foldNums = [0,1,2,3,4]
                    else:
                        foldNums = [3]
                    globalAcc = []
                    globalP   = []
                    globalR   = []
                    globalF   = []
                    for foldNum in foldNums:
                        featuresExtraction_original(foldNum, exMode)
                        featuresExtraction_fastext(foldNum, exMode)
                        modalities = ['text','audio','video']
                        for i in range(1):
                            for mode in itertools.combinations(modalities, 3):
                                modality = '_'.join(mode)
                                print('\n',modality)
                                filePath  = modality + '_' + str(drop) + '_' + str(drop) + '_' + str(rdrop) + '_' + str(r_units) + '_' + str(td_units) + ', numSplit: ' + str(numSplit) + ', foldNum: ' + str(foldNum) + ', ' + str(exMode)
                                testtt = multiTask_multimodal(mode, filePath, drops=[drop, drop, rdrop], r_units=r_units, td_units=td_units,numSplit=numSplit,foldNum=foldNum,exMode=exMode)
                        globalAcc.append(tempAcc)
                        globalP.append(tempP)
                        globalR.append(tempR)
                        globalF.append(tempF)

                        globalAcc_is.append(tempAcc_is)
                        globalP_is.append(tempP_is)
                        globalR_is.append(tempR_is)
                        globalF_is.append(tempF_is)

                        globalAcc_es.append(tempAcc_es)
                        globalP_es.append(tempP_es)
                        globalR_es.append(tempR_es)
                        globalF_es.append(tempF_es)

                        globalAcc_ie.append(tempAcc_ie)
                        globalP_ie.append(tempP_ie)
                        globalR_ie.append(tempR_ie)
                        globalF_ie.append(tempF_ie)

                        globalAcc_ee.append(tempAcc_ee)
                        globalP_ee.append(tempP_ee)
                        globalR_ee.append(tempR_ee)
                        globalF_ee.append(tempF_ee)
                        open('results/sarcasm_speaker_dependent_wse_'+exMode+'_without_context_and_speaker.txt', 'a').write('-'*150 + '\n'*2)
                    open('results/sarcasm_speaker_dependent_wse_'+exMode+'_without_context_and_speaker.txt', 'a').write('#'*150 + '\n'*2)
                    open('results/sarcasm_speaker_dependent_wse_'+exMode+'_without_context_and_speaker_average.txt', 'a').write(filePath +'\n'+
                                                                                                                    'globalAcc: '+ str(np.mean(globalAcc,axis=0)) + '\n' +
                                                                                                                    'globalP: '+ str(np.mean(globalP,axis=0)) + '\n' +
                                                                                                                    'globalR: '+ str(np.mean(globalR,axis=0)) + '\n' +
                                                                                                                    'globalF: '+ str(np.mean(globalF,axis=0)) + '\n'*2)
                    
