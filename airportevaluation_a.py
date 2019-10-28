import nltk
import gensim
import re

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

import numpy as np
from numpy import asarray

from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Embedding, LSTM,Convolution1D,MaxPooling1D,Flatten,Activation,Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras import regularizers
from keras.optimizers import Adamax,Adadelta,adam,rmsprop,SGD
from keras.losses import categorical_crossentropy
from keras.regularizers import l2

tokenizer = TweetTokenizer()

nltk.download('stopwords')

stopWords = set(stopwords.words('english'))

from google.colab import drive
drive.mount('/content/drive')

file_dir='/content/drive/My Drive/drive/data/'
classes=3
embed_size=300
vocab_size=0

def read_test_data(fileName):  
  df=pd.read_excel(fileName)
  ytest=df['Label']
  testTweets=df['tweet']
  return testTweets,ytest

def read_train_data(fileName):    
  df=pd.read_excel(fileName)
  
  tweets = list(df['text'])
  y = list(df['airline_sentiment'])
  
  return tweets,y

def tweet_preprocessing(tweet):
    tweet = tweet.lower()
    tweet= re.sub(r'@\w+', ' ', tweet)
    tweet = re.sub(r'[a-zA-Z0-9]+', '',tweet,flags=re.MULTILINE)#remove english letters
    tweet = re.sub(r'\xa0', ' ',tweet)
    tweet = re.sub(r':', ' : ', tweet)
    tweet = re.sub(r'#', ' # ', tweet)
    tweet = re.sub(r'@', ' @ ', tweet)
    tweet = re.sub(r'[^\u1F600-\u1F6FF\s]', ' ', tweet)#remove emoji

    search = ["أ","إ","آ","ة","_","-","/",".","،",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!',':','*','>','<','[',']','«','»','٪''÷','×','\xa0','؛','”','“','…','٠','١','۱۰','٢','٣','٤','٥','٦','٧','٨','٩']              
    replace = ["ا","ا","ا","ه "," "," "," "," "," "," "," "," ","ي"," ",' ', ' ',' ',' ?',' ؟ ',' ! ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ']
            
    #remove tashkeel
    arab_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    tweet = re.sub(arab_tashkeel,"", tweet)
    
    #remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    tweet = re.sub(p_longation, subst, tweet)
    
    tweet = tweet.replace('ووو', 'و')
    tweet = tweet.replace('ييي', 'ي')
    tweet = tweet.replace('ااا', 'ا')    
      
    
    for i in range(0, len(search)):
        tweet = tweet.replace(search[i], replace[i])
    
    #remove extra spaces
    tweet=re.sub("\s+", " ", tweet) 
    #trim    
    tweet = tweet.trim()

    return tweet

def load_embedding():
  #word embedding for Arabic
  
  embeddings_index=gensim.models.Word2Vec.load('/content/drive/My Drive/drive/embeddings/AllData6_CBOW_300')

  return embeddings_index

def map_words_embeddings():

  OOV=0 #out off vocabulary
  embedding_matrix = np.zeros((vocab_size, embed_size))
  
  for word, i in tokenizer.word_index.items():
            try:
              embd_vec=model[word]          
              embedding_matrix[i]=embd_vec
            except:
              OOV+=1              
              
  return embedding_matrix,OOV

# calcualte evaluation measures: Percision, Recall,F1
# test_y: Ground truth values
# prds: predicted values
# number of classes used
def test_eval(test_y,prds)  :  
    y_val=[]
    pred_val=[]
    
    for p,s in enumerate(test_y):#[natural,negative,positive]

          y_val.append(np.argmax(s))
          pred_val.append(np.argmax(prds[p]))
    
    
    #F1  
    f1 = metrics.f1_score(y_val,pred_val,average='weighted')          
    #recall
    rec = metrics.recall_score(y_val,pred_val,average='weighted')
    # precision and           
    prec = metrics.precision_score(y_val,pred_val,average='weighted')
         
    #print('perc=',prec)
    #print('rec=',rec)
    #print('f1=',f1)
    return f1 #[f1,rec,prec]

def get_model(sel,dropout_rate,l_r):
  if sel==1:#CNN
        #dropout 0.3, batch=512,lr=0.005 epochs 1000  
        bias_init=initializers.Constant(value=0)
        np.random.seed(1337) 
        model = Sequential()
        e = Embedding(vocab_size, embed_size, weights=[embedding_matrix], 
                      input_length=max_tweet_length, trainable=True)
        model.add(e)
        model.add(Convolution1D(32, 3, border_mode='same',kernel_initializer='glorot_uniform',
                        bias_initializer=bias_init))
        model.add(MaxPooling1D())
        model.add(Dropout(dropout_rate))
        model.add(Convolution1D(64, 5, border_mode='same',kernel_initializer='glorot_uniform',
                        bias_initializer=bias_init))
        model.add(MaxPooling1D())
        model.add(Dropout(dropout_rate))
        model.add(Convolution1D(128, 7, border_mode='same',kernel_initializer='glorot_uniform',
                        bias_initializer=bias_init))
        model.add(Dropout(dropout_rate))
        model.add(Flatten())
        model.add(Dense(1000, kernel_initializer='glorot_uniform',
                        bias_initializer=bias_init))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=classes,
                           kernel_regularizer=regularizers.l2(0.004),                    
                           activation='softmax'))
        model.compile(loss=categorical_crossentropy, optimizer=adam(l_r), metrics=['accuracy'])        
  else: #LSTM       
        #Arabic
        #dropout 0.5, batch=512,lr=0.001 epochs 1000 
        model = Sequential()
        e = Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=max_tweet_length, trainable=True)
        model.add(e)
        model.add(Dropout(dropout_rate))
        model.add(LSTM(100,recurrent_dropout=dropout_rate) ) 
        model.add(Dense(classes, activation='softmax'))
        model.compile(loss=categorical_crossentropy, optimizer=adam(l_r), metrics=['accuracy'])        
        
  return model

def train_model(m,early_stop,batchSize):
  print('model training...')
  monitor = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=0, restore_best_weights=True)        
  m.fit(x_train, y_train, validation_split=0.1,epochs=1000, batch_size=batchSize, callbacks=[monitor],verbose=0, shuffle=False)
  return m

#fileName=file_dir+'HIAQatar_tweets.xlsx'
fileName=file_dir+'KKAISA_tweets.xlsx'
#fileName=file_dir+'AraSenti_all.xlsx'

tweets,y=  read_test_data(file_dir+'AraSenti_all.xlsx')
testTweets,ytest=  read_train_data(file_dir+'KKAISA_tweets.xlsx')

#tweets preprocessing

tweets=[tweet_preprocessing(t) for t in tweets]
testTweets=[tweet_preprocessing(t) for t in testTweets]

max_tweet_length = max([len(x.split()) for x in (tweets+testTweets)])

## Tokenization and padding

tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets+testTweets)

sequences = tokenizer.texts_to_sequences(tweets)
x_train = pad_sequences(sequences, maxlen=max_tweet_length)

sequences = tokenizer.texts_to_sequences(testTweets)
x_test = pad_sequences(sequences, maxlen=max_tweet_length)

vocab_size = len(tokenizer.word_index) + 1## in my dataset

#create one hot_vectors for labels
y_train=to_categorical(y,classes)
y_test=to_categorical(ytest,classes)

#upload pre_trained embedding
embeddings_index=load_embedding()

#map our data to word embedding
embedding_matrix,oov=map_words_embeddings()

#split data
x1_train, x1_test, y1_train, y1_test = train_test_split(x_train,y_train, test_size=0.2,random_state=37)

#model=get_model(1,0.3,0.005)#CNN
#model=train_model(model,10,512)#CNN

model=get_model(2,0.5,0.001)#LSTM
model=train_model(model,50,512)#LSTM

acc = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', acc[1])

preds=model.predict(x_test)  
f1=test_eval(y_test,preds)
print('Test f1=',f1)

