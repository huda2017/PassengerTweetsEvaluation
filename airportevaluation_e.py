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
  df=pd.read_csv(fileName)
  
  tweets = list(df['text'])
  y = list(df['airline_sentiment'])
  
  yy=[]
  for v in y:
        if v=='positive':
            yy.append(1)
        elif v=='neutral':            
            yy.append(2)
        elif v=='negative':            
            yy.append(0)
        else:
            yy.append(int(v))

              
  
  return tweets,yy

def tweet_preprocessing(raw_text):
    s = re.sub(r'http\S+', ' ',raw_text)    #remove http
    s= re.sub(r'@\w+', ' ', s)    #remove mentions
    s = re.sub(r'[^\u1F600-\u1F6FF\s]', ' ', s)#remove emoji       
    s=re.sub("[^a-zA-Z]", " ", s)    #remove non english letters
    s=re.sub("\s+", " ", s) #remove extra spaces
    
    # convert to lower case and tokenize
    words = tokenizer.tokenize(s.lower())         

    # remove stopwords
    #stopword_set = set(stopwords.words("english"))
    #meaningful_words = [w for w in words if w not in stopword_set]
    
    #remove length 1 words
    meaningful_words = [w for w in words if len(w)>1 ]
    
    # join the cleaned words in a list
    s = " ".join(meaningful_words)

    return s

def load_embedding():
  #word embedding for English
  #glove.840B.300d
  
  embeddings_index = {}

  with open('/content/drive/My Drive/drive/embeddings/glove.840B.300d.txt',encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        try:
          coefs = asarray(values[1:], dtype='float32')
          embeddings_index[word] = coefs
        except:
          #print(values)
          pass


  #print('Loaded %s word vectors.' % len(embeddings_index))
  return embeddings_index

def map_words_embeddings():

  OOV=0 #out off vocabulary
  
  embedding_matrix = np.zeros((my_vocab_size, embed_size))

  for word, i in tokenizer.word_index.items():
              embd_vec=embeddings_index.get(word)         
              if embd_vec is not None:
                embedding_matrix[i]=embd_vec
              else:
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
        #dropout 0.6, batch=64,lr=0.001 epochs 500  for english
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
        #English
        #dropout 0.7, batch=128,lr=0.1 epochs 500  for english
        model = Sequential()
        e = Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=max_tweet_length, trainable=True)
        model.add(e)
        model.add(Dropout(dropout_rate))
        model.add(LSTM(100,recurrent_dropout=dropout_rate) ) 
        model.add(Dense(classes, activation='softmax'))
        model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=l_r, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])
  return model

def train_model(m,early_stop,batchSize):
  print('model training...')
  monitor = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=0, restore_best_weights=True)        
  m.fit(x_train, y_train, validation_split=0.1,epochs=1000, batch_size=batchSize, callbacks=[monitor],verbose=0, shuffle=False)
  return m

fileName=file_dir+'Heathrow_tweets.xlsx'
#fileName=file_dir+'Gatwick_tweets.xlsx'
#fileName=file_dir+'US_airline_tweets.csv'

tweets,y=  read_test_data(file_dir+'Heathrow_annotator1.xlsx')
testTweets,ytest=  read_train_data(file_dir+'Tweets.csv')

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

#model=get_model(1,0.6,0.001)#CNN
#model=train_model(model,10,256)#CNN

model=get_model(2,0.7,0.1)#LSTM
model=train_model(model,10,128)#LSTM

acc = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', acc[1])

preds=model.predict(x_test)  
f1=test_eval(y_test,preds)
print('Test f1=',f1)

