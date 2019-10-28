# PassengerTweetsEvaluation
Deep learning models for sentiment analysis of passengers tweets. 

Goal:

Labeled tweets are clustered to groups that represent airport services. 
Based on label of tweets under each service we can evaluate the services provided at airports.


Models used:

CNN

LSTM

In this work we devleoped two modles to predict sentimnet of passnger tweets written either in English or Arabic.
Each model is trained twice: once with Englis data and the other with Arabic data


Word Embedding:
Arabic: We train our own word embedding on more than 40 million Arabic tweets:
https://drive.google.com/file/d/1-1SNUte6ypj7zmcgd0VK4difLs8UkoFb/view?usp=sharing

English: We used the publicly available pre-trained word embedding model offered from Stanford University:
https://nlp.stanford.edu/projects/glove/
