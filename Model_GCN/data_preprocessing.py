import json
import gzip
import nltk
import os
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from utils import clean_str


def chkList(lst):
    return len(set(lst)) == 1

train_or_test = []
comment_train=[]
comment_test = [] 
rating_train = [] 
each_category= 1000
rating_test  = []
data_files=os.listdir('RawData')            
for data in data_files:
  if data[-2:] == 'gz':
    print("Extracting data from file: " + data)
    data_from_each_category =[each_category,0,0,0,0,0]
    comments=[]
    ratings=[]  
    with gzip.open('RawData/'+data) as f:
        for l in f:
            i=(json.loads(l.strip()))
            if i.__contains__('overall') and  i.__contains__('reviewText') and len(clean_str(' '.join(i['reviewText'].split())).split())>10:
                if(data_from_each_category[int(i['overall'])] < each_category):
                    ratings.append(int(i['overall']))
                    comments.append(' '.join(i['reviewText'].split()))
                    data_from_each_category[int(i['overall'])] = data_from_each_category[int(i['overall'])]+1
            if(chkList(data_from_each_category) and data_from_each_category[1] == each_category):
                break
        print('Data of each class from  data file '+ data + ':: ' +str(data_from_each_category))
        comment_train_, comment_test_, rating_train_, rating_test_ = train_test_split(comments, ratings, test_size=0.33, shuffle=False)
        comment_train=comment_train + comment_train_
        comment_test=comment_test+comment_test_
        rating_train=rating_train+rating_train_
        rating_test=rating_test+rating_test_
    
print(len(rating_train))
print(len(rating_test))    
print("\n Creating test data")
test_data=[]
for i in range(len(comment_test)):
    test_str= str(rating_test[i]) + '\t' + comment_test[i]
    train_or_test.append('test')
    test_data.append(test_str)

final_test_data = '\n'.join(test_data)

f = open('Data/Amazon/' + 'test.txt', 'w')
f.write(final_test_data)
f.close()
print(" Test data created! Size of Test Data: ",len(test_data))


print("\n Creating train data")

train_data=[]
for i in range(len(comment_train)):
    train_str= str(rating_train[i]) + '\t' + comment_train[i]
    train_or_test.append('train')
    train_data.append(train_str)

final_train_data = '\n'.join(train_data)
f = open('Data/Amazon/' + 'train.txt', 'w')
f.write(final_train_data)
f.close()
print(" Train data created! Size of Train Data:",len(train_data))

final_data_comment=comment_test + comment_train
final_data_rating=rating_test + rating_train

final_data=[]
for i in range(len(final_data_comment)):
    final_data_str = str(i) + '\t' + train_or_test[i] + '\t' + str(final_data_rating[i])
    final_data.append(final_data_str)
    
final_data_str = '\n'.join(final_data)
f = open('data/' + 'Amazon.txt', 'w')
f.write(final_data_str)
f.close()

corpus_str = '\n'.join(final_data_comment)
f = open('Data/corpus/' + 'Amazon.txt', 'w')
f.write(corpus_str)
f.close()

print('\n Removing words \n')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


data = []
f = open('data/corpus/' + 'Amazon.txt', 'rb')
for line in f.readlines():
    data.append(line.strip().decode('latin1'))
f.close()

word_freq = {}

for comment in data:
    comment = clean_str(comment)
    words = comment.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

clean_comments = []
for comment in data:
    comment = clean_str(comment)
    words = comment.split()
    if(len(words)<1):
        print('failing'+str(len(words)))
    doc_words = []
    for word in words:
        if word not in stop_words and word_freq[word] >= 3:
            doc_words.append(word)
    doc_str = ' '.join(doc_words).strip()
    clean_comments.append(doc_str)

clean_comment_str = '\n'.join(clean_comments)
f = open('data/corpus/' + 'Amazon.clean.txt', 'w')
f.write(clean_comment_str)
f.close()

print('\n Words Removed')
