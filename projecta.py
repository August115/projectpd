#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install "scikit_learn==0.22.2.post1"


# In[3]:


import numpy as np
import os
import matplotlib.pyplot as plt
import sagemaker
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_header, strip_newsgroup_quoting, strip_newsgroup_footer
newsgroups_train = fetch_20newsgroups(subset='train')['data']
newsgroups_test = fetch_20newsgroups(subset = 'test')['data']
NUM_TOPICS = 30
NUM_NEIGHBORS = 10
BUCKET = 'sagemaker-aw'
PREFIX = '20newsgroups'


# In[4]:


for i in range(len(newsgroups_train)):
 newsgroups_train[i] = strip_newsgroup_header(newsgroups_train[i])
 newsgroups_train[i] = strip_newsgroup_quoting(newsgroups_train[i])
 newsgroups_train[i] = strip_newsgroup_footer(newsgroups_train[i])


# In[5]:


newsgroups_train[1]


# In[6]:


get_ipython().system('pip install nltk')
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
import re
token_pattern = re.compile(r"(?u)\b\w\w+\b")
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if len(t) >= 2 and re.match("[a-z].*",t) 
                and re.match(token_pattern, t)]


# In[7]:


import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
vocab_size = 2000
print('Tokenizing and counting, this may take a few minutes...')
start_time = time.time()
vectorizer = CountVectorizer(input='content', analyzer='word', stop_words='english',
                             tokenizer=LemmaTokenizer(), max_features=vocab_size, max_df=0.95, min_df=2)
vectors = vectorizer.fit_transform(newsgroups_train)
vocab_list = vectorizer.get_feature_names()
print('vocab size:', len(vocab_list))

# random shuffle
idx = np.arange(vectors.shape[0])
newidx = np.random.permutation(idx) # this will be the labels fed into the KNN model for training
# Need to store these permutations:

vectors = vectors[newidx]

print('Done. Time elapsed: {:.2f}s'.format(time.time() - start_time))


# In[8]:


import scipy.sparse as sparse
vectors = sparse.csr_matrix(vectors, dtype=np.float32)
print(type(vectors), vectors.dtype)


# In[9]:


# Convert data into training and validation data
n_train = int(0.8 * vectors.shape[0])

# split train and test
train_vectors = vectors[:n_train, :]
val_vectors = vectors[n_train:, :]

# further split test set into validation set (val_vectors) and test  set (test_vectors)

print(train_vectors.shape,val_vectors.shape)


# In[10]:


from sagemaker import get_execution_role

role = get_execution_role()

bucket = BUCKET
prefix = PREFIX

train_prefix = os.path.join(prefix, 'train')
val_prefix = os.path.join(prefix, 'val')
output_prefix = os.path.join(prefix, 'output')

s3_train_data = os.path.join('s3://', bucket, train_prefix)
s3_val_data = os.path.join('s3://', bucket, val_prefix)
output_path = os.path.join('s3://', bucket, output_prefix)
print('Training set location', s3_train_data)
print('Validation set location', s3_val_data)
print('Trained model will be saved at', output_path)


# In[11]:


def split_convert_upload(sparray, bucket, prefix, fname_template='data_part{}.pbr', n_parts=2):
    import io
    import boto3
    import sagemaker.amazon.common as smac
    
    chunk_size = sparray.shape[0]// n_parts
    for i in range(n_parts):

        # Calculate start and end indices
        start = i*chunk_size
        end = (i+1)*chunk_size
        if i+1 == n_parts:
            end = sparray.shape[0]
        
        # Convert to record protobuf
        buf = io.BytesIO()
        smac.write_spmatrix_to_sparse_tensor(array=sparray[start:end], file=buf, labels=None)
        buf.seek(0)
        
        # Upload to s3 location specified by bucket and prefix
        fname = os.path.join(prefix, fname_template.format(i))
        boto3.resource('s3').Bucket(bucket).Object(fname).upload_fileobj(buf)
        print('Uploaded data to s3://{}'.format(os.path.join(bucket, fname)))
split_convert_upload(train_vectors, bucket=bucket, prefix=train_prefix, fname_template='train_part{}.pbr', n_parts=8)
split_convert_upload(val_vectors, bucket=bucket, prefix=val_prefix, fname_template='val_part{}.pbr', n_parts=1)


# In[12]:


import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'ntm')


# In[13]:


sess = sagemaker.Session()
ntm = sagemaker.estimator.Estimator(container,
                                    role, 
                                    instance_count=2, 
                                    instance_type='ml.c4.xlarge',
                                    output_path=output_path,
                                    sagemaker_session=sess)


# In[14]:


ntm.set_hyperparameters(num_topics=NUM_TOPICS, feature_dim=vocab_size, mini_batch_size=128, 
                        epochs=100, num_patience_epochs=5, tolerance=0.001)


# In[15]:





# In[28]:


np.savetxt('trainvectors.csv',
           vectors.todense(),
           delimiter=',',
           fmt='%i')
batch_prefix = '20newsgroups/batch'

train_s3 = sess.upload_data('trainvectors.csv', 
                            bucket=bucket, 
                            key_prefix='{}/train'.format(batch_prefix))
print(train_s3)
batch_output_path = 's3://{}/{}/test'.format(bucket, batch_prefix)

ntm_transformer = ntm.transformer(instance_count=1,
                                  instance_type ='ml.m4.xlarge',
                                  output_path=batch_output_path
                                 )
ntm_transformer.transform(train_s3, content_type='text/csv', split_type='Line')
ntm_transformer.wait()


# In[29]:


get_ipython().system('aws s3 cp --recursive $ntm_transformer.output_path ./')
get_ipython().system('head -c 5000 trainvectors.csv.out')


# In[30]:


labels = newidx 
labeldict = dict(zip(newidx,idx))


# In[31]:


import io
import sagemaker.amazon.common as smac


print('train_features shape = ', predictions.shape)
print('train_labels shape = ', labels.shape)
buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, predictions, labels)
buf.seek(0)

bucket = BUCKET
prefix = PREFIX
key = 'knn/train'
fname = os.path.join(prefix, key)
print(fname)
boto3.resource('s3').Bucket(bucket).Object(fname).upload_fileobj(buf)
s3_train_data = 's3://{}/{}/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))


# In[32]:


def trained_estimator_from_hyperparams(s3_train_data, hyperparams, output_path, s3_test_data=None):
    """
    Create an Estimator from the given hyperparams, fit to training data, 
    and return a deployed predictor
    
    """
    # set up the estimator
    knn = sagemaker.estimator.Estimator(get_image_uri(boto3.Session().region_name, "knn"),
        get_execution_role(),
        train_instance_count=1,
        train_instance_type='ml.c4.xlarge',
        output_path=output_path,
        sagemaker_session=sagemaker.Session())
    knn.set_hyperparameters(**hyperparams)
    
    # train a model. fit_input contains the locations of the train and test data
    fit_input = {'train': s3_train_data}
    knn.fit(fit_input)
    return knn

hyperparams = {
    'feature_dim': predictions.shape[1],
    'k': NUM_NEIGHBORS,
    'sample_size': predictions.shape[0],
    'predictor_type': 'classifier' ,
    'index_metric':'COSINE'
}
output_path = 's3://' + bucket + '/' + prefix + '/knn/output'
knn_estimator = trained_estimator_from_hyperparams(s3_train_data, hyperparams, output_path)


# In[36]:


def predictor_from_estimator(knn_estimator, estimator_name, instance_type, endpoint_name=None): 
    knn_predictor = knn_estimator.deploy(initial_instance_count=1, instance_type=instance_type,
                                        endpoint_name=endpoint_name)
    knn_predictor.serializer = CSVSerializer()
    knn_predictor.deserializer = JSONDeserializer()
    return knn_predictor
import time

instance_type = 'ml.m4.xlarge'
model_name = 'knn_%s'% instance_type
endpoint_name = 'knn-ml-m4-xlarge-%s'% (str(time.time()).replace('.','-'))
print('setting up the endpoint..')
knn_predictor = predictor_from_estimator(knn_estimator, model_name, instance_type, endpoint_name=endpoint_name)


# In[ ]:




