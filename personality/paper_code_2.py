
import pandas as pd
from transformers import pipeline
import torch

# Loading data
neo_items =  pd.read_csv(
    'NEO_items.csv', usecols=['construct', 'text']
)

# Switching to GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Loading the feature extraction pipeline
model_ckpt = 'distilbert-base-uncased'
feature_extractor = pipeline(
    'feature-extraction', model=model_ckpt, tokenizer=model_ckpt,
    device=device, framework='pt', batch_size=8
)

# Extracting the features for all items
features = feature_extractor(
    neo_items['text'].to_list(), return_tensors='pt',
    tokenize_kwargs= {'padding': True, 'truncation': True}
)

# Extracting [CLS] features and converting to dataframe
features = pd.DataFrame([sample[0][0].numpy() for sample in features])


########################################
####### Construct similarity ###########
########################################

from sklearn.metrics.pairwise import cosine_similarity

# Converting to dataframe and adding construct labels
features['construct'] = neo_items['construct']

# Averaging embeddings for each construct
construct_features = features.groupby('construct').mean()

# Calculating the cosine similarity between construct embeddings
sims = cosine_similarity(construct_features)