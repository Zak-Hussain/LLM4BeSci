############ Data processing ###########

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Loading data and converting to HF Dataset
neo_items =  pd.read_csv(
    'NEO_items.csv', usecols=['construct', 'text']
)
dat = Dataset.from_pandas(neo_items)

# Tokenizing
model_ckpt = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
batch_tokenizer = lambda batch: tokenizer(
    batch['text'], padding=True, truncation=True
)
dat = dat.map(batch_tokenizer, batched=True)

# Setting to torch format
dat.set_format(
    'torch', columns=['input_ids', 'attention_mask']
)

######### Feature extraction ###########

from transformers import AutoModel
import torch

# Initialising model and moving to the GPU
model = AutoModel.from_pretrained(model_ckpt)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
model = model.to(device)

# Extracting features
def extract_features(batch):
    inputs = {
        k:v.to(device) for k, v in batch.items()
        if k in tokenizer.model_input_names
    }
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
        return {
            "hidden_state": last_hidden_state[:,0].cpu().numpy()
        }

dat = dat.map(extract_features, batched=True, batch_size=8)

# Converting to dataframe for further processing
features = pd.DataFrame(dat['hidden_state'])

#########################################################################################
