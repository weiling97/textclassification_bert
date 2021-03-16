#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
from tqdm.notebook import tqdm


# In[2]:


df = pd.read_csv("C:/Users/Downloads/textclassification_bert/smileannotationsfinal.csv", names=['id','text','category'])
df.set_index('id', inplace=True)


# In[3]:


df.head()


# In[4]:


df.category.value_counts()


# In[5]:


df = df[-df.category.str.contains('\|')]


# In[6]:


df = df[df.category != 'nocode']


# In[7]:


df.category.value_counts()


# In[8]:


possible_labels = df.category.unique()


# In[9]:


label_dict = {}
for index, possible_labels in enumerate(possible_labels):
    label_dict[possible_labels] = index


# In[10]:


label_dict


# In[11]:


df['label'] = df.category.replace(label_dict)
df.head()


# # Step 3: Training/Validation Split

# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_val, y_train, y_val = train_test_split(df.index.values, df.label.values, test_size=0.15, random_state=17, stratify=df.label.values)


# In[14]:


df['data_type'] = ['not_set']*df.shape[0]


# In[15]:


df.head()


# In[16]:


df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'


# In[17]:


df.head()


# In[18]:


df.groupby(['category','label','data_type']).count()


# # Step 4: Loading Tokenizer and Encoding our Data

# In[27]:


from transformers import BertTokenizer
from torch.utils.data import TensorDataset


# In[28]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) #converting everything to lower case


# In[21]:


encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].text.values, #the exact sentence
    add_special_tokens=True, #bert knows where the sentence ends and begins
    return_attention_mask=True, #so we know when the sentence finishes 
    pad_to_max_length=True, #want to set sentences to certain max length
    max_length=256, 
    return_tensors='pt', #pt == pytorch
    truncation=True
)  #convert all our tweets to encoded data form

encoded_data_val = tokenizer.batch_encode_plus(df[df.data_type=='val'].text.values, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, max_length=256, return_tensors='pt', truncation=True) #pt == pytorch

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)


# In[25]:


attention_masks_train


# In[23]:


dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)


# In[24]:


len(dataset_train)


# In[25]:


len(dataset_val)


# # Step 5: Setting up BERT Pretrained Model

# In[29]:


from transformers import BertForSequenceClassification


# In[30]:


model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', #base version is more efficient
    num_labels = len(label_dict), 
    output_attentions = False, #attention: attending to certain words more than the others
    output_hidden_states = False)


# # Step 6: Creating Data Loaders

# In[31]:


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


# In[45]:


batch_size = 4 #32 #since we have limited memory so use 4 only

dataloader_train = DataLoader(
    dataset_train,
    sampler=RandomSampler(dataset_train),
    batch_size=batch_size
)

dataloader_val = DataLoader(
    dataset_val,
    sampler=RandomSampler(dataset_val),
    batch_size=32 #since dont have to do many propogation
)


# # Step 7: Setting Up Optimizer and Scheduler

# In[33]:


#optimizer and scheduler are part of what makes BERT works
from transformers import AdamW, get_linear_schedule_with_warmup #adam algorithm with weight: optimising our weight


# In[34]:


optimizer = AdamW( #how our learning rate changes through time
    model.parameters(),
    lr=1e-5, #recommended: 2e-5 > 5e-5 learning rate depends on ur dataset
    eps = 1e-8
)


# In[35]:


epochs = 10

scheduler = get_linear_schedule_with_warmup( #what controls the learning rate
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(dataloader_train)*epochs #how many times u want ur learning rate to change
)


# # Step 8: Defining our Performance Metrics

# In[36]:


import numpy as np


# In[37]:


from sklearn.metrics import f1_score


# In[38]:


#preds =  [0.9 0.05 0.05 0 0 0] #almost like a probablility distr
#preds = [1 0 0 0 0 0] #want to make it binary


# In[39]:


#use f1 cause theres a class imbalance
def f1_score_func(preds, labels):
    #flatten and get the form we want it to be
    preds_flat = np.argmax(preds, axis=1).flatten() #make preds into a flat vector, meaning change it to binary form
    labels_flat = labels.flatten()

    return f1_score(labels_flat, preds_flat, average='weighted') #weigh each class based on how many samples exists #'weighted' able to change to 'macro'


# In[40]:


def accuracy_per_class(preds, labels): #print out the accuracy per class i.e. take the true labels of class 5 and see how many of them are actually class 5
    label_dict_inverse = {v: k for k, v in label_dict.items()} #create an inverse dict to the one we had before, i.e. b4: happy -> 0, now: 0 -> happy. 
                                                               #instead if key to value, now value to key
    
    #flatten and get the form we want it to be
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n') #y_preds==label means the correct label


# # Step 9: Creating our Training Loop

# Approach adapted from an older version of HuggingFace's run_glue.py script. Recommended way if u want to fine-tune BERT on ur own data set

# In[46]:


import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# In[47]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device) #sending the model to the device that we are using

print(device)


# In[53]:


def evaluate(dataloader_val): #similar to training function code so wont go through
    
    model.eval()
     
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in tqdm(dataloader_val):
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':  batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}
        
        with torch.no_grad():
            outputs = model(**inputs) 
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
        
    loss_val_avg = loss_val_total/len(dataloader_val)
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    
    return loss_val_avg, predictions, true_vals


# In[54]:


for epoch in tqdm(range(1, epochs+1)): #epoch defined earlier to be 10 so running thru 1 to 10 #tqdm is a progress bar
    
    model.train() #set model to be in training mode
    
    loss_train_total = 0 #ltr will add each batch loss to this variable
    
    progress_bar = tqdm(dataloader_train,
                        desc='Epoch {:1d}'.format(epoch),
                        leave=False, #let it overwrite itself
                        disable=False)
    for batch in progress_bar:
        
        model.zero_grad() #set gradiant to zero, a standard procedure
        
        batch = tuple(b.to(device) for b in batch) #batches of 3 items, make sure each item is on the correct device
        
        inputs = {
            'input_ids'     : batch[0],
            'attention_mask': batch[1],
            'labels'        : batch[2]
        }
        
        
        outputs = model(**inputs) #run our model, while unpacking the dictionary of inputs
        
        loss = outputs[0]
        loss_train_total += loss.item() #add the lost items
        loss.backward() #back propagation to improve performance?
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #give our grad a norm value == 1, prevent grad to become exceptionally small or too big, help promote generalization
                                                                #do that to all our parameters, all our weights
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))}) #include that in progress bar
        
    torch.save(model.state_dict(), f'Models/BERT_ft_epoch{epoch}.model') #save the model every epoch, name it BERT finetuned (ft) model
    
    tqdm.write('\nEpoch {epoch}') #type which epoch we are on
    
    loss_train_avg = loss_train_total/len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_val) #use the evaluate function to get validation loss 
                                                                #similar to training's except we dont change any grad, dont do backpropagation
                                                                #want to know if model is overtraining, which will occur when training loss is going down but validation loss is going up
                                                                #means it doesnt have generalisation ability, validation replicate training totally
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (weighted): {val_f1}')

