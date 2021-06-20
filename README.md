# 365 Days Of Machine Learning and Deep Learning ⚒
⌚ Here I will be documenting my journey from ▶14 June 2021 to 14 June 2022🔚 

## 🏆 Day 01 : 
Started working on an NLP project (Depression-Detection) implementing advanced NLP practices and got suggested the torchtext library from pytorch. 
The library provides a set of classes that are useful in NLP tasks. Bascially this library takes care of the typical components of NLP tasks namely : 

1. Preprocessing and Tokenization
2. Generating vocabulary of unique toke and converting words to indices
3. Loading Pretrained vectors like Glove, Word2vec,etc 
4. Padding text with zeroes in case of variable lengths
5. Dataloading and batching

So basically the preprocessing stages of NLP with minimal code. Looking into these points in detail and applying them into the preprocessing phases of the project tomorrow.

On the process of learning about torchtext i revised on Padding and came across word vector libraries like glove and fasttext. 
#### Padding in NLP
Why do we use padding in NLP tasks? → Padded sequences basically convert our tokenized lists of words into the same length. Example : 

```python
padded = pad_sequences(sequences, maxlen = 5)
print('\nPadded Sequences = {}'.format(padded)) 
```
The above snippet of code converts our tokenized sentences into the maximum length of 5 words.
Meaning a 7 word sentence will be padded into a 5 word sentence.  Similarly if the max padding length in 10 and our sentence has 5 words than our remaining places will be replaced be padded with 0.
#### Word Vectors
Word vectors or embeddings is a methodology in NLP to map words or phrases from vocabulary to a corresponding vector of real numbers where each number represents a word from the phrase.
What i learnt is that torchtext makes implementing pretrained word vectors much easier by just mentioning the name of one or specify the path of a word vector that we will be using and is predownloaded. 

## 📃 Day 02 : ***Preprocessing phase for Twitter Depression Detection Project #01***

- Removing abnormalies in the tweets : 

On the preprocessing phase, needed to remove the URLs that may contribute more towards advertisements than a potential depressive tweets, the hashtag symbols, the mentions, the emoticons and all other symbols and punctuations other than `?,! and .` 

So I read up and saw a few videos that discussed the use of RegEx in Modern NLP and after looking for some code online came up with this

```python
def tweet_clean(text):
    text = re.sub(r'https?:/\/\S+', ' ', text) # remove urls
    text = re.sub(r'<([^>]*)>', ' ', text) # remove emojis
    text = re.sub(r'@\w+', ' ', text) # remove at mentions
    text = re.sub(r'#', '', text) # remove hashtag symbol
    text = re.sub(r'[0-9]+', ' ', text) # remove numbers
    text = replace_contractions(text)
    pattern = re.compile(r"[ \n\t]+")
    text = pattern.sub(" ", text)      
    text = "".join("".join(s)[:2] for _, s in itertools.groupby(text))    
    text = re.sub(r'[^A-Za-z0-9,?.!]+', ' ', text) # remove all symbols and punctuation except for . , ! and ?
    return text.strip()
```
Which should work in theory, even though RegEx are pretty tedious and i pretty much skimmed through them in the college course lel.

For further preprocessing I had to pause and get some more knowledge on Pytorch.
Today I came across `torch.autograd` and made the following notes on the same, here's a brief summary : 

#### 🔦 Automatic Differentiation Package - torch.autograd

 → For the differentiation for all operations on tensors 

→ Performs the backpropagation starting from a variable.

→ This variable often holds the value of the cost function. backward executes the backward pass and computes all the backpropagation gradients automatically 


Backpropagation: In order to update the weights and reduce the loss we need to perform backpropagation. For that, we need to calculate the gradients. That's an advantage in PyTorch since the gradients will be updated automatically with Autograd.

Gradient : The derivative is the instantaneous rate of change of a function with respect to one of it's variables


```python
import torch

a = torch.tensor(5., requires_grad = True) #5. cause we always need floating point tensors 
b = 2*a**3
b.backward()
a.grad
# Output : tensor(150.) which is the derivative of b and proves our point
x = torch.tensor(5.)
w = torch.tensor(10., requires_grad = True)
b = torch.tesnor(5., requires_grad = True)
y = w*x+b
y 
# Output(55., grad_fn=<AddBackward0>)
y.backward()
print(x.grad) # Output : None
print(w.grad) # Output : tensor(5.)
print(b.grad) # Output : tensor(1.)
```

When we call .backward() it computes the gradient w.r.t all the parameters that have `required_grad = True` and store them in `parameter_grad`

Remember, the backward graph is already made dynamically during the forward pass. The backward function only calculates the gradient using the already made graph and stores them in leaf nodes

## 🔰 Day 03 : ***Preprocessing Phase for Twitter Depression Project #02***

Learnt about the **preprocessing pipeline for spaCy**. Will need to go into a little more detail later, for now I understand the bare bones of it. 
- Contains the preprocessing pipeline
- inclues language specific used for tokenization

When we process a text with the NLP object it creates a `doc` object 

```python
# A Doc is Created by processing a string of text with the nlp object
doc = nlp("Hello World") # The doc lets us access info about text in a structured way and no info is lost
	# Works as a normal python sequence and let's us iterate over the tokens or get a token by it's index
	for token in doc:
		print(token.text) 
# Output :
''' 
Hello
World
'''
```

We tokenize our tweets using spacy by making `tweet_clean(s)` as an `nlp` object.
```python
nlp = spacy.load("en_core_web_sm",disable=['lemmatizer', 'tagger', 'ner'])
def tokenizer(s): return [w.text.lower() for w in nlp(tweet_clean(s))]
```
#### Implementing torchtext :
- Defining Fields 
```python
# We define the fields for our tweets 
TEXT = Field(sequential=True, tokenize=tokenizer, include_lengths=True, use_vocab = True)
# We define the fields for our target variable. It does not need any tokenization since it is already in it's class form of 1s and 0s
TARGET = Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None, is_target=False)
# Here we assign our fields to our dataset
data_fields = [
    (None, None),
    ("tweet", TEXT), 
    ("target", TARGET)
]
```
#### 💬 ***Building our train, validation and test datasets***

I was honestly pretty confused on the significant differences between a test and validation datasets,

So I referred to [this insanely dedicated article](https://machinelearningmastery.com/difference-test-validation-datasets/#:~:text=%E2%80%93%20Validation%20set%3A%20A%20set%20of,of%20a%20fully%2Dspecified%20classifier.&text=These%20are%20the%20recommended%20definitions%20and%20usages%20of%20the%20terms) to have a better understanding for them. 
> Suppose that we would like to estimate the test error associated with fitting a particular statistical learning method on a set of observations. The validation set approach […] is a very simple strategy for this task. It involves randomly dividing the available set of observations into two parts, a training set and a validation set or hold-out set. The model is fit on the training set, and the fitted model is used to predict the responses for the observations in the validation set. The resulting validation set error rate — typically assessed using MSE in the case of a quantitative response—provides an estimate of the test error rate.

In short I made the train, validation and test set (80:20) and made them into a tabular_dataset for torchtext operations. After that I fitted the vector embedding `glove.6B.50d` to my training data and wrapped up for the day. 

## 🔍 Day 04 : ***Twitter Depression Detection Project : Loading Data in Batches*** 

Today was all about batches so referred to [this](https://gmihaila.medium.com/better-batches-with-pytorchtext-bucketiterator-12804a545e2a) article to have a better understanding on it, but honestly was still confused on the implementation part. 

- After making my train, validation and test data into tabular_datasets we use the `BucketIterator` to access the Dataloader. It sorts data according to length of text, and groups similar length text in a batch, thus reducing the amount of padding required. It pads the batch according to the max length in that particular batch

```python
train_loader, val_loader, test_loader = BucketIterator.splits(datasets=(train_data, val_data, test_data), 
                                            batch_sizes=(3,3,3), 
                                            sort_key=lambda x: len(x.tweet), 
                                            device=None, 
                                            sort_within_batch=True, 
                                            repeat=False)
```

Having batches with similar length sequences provides a lot of gain for recurrent modes and transformers models since they would require a minimal amount of padding.

- We then define the `idxtosent`, and not gonna lie, not exactly sure what this does yet.
```python
def idxtosent(batch, idx):
    return ' '.join([TEXT.vocab.itos[i] for i in batch.tweet[0][:,idx].cpu().data.numpy()])
```
We get the information about the batch with the .__dict__ method and we are all set to make our BatchGeneratorClass
```python
batch.__dict__
'''
{'batch_size': 3,
 'dataset': <torchtext.data.dataset.TabularDataset at 0x223a03d5730>,
 'fields': dict_keys([None, 'tweet', 'target']),
 'input_fields': ['tweet', 'target'],
 'target_fields': [],
 'tweet': (tensor([[  48,    4,  530],
          [  56,  107,  318],
          [ 119,  145,   10],
          [ 312,   63,   24],
          [  40,    6,   72],
          [  31, 2255,   45],
          [   9, 1437,   10],
          [1431,    2,  114]]),
  tensor([8, 8, 8])),
 'target': tensor([0, 0, 0])}
'''
```
The primary aim of the BatchGenerator class is to load our dataset and provide methods to get selected samples in the dataset by indexing.  
That is basically what I understood from my implementation today. It loads the dataloader we created previously, gets the independent and the dependent varibale fields in the `__init__` method. Further the `__len__` gets the total size of our dataset and the `__iter__` provides methods for accessing the dataset by index. `__getitem()__` may also be used to get the selected sample in the dataset by indexing. 
I came across [this](https://blog.paperspace.com/dataloaders-abstractions-pytorch/) article with a comprehensive explanation of the dataloader class. I will look into it tomorrow.
```python
class BatchGenerator:
    def __init__(self, dl, x_field, y_field):
        # data loading
        self.dl, self.x_field, self.y_field = dl, x_field, y_field
        
    def __len__(self):
        # len(dataset)
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X,y)
```
Also today i started collaborating with someone i met in a discord group on an Employee_Satisfaction dataset. 
It is a real world dataset with a lot to uncover. Everything was cluttered in it so went on a discord call to discuss on what features would actually contruibute to the target_satisfaction variable. It was a confusing session. 

## 📑 Day 05 : ***Pytorch fundamentals : `DataLoader` class and Abstractions***
I was following a learning as I build approach so the fundamental concepts of loading and different types of data in Pytorch just confused me a lot. So I follwed [this](https://blog.paperspace.com/dataloaders-abstractions-pytorch/) article to have a better comprehensive look on it, in summary :

I learnt about the `DataLoader` class and its significance in handling the data neatly by organizing it in accordance with the given paramters. Then analyzed the MNIST dataset by looking at varios possible techniques to call it into the workspace and how to load the Data onto the GPUs using `CUDA`. Then learnt about the transforming and rescaling the data using the different methods of the `transforms` module like `Resize()`, `CenterCrop()`,`ToTensor()`, etc, which were admittedly completely outside the scope of the project I am currently working on but alright. 
Understood some of the differences between a map-styled and iterable datasets and how the project I am currently working on uses an Iterable dataset ( though this part still confuses me ) 

Also used the `DataLoader` class to make a custom dataset in pytorch that has numbers and their squared values. Let us call our dataset SquareDataset. Its purpose is to return squares of values in the range [a,b]. Below is the relevant code: 
```python
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class SquareDataset(Dataset):
     def __init__(self, a=0, b=1):
         super(Dataset, self).__init__()
         assert a <= b
         self.a = a
         self.b = b
        
     def __len__(self):
         return self.b - self.a + 1
        
     def __getitem__(self, index):
        assert self.a <= index <= self.b
        return index, index**2

data_train = SquareDataset(a=1,b=64)
data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True)
print(len(data_train))
```
### 🤖 GATED RECURRENT UNIT
While I had previously done a good study on LSTMS from the infamous [colah's LSTM blog post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) today is the first time I came across GRU.
GRU is short for Gatted Recurrent Unit and was introduced in 2014. Unlike LSTMs, GRUs have no cell state and it instead uses the hidden state where it combines the long and short term memory. In LSTMs we have the forget, input and output gate whereas in GRU we have - 
- Update gate which is similar to our forget and input gate and specifies how much of the past memory to retain
- Reset gate which specifies how much of the past information to forget.
Thankfully I got a good revision on LSTM as well thanks to Michael Phi's youtube video on [Illustrated Guide to LSTM's and GRU's](https://www.youtube.com/watch?v=8HyCNIVRbSU)

## 📜 Day 06 : ***Implementing RNN, GRU and LSTM***
Inorder to implement GRU into the [depression-detection-project]() I have been working on, today I dedicated most of my time into understanding the implementations for RNN, LSTM and GRU. 
For this purpose I followed the official [pytorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) as well as [this tutorial](https://www.youtube.com/watch?v=0_PgWWmauHk&t=396s) from [Python Engineer](https://www.youtube.com/channel/UCbXgNpp0jedKWcQiULLbDTA)

Below is a snippet for the implemented class for the models

```python
# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  
        # or:
        #out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
```

In one similar project relating to Depression-Detection, AGA-GRU was used for training the model which is an even more optimised version of GRU where AGA stands for Adaptive Genetic Algorithm. I tried to read up on the [research paper](https://iopscience.iop.org/article/10.1088/1742-6596/1651/1/012146#:~:text=The%20weight%20adjustment%20of%20gated,GRU) but of course it overwhelmed me so I kept it aside. 

### ➡Bidirectional RNNs⬅
Let us look at a simple RNN cell

![Insert Image](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/Untitled.png)

We can see that the previous words influence the final output. But in order to determine if apple is a fruit or a company we need to have the influence of the next words as well. 

Inorder to tackle this problem we need to input the words that come after Apple so we need to make a change to this structure.

To do this, we add another layer which processes the word from right to left ——-

![Insert Image here](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/Untitled%20(1).png)

We should use Bi-directional RNN for all sorts of NLP tasks.
However for speech recognition will not work well since the input is gotten w.r.t time.

It is slower to LSTM RNN or simple RNN.

## 📌Day 07 : Revisiting Pooling and Model Building #01 for depression-detection-project

Pooling is a type of downsampling. We use pooling for the conveniences like [1]reducing the input size for efficient computations of the convolutional layers and [2] To achieve "spacial invariance" for any given input.

"Spacial Invariance" is a property which means that our model can recognise a given object in our input vector no matter the orientation.

There are three types of pooling namely [1] Average Pooling and [2] Max Pooling and [3] Min Pooling

[1] Average Pooling: Example - from a 4x4 matrix that we divide into 4 sections, we choose the average value from each section and convert it into a 2x2 matrix similarly, for [2] Max Pooing and [3] Min Pooling we choose the Max and Min respectively.

The sections in the NxN matrix are referred to as filters and change based on the size of the given input. 

Preference :

[1] Average Pooling: Smooths out the given image and hence sharp features may nt be identified.
[2] Max Pooling: Selects the brighter pixels from the image. Useful when we only need the brighter pixels from an image with dark background and it is vice versa for [3] Min Pooling.

Let us look at the implementation for these pooling methods. 
```python
from skimage.measure import block_reduce
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('black_stick.png')
mean_pool=block_reduce(img, block_size=(9,9,1), func=np.mean)
max_pool=block_reduce(img, block_size=(9,9,1), func=np.max)
min_pool=block_reduce(img, block_size=(9,9,1), func=np.min)

plt.figure(1)
plt.subplot(221)
imgplot = plt.imshow(img)
plt.title('Original Image')

plt.subplot(222)
imgplot3 = plt.imshow(mean_pool)
plt.title('Average pooling')

plt.subplot(223)
imgplot1 = plt.imshow(max_pool)
plt.title('Max pooling')

plt.subplot(224)
imgplot1 = plt.imshow(min_pool)
plt.title('Min pooling')

plt.show()

```
### Depression-Detection-Project : Model Building #01
In the depression-detection-project today I implemented a simple GRU model with concat pooling and achieved a validation accuracy score of 78%. Next I will be looking into implementing the LSTM+CNN model inorder to try and improve the accuracy.

![insert acc image](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/depression-detection-accuracy.png)

Concat Pooling in simple terms means taking max and average pools of the output of all timesteps and then concatenating them along with the last hidden state before passing it as the output layer.

The following article gives a detailed implementation for GRU with concat pooling: 
https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130

