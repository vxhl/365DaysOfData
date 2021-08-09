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

## 📌Day 08: ***The LSTM+CNN Model & Sentiment Analysis with Variable Length Sequences in Pytorch***

### 📦 ***THE CNN-LSTM ARCHITECTURE***
The CNN-LSTM architecture involves using Convulational layers for feature extraction on input data combined with LSTMs to support sequence prediction.

This architecture is primarily aimed towards --- Generating textual description of an activity demonstarted in a sequence of images ( Videos ) or a single image.

This architecture is also being used for speech recognition and NLP tasks where CNN is used a feature extractor for the LSTMs on audio or textual data.

In other words this model is appropriate for the following kinds of problems --
1. Where the input has a spatial structures such as the pixels in an image or the structure of words that is the sequence and paragraphs that can be expressed in the form of a vector. 

2. Temporal structure, in other words the structure or order of the different images we feed to it or the different words in a text.

A CNN-LSTM layer can be implemented in Keras by taking two sub-models i.e, the CNN model for feature extraction and the LSTM model for interpreting the features along the time steps.

### 🔥 ***Sentimental Analysis with Variable Length Sequence in Pytorch***
Studied up on the essential steps for working with Variable length sequences in Pytorch. Again.
While I am quite confident that it will take a little too much time to implement on my own, atleast studying up on this gave me the proper structure that I need to follow for problems like these. For example---
- Building the vocabulary for all the unique words in lowercae, then converting the sequences to indices and calculating the length of each index. We use spaCy for this process.
- Creating our pytorch Dataset class
- Loading our custom Dataset with the DataLoader class
- Finally in the output of our padded dataset and dataloader we get the samples are of equal lengths and ouput of DataLoader as LongTensor.

On these foundations we apply our standard practices and tweak around to increase performance. 

References :
[Sentimental Analysis with Variable Length Sequences](https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130)
[LSTM-CNN Models](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/#:~:text=A%20CNN%20LSTM%20can%20be,the%20features%20across%20time%20steps)

## 📌Day 09 : ***Implementing and Learning about Word2vec + Implementing LSTM-CNN model on depression-detection-project*** 
### 📊***Vector Embeddings using Word2vec***
Before understanding Word2vec let us look into word embeddings/vector embeds.

Vector embeddings is one of the central ideas of Natural Lanuage Processing models. 

In simple words, they map a discrete categorical variable, in our case words, into a vector of continues numbers unique to the variable.

Word2vec is a method of efficiently creating word embeddings. In addition to it's utility as a word embedding method, some of its concepts are being used in creating *recommendation engines* and making sense of sequential data in commercial, non-language tasks.

Word2vec allows us to get the unique meanings of the words in vectors allowing us to do fascinating mathematical operations like --
"King - man + woman = Queen" which is honestly mindblowing.
![InsertImageHere](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/word2vec.png)
In the above example, Word2vec computes the unique properties for each word through various computations and givnes us the final word vector as "Queen"!
Let us take a look at a simple implementation using the Amazon reviews dataset for Cell Phones and accessories
```python
# We use the gensim library for implementing Word2vec
# 1. Initialising the model
import gensim
model = gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4,
)

# 2. Building the vocabulary
model.build_vocab(review_text, progress_per=1000)

# 3. Training the Word2Vec Model
model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)

# 4. Saving the model
model.save("./word2vec-amazon-cell-accessories-reviews-short.model")

# Finding Similar words and similarity between words
model.wv.most_similar("bad")
'''
Output:
[('terrible', 0.6617082357406616),
 ('horrible', 0.6136840581893921),
 ('crappy', 0.5805919170379639),
 ('good', 0.5770503878593445),
 ('shabby', 0.5749340653419495),
 ('awful', 0.5492298007011414),
 ('ok', 0.5294141173362732),
 ('cheap', 0.5288074612617493),
 ('legit', 0.5199155807495117),
 ('okay', 0.5171135663986206)]

'''
# We check the similarities between "CHEAP" and "INEXPENSIVE"
model.wv.similarity(w1="cheap", w2="inexpensive")
'''
output: 0.52680796
'''
# We check the similarities between "GREAT" and "GOOD"
model.wv.similarity(w1="great", w2="good")
'''
Output: 0.7714355
'''
```
### 📦 ***Depression-Detection-Project: Implementing the LSTM-CNN model #01***
Even though I have already gotten a 78% accuracy with the GRU using Concat pooling model, next I have started implementing the LSTM-CNN model.

Of course before that I had to learn more about implementing it all from the start. Thankfully this helped! 
-> https://github.com/vxhl/Detecting-Depression-in-Social-Media-via-Twitter-Usage

So for now I have tokenized and cleaned the tweets in a much better way than I did in my previous model. Will be adding the word embeds and be defining the Convolutional and LSTM layer tomorrow. 

References : [Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
[Codebasics Word2Vec](https://www.youtube.com/watch?v=hQwFeIupNP0)

## 📌Day 10: Understanding Batch Normalization and implementing HandTracker using google's mediapipe

### 📈Batch Normalization📈
Normalization is one of the most important preprocessing techniques where we rescale our different input data having different varying ranges into a range of 0 -> 1

Similar to this, Batch normalization is instead applied on the input that is being fed into each batch of our neural network, thus stabilizing the learning process and also reducing the number of epochs required to achieve a desirable outcome.

Normalizing our input data ensures that our data is between the same small fixed range. 

Why Batch Normalization?
1. Increases the speed of training
2. Decreases the importance of initial weights since all our input data points converge easily within the same range
3. Regularizes the model to some extent.

In more algorithmic terms-- 

The first thing that Batch normalization does is normalize the output for our activation function. After normalizing the output from the AF, batch norm then multiplies this normalized output with some arbitrary parameter and adds another arbitrary parameter to this product. This calculation with the new arbitrary parameters sets a new standard deviation and mean of the data. The mean, std deviation and arbitrary parameters are all trainable meaning they will keep getting optmized durint the training process. This makes it so that the weights within the network do not become imbalanced with extremely high or low values since normalization is included in the gradient process.

Implementation in Keras -
```python

# Without Batch Normalization
model_1 = Sequential()
model_1.add(Conv2D(32, (3,3), activation="relu"))
model_1.add(MaxPooling2D((2, 2)))
model_1.add(Conv2D(64, (3,3), activation="relu"))
model_1.add(MaxPooling2D((2, 2)))
model_1.add(Conv2D(64, (3,3), activation="relu"))
model_1.add(Flatten())
model_1.add(Dense(64, activation="relu"))
model_1.add(Dense(10, activation="softmax"))
model_1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
batch_size = 128
epochs = 5
model_1_history = model_1.fit(X_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=1,
                              validation_data=(X_test, y_test))
'''
Epoch 1/5
469/469 [==============================] - 2s 5ms/step - loss: 0.2422 - accuracy: 0.9293 - val_loss: 0.0567 - val_accuracy: 0.9836
Epoch 2/5
469/469 [==============================] - 2s 4ms/step - loss: 0.0583 - accuracy: 0.9819 - val_loss: 0.0350 - val_accuracy: 0.9888
Epoch 3/5
469/469 [==============================] - 2s 4ms/step - loss: 0.0416 - accuracy: 0.9869 - val_loss: 0.0298 - val_accuracy: 0.9898
Epoch 4/5
469/469 [==============================] - 2s 4ms/step - loss: 0.0325 - accuracy: 0.9893 - val_loss: 0.0317 - val_accuracy: 0.9899
Epoch 5/5
469/469 [==============================] - 2s 4ms/step - loss: 0.0263 - accuracy: 0.9917 - val_loss: 0.0297 - val_accuracy: 0.9900
'''

# With Batch Normalization
model_2 = Sequential()
model_2.add(Conv2D(32, (3,3), activation="relu"))
model_2.add(MaxPooling2D((2, 2)))
model_2.add(Conv2D(64, (3,3), activation="relu"))
model_2.add(MaxPooling2D((2, 2)))
model_2.add(Conv2D(64, (3,3), activation="relu"))
model_2.add(Flatten())
model_2.add(Dense(64))
model_2.add(BatchNormalization())
model_2.add(Activation("relu"))
model_2.add(Dense(10))
model_2.add(BatchNormalization())
model_2.add(Activation("softmax"))

model_2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model_2_history = model_2.fit(X_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=1,
                              validation_data=(X_test, y_test))
'''
Epoch 1/5
469/469 [==============================] - 3s 6ms/step - loss: 0.3604 - accuracy: 0.9630 - val_loss: 0.3049 - val_accuracy: 0.9866
Epoch 2/5
469/469 [==============================] - 3s 5ms/step - loss: 0.1420 - accuracy: 0.9888 - val_loss: 0.0891 - val_accuracy: 0.9910
Epoch 3/5
469/469 [==============================] - 2s 5ms/step - loss: 0.0868 - accuracy: 0.9919 - val_loss: 0.0760 - val_accuracy: 0.9887
Epoch 4/5
469/469 [==============================] - 2s 5ms/step - loss: 0.0584 - accuracy: 0.9940 - val_loss: 0.0469 - val_accuracy: 0.9930
Epoch 5/5
469/469 [==============================] - 2s 5ms/step - loss: 0.0423 - accuracy: 0.9954 - val_loss: 0.0553 - val_accuracy: 0.9917
'''
```
As we can see the normalized model reaches the maximum accuracy more quickly when compared to the non batch-normalized dataset. Even though this looks small now, this difference gets magnified immensely on bigger batches.

I also implemented the HandTracking module using `mediapipe` and `opencv` today which was surprisingly easy. 
![insert](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/handtracker.png)

## 📌Day 11: From Sliding Windows to Faster R-CNN 
I have been reading up on a research paper on [Driver Distraction Detection and Early Prediction and avoidance of accidents using CNN](https://iopscience.iop.org/article/10.1088/1742-6596/1770/1/012007#:~:text=The%20behaviour%20of%20drivers%20under,processing%20and%20computer%20vision%20problem). While trying to read and understand the terms in the said paper I came across the concept of Sliding Windows which I was unfamiliar with at the moment.

### ✂ Sliding Windows Object Detection: 
In this algorithm a NxM window slides from the top left of an image and each time we try to get a sample which is given to a CNN. The CNN determines whether the given image of the said object or not. If it is not the said object then we continue to slide our window till we find our required image. 

However we notice that the window size cannot be predetermined for the given image, because of this reason we initially pass a random size of say nXm and we keep on increasing or decreasing the window size according to our required object. But of course this requires an immense amount of computation time since we have to select an immense number of regions and thus is inefficient, even more so with multiple object detection.

### 📈R-CNN -> Fast R-CNN -> Faster R-CNN
#### 1. R-CNN
Inorder to bypass the problem of selecting a huge number of regions and adjusting them, the RCNN method uses selective search of the image to get 2000 regions from the image. So instead of iterating over the huge number of matrices we instead work with just 2000 selected regions generated using the ["selective search algorithm"](https://learnopencv.com/selective-search-for-object-detection-cpp-python/#:~:text=Selective%20Search%20is%20a%20region,texture%2C%20size%20and%20shape%20compatibility)    

To know more about the selective search algorithm, follow this [link](https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf).

Problems: 
- It would still take a large amount of time to train the network since we would have to computer 2000 regions per image
- It cannot be implemented real time since it take approx. 47 seconds to test each image. 
- The selective algorithm is fixed. So no learning is happening at that stage to tweak the process. 
#### 2. Fast R-CNN
Fast R-CNN is faster than the normal one since here we do not need to feed 2000 region proposals for each image in our dataset to the CNN. Instead we do the convolutional operation once per image. From the convolutional feature maps we then extract the region proposals and then warp then reshape them using something called RoI pooling layer which I did not go in depth on, to feed them into a fully connected layer. 
![Insert](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/rcnnvsfastrcnn.png)
As we can see, fast R-CNN is significantly faster than R-CNN

#### 3. Faster R-CNN
In this level we abandon the fixed property of the selective search algorithm since it is slow and time consuming. Here we instead use another network to predict the regions from our convolutional feature map. Then similarly we use the RoI pooling layer to reshape the regions which are then used to classify the image within the regions. 
![Insert](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/fastestRCNN.png)
As we can see Faster R-CNN is even faster.

I will cover till here for now since this will get a little too long If I go into YOLO now. Will go into more details with YOLO on some other day. 

References: https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e

https://www.youtube.com/watch?v=AimW3j7M2Uw

## 📌Day 12: Revisiting Data Augmentation techniques + Improving model performance for Depression-Detection-Project

### ⛰ Data Augmentation ⛰

This concept allows us to increase the diversity and amount of data for our training models to gain better results. 

Our supervised deep learning models require a substantial amount of data and the diversity of it to provide reliable results during training. Deep Learning models require a huge amount of data to achieve high performance.

Models trained to achieve high performance on complex tasks generally have a large number of hidden neurons and as the number of neurons increases, so does the number of trainable parameters. 

So the amount of data is proportional to the number of learnable parameters in the model.

So in cases where the number of training data is less or there is less diversity in the training data we use ***Data Augmentation***. In addition this concept is also used in addressing class imabalance in classificaion tasks. 

Our Deep Learning models are dumb so we can just change the orientation or composition of the image and it will be considered as a completely different image by our model. There are many types of augmentations that are possible using the `ImageDataGenerator` class of `keras` like -> Random Zoom, Random Brightness, Random Rotation Augmentation, etc.

Let us look at a random brightness augmentation example for images of macaws or whatever this bird is

```python
# example of brighting image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
# load the image
img = load_img('bird.jpg')
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)
# show the figure
pyplot.show()

```
Output:
![bird](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/bird.png)

As we can see the algorithm changes the exposure for the images of the birds and creates them into new data points.

### Depression-Detection-Project: Improving model performance
So I had achieved a val_acc of 75% for the GRU model which is pretty bad. So today I looked into some hyperparameter tuning practices to improve the model.

Applied dropout of 0.5 and reduced batch_size from 32 to 16 which increased the accuracy to 78% for now. But still haven't figured out how to implement augmentation for this, or if this even needs augmentation. Will look into it tomorrow. 

## 📌Day 13: ⌚ Going in depth with Time-Series Forecasting #01
Ok, I've been kind of vague and all over the place tbh, so in the next few days, I will be focusing on Time Series Forecasting while implementing a project on the AirPassengers dataset.

Before that let us look into the what and why's of time series.

A machine learning dataset is a collection of observations. We make future predictions on the basis of prior observations but of course, while keeping in mind the idea of "concept drift" to maintain our accuracy for the relevant data. For example, We may use past data only from the last two years of observations rather than all the data available to us.

In predictive analytics and machine learning, concept drift means that the statistical properties of the target variable, which the model is trying to predict, change over time in unforeseen ways. This causes problems because the predictions become less accurate as time passes.

However, a time series dataset is different. Time series adds an explicit order dependence between observations which is of course a time dimension.

This additional dimension is both a constraint and a structure that provides a source of additional information.

A Time Series Dataset may be said to be a sequence of observations taken sequentially in time.

Time Series Analysis can help us make better predictions. It involves developing models that best capture or describe an observed time series in order to understand the impact of time on the underlying classes.

Making predictions about the future is called extrapolation in the classical statistical handling of time series data. It is also referred to as Time Series Forecasting.

Forecasting involves taking models fit on historical data and using them to predict future observations. What's different about forecasting is tht the future is completely unavailable and must only be estimated from what has already happened.



What are the components of the Time Series:

The most useful components of time series :

Level → The Baseline value for the series

Trend → The linear increasing or decreasing of the behavior with respect to time

Seasonality → The Repeating cycles in trends with respect to a specific time of the year

Noise → The optional variability that cannot be explained by the model.

All-Time Series have a level and a must-have noise. The Trend and Seasonality are optional.

References: https://machinelearningmastery.com/time-series-forecasting/

## 📌Day 14: ⏳ Going in-depth with Time-Series Forecasting #02 ⌛
### Detecting Stationarity in Time Series Data
Unless our time series is stationary we cannot build a time series model.
A Time Series being stationary means that the statistical properties of a time series do not change over time. Bein stationary is important because many useful analytical tools and statistical tests and models rely on it.

There are 2 major reasons behind non-stationaruty of a TS:
1. Trend – varying mean over time. For eg, in this case we saw that on average, the number of passengers was growing over time.
2. Seasonality – variations at specific time-frames. eg people might have a tendency to buy cars in a particular month because of pay increment or festivals.

The criterions used to define stationarity are:
1. Constant Mean
2. Constant Variance
3. An autocovariance that does not depend on time.

***Checking Stationarity using python:***
We can check stationarity using the following:
1. Plotting Rolling Statistics: We can plot the moving average or moving variance and see if it varies with time. By moving average/variance we mean that at any instant ‘t’, we’ll take the average/variance of the last year, i.e. last 12 months. But again this is more of a visual technique.
2. Dickey-Fuller Test: This is one of the statistical tests for checking stationarity. Here the null hypothesis is that the TS is non-stationary. The test results comprise of a Test Statistic and some Critical Values for difference confidence levels. If the ‘Test Statistic’ is less than the ‘Critical Value’, we can reject the null hypothesis and say that the series is stationary.
```python
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput
test_stationary(ts)
```
Output:
![bird](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/stationary.png)

Though the variation in standard deviation is small, mean is clearly increasing with time and this is not a stationary series. Also, the test statistic is way more than the critical values.

The underlying principle to remove the stationarity in a time series dataset is to model or estimate the trend and seasonality in the series and remove those from the series to get a stationary series. Then statistical forecasting techniques can be implemented on this series. The final step would be to convert the forecasted values into the original scale by applying trend and seasonality constraints back.

References: [Detecting Stationary in time series data](https://www.kdnuggets.com/2019/08/stationarity-time-series-data.html#:~:text=Stationarity%20is%20an%20important%20concept%20in%20time%20series%20analysis.&text=Stationarity%20means%20that%20the%20statistical,and%20models%20rely%20on%20it)

## 📌Day 15: ⏳ Going in-depth with Time-Series Forecasting #03 ⌛
### ⌚ Time Series Forecasting on AirPassengers dataset: 
#### Loading and Handling Time Series in Pandas:
Before working with our data we need to parse our indices in the dataset into date format using the `datetime` module inorder to have better access to the data. We basically convert the dates that are in Object dtypes into `datetime64[ns]`. 

```python
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
# We use this to split our dates into a dateparse string
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
print (data.head())

# We then define ts as our time series 
ts = data['#Passengers']
ts.head(10)
```
#### Checking Stationarity in the Dataset: 
We also check for stationarity in the data which even though I did implement yesterday I did not have the right context for it. So let us look into both checking stationarity using Plot Rolling Statistics and Dickey-Fuller Method and removing the stationarity using the rolling average elimiation method.

```python
# Checking the Stationarity using these tests
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(12,5))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

test_stationarity(ts)
```
![before](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/beforefixing.png)
#### Test Results: 
***Rolling Statistics Test***: As we can see their is a very small variation in standard deviation but our mean clearly increases at an exponential rate. 

***Dickey-Fuller Test***: We see that the test statistic is much greater than the critical values so we accept the null hypothesis that TS is non-stationary.  

#### Fixing Stationarity in the Dataset:
We use one of the Smoothing method: Moving average to eliminate the trends
```python
ts_log = np.log(ts)
plt.plot(ts_lodatg)

moving_avg = ts_log.rolling(window=12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')

# Now let us eliminate the values contributing to the rolling average in our dataset.
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)

ts_log_moving_avg_diff.dropna(inplace=True)
ts_log_moving_avg_diff.head()
# Now let us once again test the Stationarity in our dataset
test_stationarity(ts_log_moving_avg_diff)
```
![wf](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/fixed.png)
##### Test Results:
1. Plot Rolling Statistics: As we can see both our mean and standard deviation are moving at a good constant rate through time indicating stationary.
2. Dickey-Fuller Test: We can see the Test Statistic value `-3.1` to be lesser than the critical values `-2.57` so we reject the null hypothesis that Time Series is non-statiionary.
 
## 📌Day 16: ⏳ Going in-depth with Time-Series Forecasting #04 ⌛
### Exponentially Weighted Average: 
This is another type of moving average that places a greater weight on the most recent data points unlike the simple moving average that applies an equal weight to all data points. 

Both EWM and simple WM are lagging indicators, i.e, an observable or measurable factor that changes sometimes after the domain it is correlated with changes. 

EWM smoothens out our average more than the Simple moving average, let us look at the implementation

```python
ts_log = np.log(ts)
expweighted_avg = ts_log.ewm(halflife=12).mean()
ts_log_ewma_diff = ts_log - expweighted_avg
test_stationarity(ts_log_ewma_diff)
```
![insertimage](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/EWM.png)
### Differencing:
This is a method of transforming a time series dataset. 
Differencing is performed by subtracting the previous observation from the current observation.

`difference(t) = observation(t) - observation(t-1)`

Inverting the process is required when a prediction must be converted back into the original scale.

This process can be reversed by adding the observation at the prior time step to the difference value.

`inverted(t) = differenced(t) + observation(t-1)`
Let us look at the implementation on the air_passengers dataset:
```python
#Take first difference:
plt.figure(figsize=(10,5))
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
plt.figure(figsize=(12,6))
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)
```
![differencing](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/differencing.png)

## 📌Day 17: ⏳ Going in-depth with Time-Series Forecasting #05 ⌛
### Decomposition on a Time Series Dataset: 
Time Series decomposition involves thinking of a series as a combination of level, trend, seasonality and noise components. 

This is used to inform forecasting models on our problem.
It provides a structured way of thinking about a time series both generally in terms of modelling complexity and specifically in terms of how to best capture each of these components in a given model.

Each of these components are something we may need to think about and address during data preparation, model selection and tuning. We may address it explicitly in terms of modeling the trend and subtracting it from our data, or implicitly by providing enough history for an algorithm to model a trend if it may exist. 

However we may not be able to break down our model as an additive or multiplicative model. Real-world problems are noisy and may have both additive and multiplicative components. There may be non-repeating cycles mixed in with repeating seasonal trends. 

The statsmodels library provides an implementation of the naive, or classical, decomposition method in a function called `seasonal_decompose()`. It requires that we specify whether the model is `additive` or `multiplicative`.


```python
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(10,5))
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
```
![dec](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/decomposition.png)

References: https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/

## 📌Day 18: 📊***Getting Started with Tableau***
Today marks the start for my first internship!
I managed to get selected for an internship at [The Sparks Foundation](https://www.thesparksfoundationsingapore.org/) for the Data Science and Business Analytics role and for the next month I will be executing 3 projects on differing domains and use visual analytics tools like Tableau to create interactive visual analytics in the form of dashboards. I will also be creating video demonstrations, explaining the approaches and steps i followed for the project and showcasing what the said project has achieved.

Before all that, of course I need to get familiar with Tableau. So I referred to a crash course from freecodecamp to understand the interface of Tableau which was pretty intuitive. I made a very basic dashboard with the Titanic Dataset. 

![lib](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/tab.png)

Today was all about planning out the next few weeks with the projects. Tomorrow I will start with the first project - ***Exploratory Data Analysis on Retail data for a superstore*** 

## 📌Day 19: 📊***Building Interactive Dashboards with Tableau***
### 📊Building Interactive Dashboard for EDA with Sample Superstore dataset
Tableau is amazing. Ngl I've never had this much fun with data before xD, made some interesting inferences with the SampleSuperstore dataset, and I am mostly done, other than of course, organizing and applying some better practices for the sheets. I can say I am at least familiar now with the interface to a good extent. I will put some more work into the dashboard before making the video demonstration for the project. 

In this project, my task is to find out what factors the losses are being incurred for the company and what we can do to reduce those losses. After some EDA with Tableau, I was able to determine that the losses are the highest and most significant in the following cities: 

Philadelphia, Houston, Chicago.

And the profit is the highest in cities like:

New York, Seattle, San Francisco, Los Angeles.

The losses are highest with the Category of furniture due to a disproportion between sales and discounts in the cities with losses. 

There is a lot more to uncover which I will be explaining in the video demonstration.

![project1](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/project1.png)

## 📌Day 20: 📊***Building Interactive Dashboards with Tableau #02***
Today I completed building and organizing the dashboard for the SampleSuperstore dataset. In the displayed worksheet I have taken the size of the circles as the number of sales in any particular category. The BLUE color is on the positive side of profit whereas RED is on the negative side of it.
### 👁 Overview of the Dashboard
#### Profit/Sales by City
For this category, I have designated the size of the circles as indicators for the number of sales in any particular city. From that knowledge, we can deduce that:

New York, Seattle, San Francisco, Los Angeles, Houston, and Philadelphia have the most considerable amount of sales.

Also, consider the colors of the circles we can deduct the following:

New York, Seattle, San Francisco, Los Angeles are making us an immense amount of profit with New York being the most profitable with the darkest shade of blue.

Philadelphia, Houston, and Chicago are in shades of red and are in fact experiencing negative profit,i.e, losses. We need to understand the factors affecting these regions and implement the practices from the blue regions onto these regions.
#### Profit/Sales by Ship Mode
Ship Mode is the means by which our goods are shipped to the customer. A shipping model is a combination of a shipping carrier and the shipping service that is offered by that carrier.

In the overall circle chart, we can see that the Standard Model of shipping has both a higher number of sales and the higher number of profits. Whereas First class, Second Class, and Same Day are going at a loss. The company needs to increase the cost for the First Class, Second Class, and Same Day shipping modes to decrease the losses.

#### Profit/Sales by Region
We see East and West having both the highest number of sales and profit whereas Central and South are going at an immense loss with Central incurring the heaviest losses. We thus need to understand the practices in the West region and apply them to the Central and South regions.

#### Profit/Sales by Segment
We see Consumers having a large amount of profit whereas Corporate and Home Office going at a loss. This can contribute to a number of factors like heavier discounts because of bulk buying by the Corporate and Home Office segments.

#### Profit/Sales by Category
Here Sales are in YELLOW and Profit is the same as before (-RED->0->+BLUE) Here we go to the root of it all. We can see while all three of the categories have a desirable amount of sales the losses in the furniture category are extremely concerning. In order to understand this, we go through our Subcategories and analyze them by Profit, Sales, and Discount on the products.

#### Profit/Sales/Discount by Subcategories

Here we have Profited as (-RED->0->+BLUE) Sales as (YELLOW) and Discount as (TEAL BLUE)

Looking into the Furniture sub-categories we can clearly see that the tables and bookcases are running at losses. This is probably due to a disproportionate amount of discount on the said sub-categories as we can see from the graphs.

On the other hand, we have Office Supplies running at a decent profit with some loss for Supplies.

Our Technology sub-categories as we can see are doing really well however Machines are running at a high discount rate which may be reduced to increase the profit.

I have more to discuss here, I will do that in the video.


![dashboard](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/dashboard.png)

## 📌Day 21: ***📊Building Interactive Dashboards with Tableau #03***
Today was all about scripting and recording the video so barely had any time to study anything else. Meanwhile took a little time to just read up on some older topics. 

Ended up going through this blog post by Karpathy again for the day: 
https://karpathy.github.io/2015/05/21/rnn-effectiveness/

## 📌Day 22: ***📊EDA with Sample Superstore Dataset***
Following are the conclusions I came up with after deeper analysis of the Sample Superstore dataset using Tableau. There are probably more, but I have more projects to cover so I skipped going more in depth with them. 

- Most amount of losses are happening in the Furnitures category, with tables taking over most of the losses.  Reducing the manufacture for the tables in most of the regions experiencing losses will probably lead to better profits for the company and cutting out on the discounts for the furnitures as well. 
- Our Machines sub-category from technologies is having a large amount of discount offered on it throughout the various regions that is causing losses so we need to reduce the discount on it to gain better profits out of it.
- For our Central Region more profit is observed in the corporate segment whereas more losses in the Consumer segment. So we need to transfer the goods that are making losses in the Customer segment  into the Corporate segment to make a good profit out of them. 
- Ship mode has a minimal amount of variation between the different regions with Standard Mode of shipping always having the most amount of sales and profit. 
- Philadelphia is an abnormality with the most amount of losses in all the categories even when it has a decent amount of sales, mainly because of the large amount of discounts that are being offered for the different sub-categories in the city. Of course there may be more reasons for this and there might be some more data that we are not seeing for this abnormality. 

## 📌Day 23: 🤖 OpenAI GPT-3 
Today I got curious about the model behind Github's copilot and so did a little overview on the GPT-3 and this model honestly blew me away. Specifying such a small amount of data to get amazing results is honestly revolutionary. Maybe I am late to the party xD, but still I am glad I checked it out. Here's a summary---

GPT-3 has been created by OpenAI, Put simply, it is an AI that is better at creating content that has a language structure – human or machine language – than anything that has come before it.

First of all GPT stands for Generative Pretrained Transformer. In short this means that it generates text using algorithms that are pretrained,i.e, they have already been fed all of the data they need to carry out the task that we specify. 

GPT-3 has more than 175Billion parameters to it which is mind boggling. In other words it has 570GB of text information gathered by crawling the internet. Example: Apparently a publicly available dataset called CommonCrawl was used along with other texts selected by OpenAI, including the text of Wikipedia. 

If you ask it a question, you would expect the most useful response would be an answer. If you ask it to carry out a task such as creating a summary or writing a poem, you will get a summary or a poem.

More technically, it has also been described as the largest artificial neural network every created!

Today I went through a small implementation of GPT-3 to convert simple English text into SQL like query text -- 


```python
!pip install openai
import json
import openai

with open('GPT_SECRET_KEY.json') as f:
    data = json.load(f)


openai.api_key = data["API_KEY"]

from gpt import GPT
from gpt import Example

gpt = GPT(engine="davinci",
          temperature=0.5,
          max_tokens=100)

# Adding Examples for the GPT model
gpt.add_example(Example('Fetch unique values of DEPARTMENT from Worker table.', 
                        'Select distinct DEPARTMENT from Worker;'))
gpt.add_example(Example('Print the first three characters of FIRST_NAME from Worker table.', 
                        'Select substring(FIRST_NAME,1,3) from Worker;'))

gpt.add_example(Example("Find the position of the alphabet ('a') in the first name column 'Amitabh' from Worker table.", 
                        "Select INSTR(FIRST_NAME, BINARY'a') from Worker where FIRST_NAME = 'Amitabh';"))

gpt.add_example(Example("Print the FIRST_NAME from Worker table after replacing 'a' with 'A'.", 
                        "Select CONCAT(FIRST_NAME, ' ', LAST_NAME) AS 'COMPLETE_NAME' from Worker;"))

gpt.add_example(Example("Display the second highest salary from the Worker table.", 
                        "Select max(Salary) from Worker where Salary not in (Select max(Salary) from Worker);"))

gpt.add_example(Example("Display the highest salary from the Worker table.", 
                        "Select max(Salary) from Worker;"))

gpt.add_example(Example("Fetch the count of employees working in the department Admin.", 
                        "SELECT COUNT(*) FROM worker WHERE DEPARTMENT = 'Admin';"))

gpt.add_example(Example("Get all details of the Workers whose SALARY lies between 100000 and 500000.", 
                        "Select * from Worker where SALARY between 100000 and 500000;"))

gpt.add_example(Example("Get Salary details of the Workers", 
                        "Select Salary from Worker"))

# EXAMPLES

prompt = "Display the lowest salary from the Worker table."
output = gpt.submit_request(prompt)
output.choices[0].text
'''
'output: Select min(Salary) from Worker;\n'

'''
prompt = "Tell me the count of employees working in the department HR."
output = gpt.submit_request(prompt)
output.choices[0].text
'''
"output: SELECT COUNT(*) FROM worker WHERE DEPARTMENT = 'HR';\n"

'''
```
## 📌Day 24: 🌳Prediction using Decision Tree Algorithm📈
Decision Tree is one of the most commonly used, practical approaches for supervised learning. It can also be used for Classification tasks. The decision of making strategic splits heavily affects a tree’s accuracy. The decision criteria is different for classification and regression trees.Decision trees regression normally use mean squared error (MSE) to decide to split a node in two or more sub-nodes. For each subset, it will calculate the MSE separately. The tree chooses the value with results in smallest MSE value.

The purpose of this project is to create a Decision Tree Classifier and visualize it graphically. Feeding any data into this classifier, the model should be able to predict the right class of the said element.

```python
# importing the libraries 
from sklearn.datasets import load_iris                             # We will be using the iris dataset for this project
from sklearn.tree import DecisionTreeClassifier, export_graphviz   # graphviz is used to visualize our decision tree classifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from io import StringIO
import pydot # is an interface to Graphviz
from IPython.display import Image

## Step 1 - Loading the Dataset
# Loading Dataset
iris = load_iris()
X = iris.data[:,:]
y = iris.target

## Step 2 - Decision Tree Modelling 
# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1) 
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train,y_train)
print("Training Complete.")
y_pred = tree_classifier.predict(X_test)

## Step 3 - Model Visualization
# Visualizing the trained Decision Tree Classifier taking all 4 features in consideration
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

export_graphviz(
        tree_classifier,
        out_file="img\desision_tree.dot",
        feature_names=iris.feature_names[:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
)
(graph,) = pydot.graph_from_dot_file('img\desision_tree.dot')
graph.write_png('img\desision_tree.png')
Image(filename='img\desision_tree.png') 

```
Output: 
![dtree](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/dtree.png)
```python


## Step 4 - We test our model on some random variables
print("Class Names = ",iris.target_names)
# Estimating class probabilities
print()
print("Estimating Class Probabilities for flower whose petals length width are 4.7cm and 3.2cm and sepal length and width are 1.3cm and 0.2cm. ")
print()
print('Output = ',tree_classifier.predict([[4.7, 3.2, 1.3, 0.2]]))
print()
print("Our model predicts the class as 0, that is, setosa.")

'''
Output: 
Class Names =  ['setosa' 'versicolor' 'virginica']

Estimating Class Probabilities for flower whose petals length width are 4.7cm and 3.2cm and sepal length and width are 1.3cm and 0.2cm. 

Output =  [0]

Our model predicts the class as 0, that is, setosa.
'''
## Step 8 - Calculating the Model accuracy 
print("ACCURACY: ", sm.accuracy_score(y_test, y_pred))

'''
Output: 
ACCURACY:  1.0

'''
```

## 📌Day 25: 🌳Understanding Decision Trees Algorithms
A decision tree works in a step wise manner where we have a tree structure where the nodes are split using a feature based on some criterion. 

Today we look into how these features get selected.

There are 3 main splitting criteria for decision trees: 

1. Gini Impurity: In simple terms, this is the measure of impurity of a node. In more technical terms -  
A measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.

![dtree](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/gini.png)

2. Entropy: Entropy is th measure of randomness in the system. 

![dtree](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/entr.png)

3. Variance: Gini and Entropy work well only for classification scenarios, however in case of regeression we use the weighted variance of the nodes.

![dtree](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/var.png) 

4. Information Gain: A Statistical Property that governs how well a given attribute separates the training examples according to their target classification.

The basic idea behind the algorithm: 
1. We select the best attributes using the Attribute selection measures which are the above 3 splitting criteria we have defined. 
2. Make that attribute a decision node and break the dataset into smaller subsets.
3. Start tree building by repeating this process recusively for each chuld until there are no more remaining attributes.

References: https://datascience.foundation/sciencewhitepaper/understanding-decision-trees-with-python#:~:text=We%20will%20be%20using%20the,%2D%20petal%20length%20%2D%20petal%20width.&text=There%20are%20three%20classes%20of,versicolor'%20and%20'virginica'.

## 📌Day 26: 🪓 The Gini Impurity Measure
Today I went into the mathematics behind Gini index and also recorded the video for the 2nd project which I should be editing and posting tomorrow. 

The Gini impurity measure is one of the methods used in decision tree algorithms to decide the optimal split from a root node, and subsequent splits.

To put it into context, a decision tree is trying to create sequential questions such that it partitions the data into smaller groups. Once the partition is complete a predictive decision is made at this terminal node (based on a frequency).

Suppose we have a list of observations, that indicates if a person decided to stay home from work. We also have two features, namely if they are sick and their temperature.
We need to choose which feature, emotion or temperature, to split the data on. A Gini Impurity measure will help us make this decision.
 
Gini Impurity tells us what is the probability of misclassifying an observation.
Note that the lower the Gini the better the split. In other words the lower the likelihood of misclassification.

```python
def Ginx(P1,P2):
    #P1 and P2 are the counts for each class after the split
    denom = P1 + P2
    Ginx = 2 * (P1/denom) * (P2/denom)
    return(Ginx)

def Wght_Ginx(G1,G2,PL,PR):
    # G1 G2 are the gini impurity for each split, and PL PR are the proportion of the split
    WG = PL * G1 + PR * G2
    return(WG)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
X=np.arange(0.0,1.0,0.01)
Y=X[::-1]

Gini=Ginx(X,Y)
plt.plot(X,Gini)
plt.axhline(y=0.5,color='r',linestyle='--')
plt.title('Gini Impurity Graph')
plt.xlabel('P1=1')
plt.ylabel('Impurity Measure')
plt.ylim([0,1.1])
plt.show()
```
![gini1](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/gini1.png) 

## 📌Day 27: 📉 What are PairPlots exactly? 📈 
Since we cannot visualize multiple dimensions in a single graph, for that purpose we use pairpolots that is basically a matrix of the features distributed in a way to understand the distribution of the features more easily. For example:

Let us first plot the pairplot using the `seaborn.pairplot` library method in python for the iris dataset. 
```python
# Input data visualization
plt.close();
sns.set_style("whitegrid")
sns.pairplot(df, hue = "Species");
plt.show()
```
![pplot](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/pairplot.png)

From our pairplot we determine the distributions for each Species of flowers.

#### Observations from the pairplot: 
1. Petal Length andPetal Width are the most useful features to identify various flower types.
2. White Setosa can be easily identified (linearly separable), Virginica and Versicolor have some overlap (almost linearly separable)
3. We can find "lines" and "if-else" conditions to build a simple model to classify the flower types.

## 📌Day 28: 👁 EDA with pandas practice
***About the dataset:***
We will look into different ways of plotting data with python by utilizing data from the World Happiness Report 2019. The author of the below article has enriched the World Happiness Report data with information from Gapminder and Wikipedia to allow for the exploration of new relationships and visualizations.

Reference: https://towardsdatascience.com/plotting-with-python-c2561b8c0f1f

The dataset contains values for the following columns:
- Year: The year of measurement (from 2007 to 2018)
- Life Ladder: respondents measure of the value their lives today on a 0 to 10 scale (10 best) based on Cantril ladder
- Log GDP per capita: GDP per capita is in terms of Purchasing Power Parity (PPP) adjusted to constant 2011 international dollars, taken from the World - Development Indicators (WDI) released by the World Bank on November 14, 2018
- Social support: Answer to question: “If you were in trouble, do you have relatives or friends you can count on to help you whenever you need them, or not?”
- Healthy life expectancy at birth: Life expectancy at birth is constructed based on data from the World Health Organization (WHO) Global Health Observatory data repository, with data available for 2005, 2010, 2015, and 2016.
- Freedom to make life choices: Answer to question: “Are you satisfied or dissatisfied with your freedom to choose what you do with your life?”
- Generosity: Responses to “Have you donated money to a charity in the past month?” compared to GDP per capita
- Perceptions of corruption: Answer to “Is corruption widespread throughout the government or not?” and “Is corruption widespread within businesses or not?”
- Positive affect: comprises the average frequency of happiness, laughter, and enjoyment on the previous day.
- Negative affect: comprises the average frequency of worry, sadness, and anger on the previous day.
- Confidence in national government: Self-explanatory
- Democratic Quality: how democratic is a country
- Delivery Quality: How well a country delivers on its policies
- Gapminder Life Expectancy: Life expectancy from Gapminder
- Gapminder Population: Population of a country

The World Happiness Report tries to answer which factors influence happiness throughout the world. Happiness in the report is defined as the responses to the “Cantril ladder question” asking respondents to value their lives today on a 0 to 10 scale, with the worst possible life as a 0 and the best possible life as a 10. The World Happiness Report tries to answer which factors influence happiness throughout the world. Happiness in the report is defined as the responses to the “Cantril ladder question” asking respondents to value their lives today on a 0 to 10 scale, with the worst possible life as a 0 and the best possible life as a 10.

![pplot1](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/plotting1.png)

![pplot2](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/plotting2.png)

![pplot3](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/plotting3.png)

![pplot4](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/plotting4.png)

### Conclusion on plotting with Pandas: 
Plotting with pandas is convenient. It is easily accessible, and it is fast. The plots are fairly ugly. Deviating from defaults is borderline impossible, and that is okay because we have other tools for more aesthetically appealing charts. Moving on to seaborn.

## 📌Day 29: 👁 EDA with seaborn practice
Seaborn utilizes plotting defaults. To make sure that your results match mine, run the following commands.
### Plotting univariate distributions
Histograms and kernel density distributions alike are potent ways of visualizing the critical features of a particular variable. Let’s look at how we generate distributions for a single variable or distributions of multiple variables in one chart.

![sea1](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/sea1.png)
Left chart: Histogram and kernel density estimation of “Life Ladder” for Asian countries in 2018; Right chart: Kernel density estimation of “Life Ladder” for five buckets of GDP per Capita — Money can buy happiness

### Plotting bivariate distributions
Exploring the relationship between two or multiple variables visually, it typically comes down to some form of scatterplot and an assessment of distributions. There are three variations of a conceptually similar plot. In each of those plots, the center graph (scatter, bivariate KDE, and hexbin) helps to understand the joint frequency distribution between two variables. Additionally, at the right and top border of the center graph, the marginal univariate distribution of the respective variable is depicted (as a KDE or histogram).

```python
sns.jointplot(
    x='Log GDP per capita',
    y='Life Ladder',
    data=data,
    kind='scatter' # or 'kde' or 'hex'
)
```
![sea2](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/sea2.png)

Seaborn jointplot with scatter, bivariate kde, and hexbin in the center graph and marginal distributions left and on top of the center graph.

### Scatterplot
A scatterplot is a way of visualizing the joint density distribution of two variables. We can throw a third variable in the mix by adding a hue and a fourth variable by adding the size parameter.

```python
sns.scatterplot(
    x='Log GDP per capita',
    y='Life Ladder',
    data=data[data['Year'] == 2018],    
    hue='Continent',
    size='Gapminder Population'
)
# both, hue and size are optional
sns.despine() # prettier layout
```
![sea3](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/sea3.png)

Log GDP per capita against Life Ladder, colors based on the continent and size on population


## 📌Day 30: Prediction using simple linear regression 
**Aim of the project**: Predict the percentage of an student based on the no. of study hours.
What will be predicted score if a student studies for 9.25 hrs/ day?

Dataset: https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv

Simple Linear Regression: it concerns two-dimensional sample points with one independent variable and one dependent variable (conventionally, the x and y coordinates in a Cartesian coordinate system) and finds a linear function (a non-vertical straight line) that, as accurately as possible, predicts the dependent variable values as a function of the independent variable. The adjective simple refers to the fact that the outcome variable is related to a single predictor.(Wiki)

```python
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

url = r"https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)

df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()
```
![l1](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/l1.png)

```python
X = df.iloc[:,:-1].values
y = df.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
regressor = LinearRegression()  
regressor.fit(X_train.reshape(-1,1), y_train) 

# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,color='red');
plt.show()
```
![l2](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/l2.png)

```python
# Model Prediction 
y_pred = regressor.predict(X_test)
#Estimating training and test score
print("Training Score:",regressor.score(X_train,y_train))
print("Test Score:",regressor.score(X_test,y_test))
'''
Training Score: 0.9515510725211552
Test Score: 0.9454906892105354
'''
# Plotting the Bar graph to depict the difference between the actual and predicted value

df.plot(kind='bar',figsize=(5,5))
plt.grid(which='major', linewidth='0.5', color='red')
plt.grid(which='minor', linewidth='0.5', color='blue')
plt.show()
```
![l3](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/l3.png)
```python
# Testing the model with our own data
hours = 9.25
test = np.array([hours])
test = test.reshape(-1, 1)
own_pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
'''
No of Hours = 9.25
Predicted Score = 93.69173248737539
'''

from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-2:', metrics.r2_score(y_test, y_pred))
'''
Mean Absolute Error: 4.183859899002982
Mean Squared Error: 21.598769307217456
Root Mean Squared Error: 4.647447612100373
R-2: 0.9454906892105354
'''
```
We thus get a great R-2 score depicting 94% of accuracy in the mode. 


## 📌Day 31: Interpreting R-squared in Regression Analysis 
R-squared is a goodness-of-fit measure for linear regression models.
R-squared measures the strength of the relationship between our model and the dependent variable on a convenient 0 – 1 scale.

After fitting a linear regression model, you need to determine how well the model fits the data. Linear regression identifies the equation that produces the smallest difference between all the observed values and their fitted values. To be precise, linear regression finds the smallest sum of squared residuals that is possible for the dataset.
(Residuals are the distance between the observed value and the fitted value.)

Residual plots can expose a biased model far more effectively than the numeric output ( R-squared ) by displaying problematic patterns in the residuals. If our model is biased, you cannot trust the results. If your residual plots look good, go ahead and assess our R-squared and other statistics.

R-squared evaluates the scatter of the data points around the fitted regression line. It is also called the coefficient of determination. For the same data set, higher R-squared values represent smaller differences between the observed data and the fitted values and thus an overall better model. 

R-squared is always between 0 and 1:

- 0 represents a model that does not explain any of the variation in the response variable around its mean. The mean of the dependent variable predicts the dependent variable as well as the regression model.
- 1 represents a model that explains all the variation in the response variable around its mean.
- Usually, the larger the R2, the better the regression model fits your observations. 

```R^2 = Variance explained by the model / Total Variance```

![rsq](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/rsq.png)

The R-squared for the regression model on the left is 15%, and for the model on the right it is 85%. When a regression model accounts for more of the variance, the data points are closer to the regression line. In practice, we’ll never see a regression model with an R^2 of 100%. In that case, the fitted values equal the data values and, consequently, all the observations fall exactly on the regression line.

## 📌Day 32: K-means Clustering 
Clustering is used to get an intuition about the structure of the data. 

It is used for creting different sub-groups in a data, such that the data in the sub-groups are all widely different from each other. 

This is an unsupervised learning algorithm since we do not have any ground truth to compare the output of our clustering algorithm to the true labels and evaluate its performance. 

K-means is one of the most used clustering algorithm due to it's simplicity. 

It uses an iterative approach where it tries to partition the dataset into K pre-defined distinct non-overlapping subgroups of course, depending on the dataset.  

It assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster’s centroid (arithmetic mean of all the data points that belong to that cluster) is at the minimum. The less variation we have within clusters, the more similar the data points are within the same cluster.

How it works? 

1. Specify the number of clusters. 
2. We initialize the centroids by shuffling the dataset and then randomly selecting K data points for the centroids without replacement. 
3. Keep iterating until there are no changes in the centroids ( meaning the data points of the clusters are not changing )

The approach kmeans follows to solve the problem is called Expectation-Maximization. The E-step is assigning the data points to the closest cluster. The M-step is computing the centroid of each cluster.


## 📌Day 33: YOLO → You Only Look Once
Today I implemented the unsupervised ML clustering project for the internship and moved on to the Computer vision projects. I'll be working on a Social Distance Detector project, to detect if two or more people are in a proximity of 2M(metres) between each other. If their proximity < 2M we return a high chance of exposure to the disease, and if >=2M we return safe. 

I will be using YOLO for this, but for that a lot of installations had to be done, the list goes like: 
- CUDA 
- CUDANN (Neural Network for CUDA)
- CMAKE 
- Compatible Open CV files

And I am still not done with setting it up yet. Maybe I should be allocating more time for the sessions.

Anyway, what is YOLO exactly 

YOLO is an algorithm that uses neural networks to provide real-time object detection. This algorithm is popular because of its speed and accuracy. It has been used in various applications to detect traffic signals, people, parking meters, and animals.

Object detection in YOLO is done as a regression problem and provides the class probabilities of the detected images. YOLO algorithm employs convolutional neural networks (CNN) to detect objects in real-time. As the name suggests, the algorithm requires only a single forward propagation through a neural network to detect objects.

This means that prediction in the entire image is done in a single algorithm run. The CNN is used to predict various class probabilities and bounding boxes simultaneously.

Presently the latest version is YOLO v4.

You may read the official YOLO v4 research paper here: https://arxiv.org/abs/2004.10934

![yolo2](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/yolo2.png)

## 📌Day 34: Social Distancing Detector #01
This project basically has 3 projects integrated in it -> 
object detection, object tracking and measuring the distances between the objects. 

First we implement for detecting the people. We first define our: 
- Minimum probability to filter out weak detections
- The NMS threshold when applying non-maxima suppression
- Minimum safe distance between two people(we may consider this to be 2M for now)

We then use cv2 to preprocess our images in the detect_people function. Here we define our frames, our darknet framework(as net), the layer names from darknet (as ln) and we get the personIdx to get position for each person dynamically in the image. ( Things will be clearer tomorrow, today I am just playing around with the code )

```python
# initialise minimum probability to filter out weak detections. 
MIN_CONF = 0.3  
# We also define the threshold when applying non-maxima suppression
NMS_THRESH = 0.3

# We define the minimum safe distance that two people can be at
MIN_DISTANCE = 50 # This is defined in pixels, not standard metrics

# importing the necessary packages
import numpy as np
import cv2

# Inside our function we define the frame that we display around the people
def detect_people(frame, net, ln, personIdx=0):
  # We get the dimensions of the frame 
  (H,W) = frame.shape[:,2]
  results = []

  blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
  net.setInput(blob)
  layerOutputs = net.forward(ln)

	# initialize our lists of detected bounding boxes, centroids, and
	# confidences, respectively
  boxes = []
  centroids = []
  confidences = []

	# loop over each of the layer outputs
  for output in layerOutputs:
		# loop over each of the detections
  	for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
  		scores = detection[5:]
  		classID = np.argmax(scores)
  		confidence = scores[classID]

			# filter detections by (1) ensuring that the object
			# detected was a person and (2) that the minimum
			# confidence is met
  		if classID == personIdx and confidence > MIN_CONF:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
  			box = detection[0:4] * np.array([W, H, W, H])
  			(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
  			x = int(centerX - (width / 2))
  			y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# centroids, and confidences
  			boxes.append([x, y, int(width), int(height)])
  			centroids.append((centerX, centerY))
  			confidences.append(float(confidence))

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
  idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)



	# ensure at least one detection exists
  if len(idxs) > 0:
		# loop over the indexes we are keeping
  	for i in idxs.flatten():
			# extract the bounding box coordinates
  		(x, y) = (boxes[i][0], boxes[i][1])
  		(w, h) = (boxes[i][2], boxes[i][3])

			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
  		r = (confidences[i], (x, y, x + w, y + h), centroids[i])
  		results.append(r)

	# return the list of results
  return results

```

## 📌Day 35: Social Distancing Detector #02
Completed the project implementation wise but there are some interesting concepts to this that I will discuss later on. For now, the way the detector works-
Detects the people who are not at the required desirable distances between each other and increment the value for the violate variable by 1. We can see the actual visualization below. The value for violate may keep changing in real-time for every frame in our video depending on the object's position. I will add the distances between them later.
```python
# USAGE
# python social_distance_detector.py --input pedestrians.mp4
# python social_distance_detector.py --input pedestrians.mp4 --output output.avi

# import the necessary packages
from google.colab.patches import cv2_imshow
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args(["--input","/content/drive/My Drive/social-distance-detector/pedestrians.mp4","--output","my_output.avi","--display","1"]))

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["/content/drive/My Drive/social-distance-detector/coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["/content/drive/My Drive/social-distance-detector/yolov3.weights"])
configPath = os.path.sep.join(["/content/drive/My Drive/social-distance-detector/yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# loop over the frames from the video stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# resize the frame and then detect people (and only people) in it
	frame = imutils.resize(frame, width=700)
	results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

	# initialize the set of indexes that violate the minimum social
	# distance
	violate = set()

	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	if len(results) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of pixels
				if D[i, j] < MIN_DISTANCE:
					# update our violation set with the indexes of
					# the centroid pairs
					violate.add(i)
					violate.add(j)

	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# if the index pair exists within the violation set, then
		# update the color
		if i in violate:
			color = (0, 0, 255)

		# draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)

	# draw the total number of social distancing violations on the
	# output frame
	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

	# check to see if the output frame should be displayed to our
	# screen
	if args["display"] > 0:
		# show the output frame
		cv2_imshow(frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)
  # if the video writer is not None, write the frame to the output
	# video file
	if writer is not None:
		writer.write(frame)

```
![mall2](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/mall.png)


## 📌Day 36: Non-Maxima Suppression
NMS is a technique used mainly in object detection that aims at selecting the best bounding box out of a set of overlapping boxes. 

IOU is a term used to describe the extent of overlap of two boxes. The greater the region we overlap the greater is the value of IOU

![nms](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/nms.jpeg)

### ***Algorithm***
1. We define a value for `confidence_threshold` and `IOU_threshold`. 
2. Sort the bounding boxes in a descending order of confidence.
3. Remove the boxes that have a `confidence` < `confidence_threshold`
4. Loop over all the boxes in descending order of confidence. 
5. Calculate the IOU of the current box with every remaining box that belong to the same class. 
6. If the IOU of the 2 boxes > `IOU_threshold`, we remove the box with a lower confidence from our list of boxes. 
7. We repeat this operation until we have gone through all the boxes in the list. 
### `CODE`
```python
def nms(boxes, MIN_CONF=0.7, MIN_IOU=0.4):
    box_list = []
    box_list_new = []
    # Stage-1__We sort the boxes and filter out the ones with the low confidence
    boxes_sorted = sorted(boxes, reverse = True, key = lambda x: x[5])
    for box in boxes_sorted:
        if box[5]>MIN_CONF:
            box_list.append(box)
        else:
            pass
    
    # Stage-2__We now loop over all the boxes and remove the boxes that have a high IOU
    while(len(box_list)>0):
        current_box = box_list.pop(0)
        box_list_new.append(current_box)
        for box in box_list:
            if current_box[4] == box[4]:
                iou = IOU(current_box[:4], box[:4])
                if iou>MIN_IOU:
                    box_list.remove(box)
    return box_list_new

def IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
```
![nms1](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/nms2.jpeg)
References: https://medium.com/analytics-vidhya/non-max-suppression-nms-6623e6572536
https://medium.com/analytics-vidhya/iou-intersection-over-union-705a39e7acef


## 📌Day 37: Color-Detection using OpenCV
Today I made a color identification application using OpenCV that used the colors.csv dataset for identifying the colors from any given point in an image when we double click on it, by giving us the respective names and the corresponding RGB values for that particular point in the image.

```python
# Importing the important libraries
import cv2
import pandas as pd

img_path = r'crazynoisybizarreworld.png'
new_img_path = r'new_img.jpg'
img = cv2.imread(img_path)

# declaring global variables (are used later on)
clicked = False # mouse pointer set to false before double click
r = g = b = x_pos = y_pos = 0 # mouse pointer position all are set to zero firstly

# Reading csv file with pandas and giving names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)

# A function to calculate minimum distance from all colors and get the most matching color
def get_color_name(R, G, B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname

# A function to get x,y coordinates of mouse pointer when left button double clicked
def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, x_pos, y_pos, clicked
        clicked = True
        x_pos = x
        y_pos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

# Creating a window called Image 
cv2.namedWindow("Image - Double Click on Image to See The Colour and Press 'Esc' Button to Exit")
cv2.setMouseCallback("Image - Double Click on Image to See The Colour and Press 'Esc' Button to Exit", draw_function)

while True:

    cv2.imshow("Image - Double Click on Image to See The Colour and Press 'Esc' Button to Exit", img)
    if clicked:

        # cv2.rectangle(image, start point, endpoint, color, thickness)-1 fills entire rectangle
        cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)

        # Creating text string to display( Color name and RGB values )
        text = get_color_name(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)

        # cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
        cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # For very light colours we will display text in black colour
        if r + g + b >= 600:
            cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        clicked = False

    # Break the loop when user hits 'esc' key
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
```
![jojo1](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/jojo.png)

I also made a youtube video explaining the process of how the algorithm works and exhibited an overall implementation of the project: 
https://www.youtube.com/watch?v=-BDKSZTCtGA&t=10s 

This was the last project I implemented from the Sparks Foundation Internship and I learnt a lot from them. When I started doing the first video explanation for EDA retail I honestly struggled a lot, I was exhausted even. Now after making 5 video explanations for different projects I can say that I have gained a reliable experience in the explaining side of things atleast and that was the most valuable experience I gained from this internship.


## 📌Day 38: ⌛ Time Series Forecasting #4 ⏳
Now that my internship is over it's time to get back to the usual. 

Today I got back to Time Series Forecasting. I put it on halt cause of the internship since there were no projects related to it in there.

### ARIMA -> Auto Regressor Integrated Moving Average
An ARIMA model is a form of regression analysis that gauges the strength of one dependent variable relative to other changing variables. 

Instead of predicting the time series itself, we use ARIMA in predicting the differences of the time series from one time stamp to the next time stamp.  

- AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.
- I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
- MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.

Each of these components are explicitly specified in the model as a parameter. A standard notation is used of `ARIMA(p,d,q)` where the parameters are substituted with integer values to quickly indicate the specific ARIMA model being used.

`ARIMA(1,1,1)` is the simplest ARIMA model. 

The parameters of the ARIMA model are defined as follows:

- p: The number of lag observations included in the model, also called the lag order.
- d: The number of times that the raw observations are differenced, also called the degree of differencing.
- q: The size of the moving average window, also called the order of moving average.

```python
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  

plt.figure(figsize=(12,5))
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
```
![arima](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/arima.png)

I'll go into more depth tomorrow, this was just an introduction. 

Reference: https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

## 📌Day 39: ⌛ Time Series Forecasting #5 ⏳
### the ACF and PACF 

The ACF stands for Autocorrelation function and PACF stands for Partial Autocorrelation function. Looking at these two plots together can help us form and idea of what models to fit. 

ACF is the correaltion between the observation at the current time stamp and the observation at the previous time stamps all having an indirect effect on the current time stamp. 

PACF on the other hand is the correlation at two time spots in such a way that we don't consider all the other indirect effects on the time stamps between them. 

For example: The price of petrol today can be correlated to the day before yesterday, and yesterday can also be correalted to the day before yesterday. Then PACF of yesterday is the "real" correlation between today and yesterday after taking out the influence of the day before yesterday. 

For finding ACF we simply use the pearson correlation algorithm lining up our dataset comparing the previous time periods to the present one. 

However for PACF we build a regression function and getting the coefficient of that term. 

In our ARIMA model we use ACF and PACF plots to detect the AR(Auto Regression) and MA(Mean Average) in our time series. 

```python
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf  

lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF:  
#plt.figure(figsize=(10,5))
plt.subplot(121)    
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
#plt.figure(figsize=(10,5))
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
```
![auto](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/autocorr.png)

## 📌Day 40: Understanding the Intuition for Ridge Regression
The difference between Ridge Regression and Simple Linear Regression(OLS regression) is that the former implements a penalty to the loss function to decrease the variance for the model and prevent overfitting. 

Ridge regression has a low sensitivity to changes made in the model parameters because of the additional penalty we have in the loss function. This is not the case for OLS regression since very small changes can result in large changes in the regression model. 

In more mathematical terms: 

When implementing Ridge Regression using gradient descent, the added L2 regularisation ( the penalty term ) terms lead to reducing the weights of our model to 0 or close to 0. Because of this penalization, our model gets less prone to overfitting since it gets simpler or more generalized. 

### The Cost Function for Linear Regression 

![line](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/linreg.png)

h(x(i)) represents the hypothetical function for prediction.

y(i) represents the value of target variable for ith example.

m is the total number of training examples in the given dataset.

### Modified to Ridge Regression by adding the additional penalty terms (lambda)

![rid](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/Ridgereg.png)

Here, lambda is our penalty

wj represents the weight for jth feature.


References: https://machinelearningcompass.com/machine_learning_models/ridge_regression/


## 📌Day 41: L1 and L2 Regularisation 
These regularisation techniques are used to address overfitting and feature selection.

A regression model that uses the L1 regularisation technique is called Lasso Regression and the model which uses L2 regularisation is Ridge Regression. 

The key difference between these two is the penalty term that is added to the loss functions for the algorithms. 

Ridge Regression adds "square magnitude" of coefficient as penalty term to the loss function. If the value is 0 then it becomes normal OLS regression, on the other hand, if the value is too large it may lead to overfitting. 

![ried](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/ridge.png)

Lasso Regression where LASSO stands for Least Absolute Shrinkage and Selection Operator. This adds the "absolute value of magnitude" of coefficient as a penalty term to the loss function. 

![riedL](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/LASSO.png)

Here if the value of 0 we get back the OLS regression whereas large values, of course, lead to underfitting. 

> The key difference between these techniques is that Lasso shrinks the less important feature’s coefficient to zero thus, removing some features altogether. So, this works well for feature selection in case we have a huge number of features.

These features are a great alternative when we are dealing with a large set of features. 

References: https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c

## 📌Day 42: Implementing Ridge Regression from Scratch in Python
We have used the `YearsOfExperience_vs_salary` dataset for this.
It has 2 columns `YearsExperience` and `Salary` fo 30 employees in a company. 

Here we train a Ridge Regression model to learn the correlation between the number of years of experience and their respective salary. 

Once the model is trained, we are able to predict the salary of an employee on the basis of their years of experience. 


```python
# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
  
# Ridge Regression
class RidgeRegression() :
      
    def __init__( self, learning_rate, iterations, l2_penality ) :
          
        self.learning_rate = learning_rate 
        self.iterations = iterations        
        self.l2_penality = l2_penality # for the penalty term to be added to the loss
          
    # Function for model training            
    def fit( self, X, Y ) :
          
        # no_of_training_examples, no_of_features        
        self.m, self.n = X.shape
          
        # weight initialization        
        self.W = np.zeros( self.n )
          
        self.b = 0        
        self.X = X        
        self.Y = Y
          
        # gradient descent learning
                  
        for i in range( self.iterations ) :            
            self.update_weights()            
        return self
      
    # Helper function to update weights in gradient descent
      
    def update_weights( self ) :           
        Y_pred = self.predict( self.X )
          
        # calculate gradients      
        dW = ( - ( 2 * ( self.X.T ).dot( self.Y - Y_pred ) ) +               
               ( 2 * self.l2_penality * self.W ) ) / self.m     
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m 
          
        # update weights    
        self.W = self.W - self.learning_rate * dW    
        self.b = self.b - self.learning_rate * db        
        return self
      
    # Hypothetical function  h( x ) 
    def predict( self, X ) :    
        return X.dot( self.W ) + self.b
      
# Driver code
  
def main() :
      
    # Importing dataset    
    df = pd.read_csv( "salary_data.csv" )
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, 1].values    
  
    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, 
                                            
                                          test_size = 1 / 3, random_state = 0 )
      
    # Model training    
    model = RidgeRegression( iterations = 1000,                             
                            learning_rate = 0.01, l2_penality = 1 )
    model.fit( X_train, Y_train )
      
    # Prediction on test set
    Y_pred = model.predict( X_test )    
    print( "Predicted values ", np.round( Y_pred[:3], 2 ) )     
    print( "Real values      ", Y_test[:3] )    
    print( "Trained W        ", round( model.W[0], 2 ) )    
    print( "Trained b        ", round( model.b, 2 ) )
      
    # Visualization on test set     
    plt.scatter( X_test, Y_test, color = 'blue' )    
    plt.plot( X_test, Y_pred, color = 'orange' )    
    plt.title( 'Salary vs Experience' )    
    plt.xlabel( 'Years of Experience' )    
    plt.ylabel( 'Salary' )    
    plt.show()
      
if __name__ == "__main__" : 
    main()
```
![rL](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/ridgeee.png)


## 📌Day 43: Lasso Regression 
Lasso Regression where LASSO stands for Least Absolute Shrinkage and Selection Operator. This adds the "absolute value of magnitude" of coefficient as a penalty term to the loss function. 

![riedL](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/LASSO.png)

Lasso Regression is very similar to Ridge regression, the superficial difference between Ridge regression squares the variables and Lasso takes the absolute value.

The big difference and advantage of LASSO however is that it can exclude useless varibales from equations. This makes the equation simpler and easier to interpret. 

![lass](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/lass.png)



## 📌Day 44: Implementing Lasso Regression in Python

Dataset used in this implementation can be downloaded from the [link](https://github.com/mohit-baliyan/References.).

It has 2 columns — “YearsExperience” and “Salary” for 30 employees in a company.

We train a Lasso Regression model to learn the correlation between the number of years of experience of each employee and their respective salary. Once the model is trained, we will be able to predict the salary of an employee on the basis of his years of experience.


```python
# Importing libraries
  
import numpy as np
  
import pandas as pd
  
from sklearn.model_selection import train_test_split
  
import matplotlib.pyplot as plt
  
# Lasso Regression
  
class LassoRegression() :
      
    def __init__( self, learning_rate, iterations, l1_penality ) :
          
        self.learning_rate = learning_rate
          
        self.iterations = iterations
          
        self.l1_penality = l1_penality
          
    # Function for model training
              
    def fit( self, X, Y ) :
          
        # no_of_training_examples, no_of_features
          
        self.m, self.n = X.shape
          
        # weight initialization
          
        self.W = np.zeros( self.n )
          
        self.b = 0
          
        self.X = X
          
        self.Y = Y
          
        # gradient descent learning
                  
        for i in range( self.iterations ) :
              
            self.update_weights()
              
        return self
      
    # Helper function to update weights in gradient descent
      
    def update_weights( self ) :
             
        Y_pred = self.predict( self.X )
          
        # calculate gradients  
          
        dW = np.zeros( self.n )
          
        for j in range( self.n ) :
              
            if self.W[j] > 0 :
                  
                dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.Y - Y_pred ) ) 
                           
                         + self.l1_penality ) / self.m
          
            else :
                  
                dW[j] = ( - ( 2 * ( self.X[:, j] ).dot( self.Y - Y_pred ) ) 
                           
                         - self.l1_penality ) / self.m
  
       
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m 
          
        # update weights
      
        self.W = self.W - self.learning_rate * dW
      
        self.b = self.b - self.learning_rate * db
          
        return self
      
    # Hypothetical function  h( x ) 
      
    def predict( self, X ) :
      
        return X.dot( self.W ) + self.b
      
  
def main() :
      
    # Importing dataset
      
    df = pd.read_csv( "salary_data.csv" )
  
    X = df.iloc[:, :-1].values
  
    Y = df.iloc[:, 1].values
      
    # Splitting dataset into train and test set
  
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1 / 3, random_state = 0 )
      
    # Model training
      
    model = LassoRegression( iterations = 1000, learning_rate = 0.01, l1_penality = 500 )
  
    model.fit( X_train, Y_train )
      
    # Prediction on test set
  
    Y_pred = model.predict( X_test )
      
    print( "Predicted values ", np.round( Y_pred[:3], 2 ) ) 
      
    print( "Real values      ", Y_test[:3] )
      
    print( "Trained W        ", round( model.W[0], 2 ) )
      
    print( "Trained b        ", round( model.b, 2 ) )
      
    # Visualization on test set 
      
    plt.scatter( X_test, Y_test, color = 'blue' )
      
    plt.plot( X_test, Y_pred, color = 'orange' )
      
    plt.title( 'Salary vs Experience' )
      
    plt.xlabel( 'Years of Experience' )
      
    plt.ylabel( 'Salary' )
      
    plt.show()
      
  
if __name__ == "__main__" : 
      
    main()

```
### Output: 
```
Predicted values  [ 40600.91 123294.39  65033.07]
Real values       [ 37731 122391  57081]
Trained W         9396.99
Trained b         26505.43

```
![lwfrqass](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/lassimp.png)


## 📌Day 45: 💻 Practicing MYSQL #01
I could barely get used to SQL during the semester classes, so this week will be all about writing the most basic queries to maybe some more advanced ones by the end of the week.

I went through a few resources online and at the end of it [Mosh Hamedani's youtube tutorial](https://www.youtube.com/watch?v=7S_tz1z_5bA&t=662s) was by far the quickest and more concise ones out of all of them. So following the tutorial -- 

Today I did some basic operations on a STORE database having data on customers, order_items, products, shippers, etc. 

![regexp](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/sql.png)
![gfw1231eg](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/SQL5.png)
![gfwe2415g](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/sql8.png)
![gfweg](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/7.png)
![regexp](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/sql.png)

## 📌Day 46: 💻 Practicing MYSQL #02
Today I performed a few operations using the `ORDER BY` and `LIMIT` clauses, also started from the fundamentals of `JOIN` operations with `INNER JOINS`,JOINING ACROSS DATABASES and SELF JOINS. 

```sql
-- => MySQL Practice #02 <=

-- ## 1. ORDER BY CLAUSE ##

-- SELECT *, quantity * unit_price AS total_price
-- FROM order_items
-- WHERE order_id = 2
-- ORDER BY total_price DESC
-- # In MySQL we can sort data by any columns whether that col is in the select clause or not

-- ## 2. THE LIMIT CLAUSE ##

-- # How to limit the number of records returned from our queries? 
-- Q. Get the top 3 loyal customers
-- SELECT * 
-- FROM customers
-- ORDER BY points DESC
-- LIMIT 3
-- LIMIT 6,3	 -- Skip the first 6 records and pick 3 customers after that

-- ## 3. INNER JOINS ##
-- # Selecting columns from multiple tables
-- SELECT order_id, oi.product_id, quantity, oi.unit_price
-- FROM order_items oi
-- JOIN products p ON oi.product_id = p.product_id
-- ## 4. JOINING ACROSS DATABASES ## 
-- # How to combine columns from tables across multiple databases
-- Q. We want to join the order_items table with the products table in the sql_inventory database  
-- SELECT *
-- FROM order_items oi
-- JOIN sql_inventory.products p
-- 	ON oi.product_id = p.product_id

-- ## 5. SELF JOINS
-- # Joining a table with itself
-- Q. We write a query to join the employees table with itself so we can select the name
-- of each employee and their manager
-- USE sql_hr;
-- SELECT 
-- 	e.employee_id,
--  e.first_name,
--  m.first_name AS Manager
-- FROM employees e
-- JOIN employees m
-- 	ON e.reports_to = m.employee_id

```
References: [Mosh Hamedani's youtube tutorial](https://www.youtube.com/watch?v=7S_tz1z_5bA&t=662s)


## 📌Day 47: 👁 Introduction to BERT
BERT (Bidirectional Encoder Representations from Transformers) is a language model that was introduced by google back in 2018. 

BERT makes use of Transformer, an attention mechanism that learns contextual relations between words (or sub-words) in a text. 

As opposed to directional models, which read the text input sequentially (left-to-right or right-to-left), the Transformer encoder reads the entire sequence of words at once. Therefore it is considered bidirectional, though it would be more accurate to say that it’s non-directional. This characteristic allows the model to learn the context of a word based on all of its surroundings (left and right of the word).

BERT’s bidirectional approach (Masked LM) converges slower than left-to-right approaches (because only 15% of words are predicted in each batch) but bidirectional training still outperforms left-to-right training after a small number of pre-training steps.

[Bert Source Code](https://github.com/google-research/bert)

![bert](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/bert.png)



References: https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

## 📌Day 48: 💻 Practicing MYSQL #03
Today I learnt about: 

- JOINING MULTIPLE TABLES
- COMPOUND JOIN CONDITIONS
- OUTER JOINS
- OUTER JOINS with Multiple tables

And did a few practice exercises for the same. 

So far SQL is just what I expected, I'll be done with the entire tutorial by tomorrow and then work on more complex databases to wrap up the SQL week. 

```sql
-- => MySQL Practice #03 <=

## 1. JOINING MULTIPLE TABLES ## 
-- Joining 3 Tables 
-- SELECT 
-- 	o.order_id,
--  o.order_date,
--  c.first_name,
--  c.last_name,
--  os.name AS status  
--  FROM orders o
--  JOIN customers c 
--  	ON o.customer_id = c.customer_id
--  JOIN order_statuses os
--  	ON o.status = os.order_status_id

-- Q. PRODUCE A REPORT THAT SHOWS A PAYMENT WITH MORE DETAILS SUCH AS THE NAME OF THE CLIENT AND
-- PAYMENT METHOD
-- USE sql_invoicing;
-- SELECT p.date, p.invoice_id, p.amount, c.name, pm.name
-- FROM payments p 
-- JOIN clients c
--  ON p.client_id = c.client_id
-- JOIN payment_methods pm
--  ON p.payment_method = pm.payment_method_id
 
## 2. COMPOUND JOIN CONDITIONS ##
-- When we have more than two primary key columns it is called a composite primary key
-- USE sql_store;
-- SELECT *
-- FROM order_items oi
-- JOIN order_item_notes oin
-- 	ON oi.order_id = oin.order_id
--   AND oi.product_id = oin.product_id

## 3. IMPLICIT JOIN SYNTAX ##
-- SELECT * 
-- FROM orders o, customers c
-- WHERE o.customer_id = c.customer_id
-- # It is always better to do the explicit JOIN syntax becaue of the cross join problem 

## 4. OUTER JOINS
-- => Two kinds: LEFT JOINS and RIGHT JOINS
-- => When we use a LEFT JOIN all the records from the left table are returned 
-- => whether the given condition is true or not. Similarly for RIGHT JOIN
-- => OUTER keyword is optional 
-- SELECT 
-- 	c.customer_id,
--  c.first_name ,
--  o.order_id
-- FROM customers c
-- LEFT JOIN orders o
-- 	ON c.customer_id = o.customer_id
-- ORDER BY customer_id

## 5. OUTER JOIN MULTIPLE TABLES ##
-- # Q. Joining Orders table with the shippers -- table to display the name of shipper
-- SELECT 
-- 	c.customer_id,
-- 	c.first_name ,
-- 	o.order_id
-- FROM customers c
-- LEFT JOIN orders o
-- 	ON c.customer_id = o.customer_id
-- LEFT JOIN shippers sh
-- 	ON o.shipper_id = sh.shipper_id
-- ORDER BY c.customer_id
-- =======================================
-- SELECT 
-- 	o.order_id,
--    o.order_date ,
--    c.first_name AS customer,
--    sh.name AS shipper,
-- 		os.name AS status
-- FROM orders o
-- JOIN customers c
-- 	ON o.customer_id = c.customer_id
-- LEFT JOIN shippers sh
-- 	ON o.shipper_id = sh.shipper_id
-- JOIN order_statuses os
-- 	ON o.status = os.order_status_id
``` 
## 📌Day 49: 💻 Practicing MYSQL #04

Today I learnt about

- SELF-OUTER JOINS
- The USING clause
- NATURAL JOIN
- CROSS JOINS

```sql

-- => MYSQL PRACTICE #04 <=
##1. SELF-OUTER JOINS ##
-- USE sql_hr;
-- SELECT
-- 	e.employee_id,
--  e.first_name,
--  m.first_name AS manager
--  FROM employees e
--  LEFT JOIN employees m
-- 	ON e.reports_to = m.employee_id

##2. THE USING CLAUSE
-- # For shortening the syntax we use USING instead of the ON keyword
-- USE sql_store;
-- SELECT 
--  o.order_id,
--  c.first_name,
--  sh.name as shipper
-- FROM orders o
-- JOIN customers c
	-- #ON o.customer_id = c.customer_id
--    USING (customer_id) 
-- LEFT JOIN shippers sh
-- 	USING(shipper_id)

-- # This also works well for composite conditions
-- SELECT *
-- FROM order_items oi
-- JOIN order_item_notes oin
-- 		USING (order_id, product_id)

## 3. NATURAL JOINS ##
-- THE database engine looks at the tables and it joins them based on the common columns
-- SELECT 
-- o.order_id,
--    c.first_name
-- FROM orders o
-- NATURAL JOIN customers c
-- # These can be dangerous since we let the database engine figure out or guess the join, We do not have control over it
-- # So they can produce unexpected results. 

## 4. CROSS JOINS ## 
-- # We use them to join or combine or join every record in the first table with every record in the second table 
-- USE sql_store;
-- SELECT 
-- 	c.first_name AS customer,
--     p.name AS product
-- FROM customers c
-- CROSS JOIN products p 
-- ORDER BY c.first_name

-- # Q. DO A CROSS JOIN between shippers and products using the Implicit and the Explicit sytax
-- SELECT
-- 	sh.name AS shipper, 
--  p.name as products
-- #IMPLICIT
-- FROM shippers sh, products p
-- #EXPLICIT
-- FROM shippers sh
-- CROSS JOIN products p
-- ORDER BY sh.name
```
## 📌Day 50: 💻 Practicing MYSQL #05
Today I learnt about: 
1. UNIONS
2. Inserting Rows
3. Inserting Hierarchical Rows
4. Creating a copy of a table 

```sql
-- 1. UNIONS --
 SELECT 
	order_id,
    order_date,
    'Active' AS status
FROM orders
WHERE order_date >= '2019-01-01'
UNION
SELECT 
	order_id,
    order_date,
    'Archived' AS status
FROM orders
WHERE order_date < '2019-01-01'

# Exercise: Sort customers by BRONZE, SILVER and GOLD on the basis of the points
SELECT 
	customer_id,
    first_name, 
    points,
    'BRONZE' AS type
FROM customers
WHERE points < 2000
UNION
SELECT 
	customer_id,
    first_name, 
    points,
    'SILVER' AS type
FROM customers
WHERE points BETWEEN 2000 AND 3000
UNION
SELECT 
	customer_id,
    first_name, 
    points,
    'GOLD' AS type
FROM customers
WHERE points > 3000
ORDER BY first_name

-- 2. COLUMN ATTRIBUTES -- 
-- How to insert, update and delete data 

-- ## INSERTING A ROW ##
INSERT INTO customers (
	first_name,
    last_name,
    birth_date,
    address,
    city,
    state
    )
# Now in VALUES we supply the values for every column in customers
VALUES (
    'John', 
    'Smith', 
    '1990-01-01', 
    'address',
    'city',
    'CA'
    ) 
-- ## INSERTING MULTIPLE ROWS ## 
INSERT INTO shippers(name)
VALUES ('Shipper1'),
		('Shipper2'),
        ('Shipper3')

-- ## INSERTING HIERARCHICAL ROWS ##
INSERT INTO orders 
(customer_id, order_date, status)
VALUES ( 1, '2019-01-02', 1);
INSERT INTO order_items
VALUES(LAST_INSERT_ID(), 1,1,2.95),
	(LAST_INSERT_ID(), 2,1,2.92),
    (LAST_INSERT_ID(), 3,1,3.95)

-- ## CREATING A COPY OF A TABLE ## -- 
CREATE TABLE orders_archived AS
SELECT * FROM orders
------------------------productsorders_archived
INSERT INTO orders_archived
SELECT * 
FROM orders 
WHERE order_date < '2019-01-01'

-- EXERCISE
USE sql_invoicing;

CREATE TABLE invoices_archived AS
SELECT
	i.invoice_id,
    i.number,
    c.name AS client,
    i.invoice_total,
    i.payment_total,
    i.due_date
FROM invoices i
JOIN clients c 
	USING (client_id) 
```

## 📌Day 51: 💻 Practicing MYSQL #06
Today I learnt about: 
1. updating records 
2. using subqueries in UPDATE 
3. deleting records

```sql
-- MYSQL PRACTICE #06 -- 

# 1. Updating a Single Row 
USE sql_invoicing;
UPDATE invoices
SET payment_total = 10, payment_date = '2019-03-01'
WHERE invoice_id = 1;

# 2. Updating Multiple Rows 
UPDATE invoices
SET
	payment_total = invoice_total * 0.5,
    payment_date = due_date 
WHERE client_id IN(3,4)

-- Q. Write a SQL statement to give any customers born before 1990 50 extra points
USE sql_store;

UPDATE customers
SET points = points + 50
WHERE birth_date < '1990-01-01'

# 3. Using subqueries in Update 
USE sql_invoicing;
UPDATE invoices
SET
	payment_total = invoice_total * 0.5,
    payment_date = due_date 
WHERE client_id IN
	(SELECT client_id
    FROM clients
    WHERE state IN('CA', 'NY'))

# Q. Exercise
UPDATE orders
SET comments = 'Gold Customer'
WHERE customer_id IN
	(SELECT customer_id
	from customers
	WHERE points>3000);

# 4. DELETEING ROWS
DELETE FROM invoices 
WHERE client_id = 
(
	SELECT *
	FROM clients
	WHERE name = 'Myworks'
)
```
I'll be keeping the SQL studies on hold for now. Serious college stuff is going down need to prepare for all that. 

## 📌Day 52: ***🎵 Generating Music using LSTM in Keras #01***
Let us first look into the tools and terminologies required for the project concept. 

### 1. ***Recurrent Neural Networks***: 
Recurrent Neural Networks make use of sequential information. They perform the same operation for every single element. So the results are dependent on various computations. For this project, we will be using LSTM Neural Networks which is also a type of RNN. LSTM's are extremely useful in solving problems where the network has to remember information for a long period of time as is the case in musical information 

### 2. Music21

[music21: a Toolkit for Computer-Aided Musicology](http://web.mit.edu/music21/)

Music21 is a set of tools for helping scholars and other active listeners answer questions about music quickly and simply.

It can help us with questions like “I wonder how often Eminem does that” or “I wish I knew which band was the first to use these chords in this order” chances are the methods provided by `music21` can help us with that.

Or even questions like, "I’ll bet we’d know more about Renaissance counterpoint (or Indian ragas or post-tonal pitch structures or the form of minutes) if I could write a program to automatically write more of them"

### 3. Keras 
We'll be using the Keras library to create and train the LSTM model. Once the model is trained we will use it to generate musical notation for our music 

### Training 
For training the model we will be using Zelda and Final Fantasy soundtracks. Final Fantasy has more consistency to it and there is a large amount of those out there. 

GitHub Repository: [https://github.com/Skuldur/Classical-Piano-Composer](https://github.com/Skuldur/Classical-Piano-Composer)

## 📌Day 53: ***🎵 Generating Music using LSTM in Keras #02***
### Training 
The Data splits into Notes and Chords. Note objects contain information about the pitch, octave, and offset of the Note. 

- Octave: Refer to which set of pitches we use on a piano
- Offset: refers to where the note is located in the piece

Chord objects are essentially a container for a set of nodes that are played at the same time. 

Inorder to generate music accurately we need to be able to predict which note or chord is next. So this means our prediction array will have to contain every note and chord object that we encounter in our training set. 

Below we can see an excerpt from a midi file that has been read using Music21:
```
...
<music21.note.Note F>
<music21.chord.Chord A2 E3>
<music21.chord.Chord A2 E3>
<music21.note.Note E>
<music21.chord.Chord B-2 F3>
<music21.note.Note F>
<music21.note.Note G>
<music21.note.Note D>
<music21.chord.Chord B-2 F3>
<music21.note.Note F>
<music21.chord.Chord B-2 F3>
<music21.note.Note E>
<music21.chord.Chord B-2 F3>
<music21.note.Note D>
<music21.chord.Chord B-2 F3>
<music21.note.Note E>
<music21.chord.Chord A2 E3>
...
```
For the dataset we'll be using, the total number of different notes and chords was 352. So we will be handling this using LSTM.

Next we predict on where we want to actually put the notes. The distribution of the notes is diverse depending on the different songs where some may be really close together and others may occur after a long pause. 

So now let us read another excerpt but this time we add the offset as well to understand the interval between the different nodes and chords.

```
...
<music21.note.Note B> 72.0
<music21.chord.Chord E3 A3> 72.0
<music21.note.Note A> 72.5
<music21.chord.Chord E3 A3> 72.5
<music21.note.Note E> 73.0
<music21.chord.Chord E3 A3> 73.0
<music21.chord.Chord E3 A3> 73.5
<music21.note.Note E-> 74.0
<music21.chord.Chord F3 A3> 74.0
<music21.chord.Chord F3 A3> 74.5
<music21.chord.Chord F3 A3> 75.0
<music21.chord.Chord F3 A3> 75.5
<music21.chord.Chord E3 A3> 76.0
<music21.chord.Chord E3 A3> 76.5
<music21.chord.Chord E3 A3> 77.0
<music21.chord.Chord E3 A3> 77.5
<music21.chord.Chord F3 A3> 78.0
<music21.chord.Chord F3 A3> 78.5
<music21.chord.Chord F3 A3> 79.0
...
```
Observing the result, the most common intervals betweent he nodes and chords in `0.5`. So we can simplify our model by removing the data that don't have that said difference. This may sounds aggressive but this won't be affecting the melodies as much and should be reliable for training. 

References: https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5