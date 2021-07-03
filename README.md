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

<<<<<<< HEAD
## 📌Day 19: 📊***Building Interactive Dashboards with Tableau #01***
### 📊Building Interactive Dashboard for EDA with Sample Superstore dataset Day-01
Tableau is amazing. Ngl I've never had this much fun with data before xD, made some interesting inferences with the SampleSuperstore dataset and I am mostly done, other than of course, the organising as you can see the clutter. I might need to remove some irrelevant fields that do not contribute to the achievements at hand.
=======
## 📌Day 19: 📊***Building Interactive Dashboards with Tableau***
### 📊Building Interactive Dashboard for EDA with Sample Superstore dataset
Tableau is amazing. Ngl I've never had this much fun with data before xD, made some interesting inferences with the SampleSuperstore dataset, and I am mostly done, other than of course, organizing and applying some better practices for the sheets. I can say I am at least familiar now with the interface to a good extent. I will put some more work into the dashboard before making the video demonstration for the project. 
>>>>>>> 92c96745f49f409c012bff9b5fb578c72bec4346

In this project, my task is to find out what factors the losses are being incurred for the company and what we can do to reduce those losses. After some EDA with Tableau, I was able to determine that the losses are the highest and most significant in the following cities: 

Philadelphia, Houston, Chicago.

And the profit is the highest in cities like:

New York, Seattle, San Francisco, Los Angeles.

The losses are highest with the Category of furniture due to a disproportion between sales and discounts in the cities with losses. 

There is a lot more to uncover which I will be explaining in the video demonstration.

![project1](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/project1.png)

## 📌Day 20: 📊***Building Interactive Dashboards with Tableau #02***
### 📊Building Interactive Dashboard for EDA with Sample Superstore dataset Day-02
Today I completed building and organising the dashboard for the SampleSuperstore dataset. 
In the displayed worksheet I have taken the size of the circles as the amount of sales in any particular category. BLUE colour is on the positive side of profit whereas RED is on the negative side of it. 

### Overview of the Dashboard

#### Profit/Sales by City
For this category I have designated the size of the circles as indicators for the number of sales in any particular city. From that knowledge we can deduce that: 

New York, Seattle, San Francisco, Los Angeles, Houston and Philadelphia have the most considerable amount of sales. 

Also considering the colours of the circles we can deduct the following: 

New York, Seattle, San Francisco, Los Angeles are making us an immense amount of profit with New York being the most profitable with the darkest shade of blue. 

Philadelphia, Houston and Chicago are in shades of red and are in fact experiencing negative profit ,i.e , losses. We need to understand the factors affecting these regions and implement the practices from the blue regions onto these regions. 

#### Profit/Sales by Ship Mode
Ship Mode is the means by which our goods are shipped to the customer. A shipping mode is a combination of a shipping carrier and the shipping service that is offered by that carrier. 

In the overall circle chart we can see that Standard Mode of shipping has both the higher number of sales and the higher number of profits. Whereas First class, Second Class and Same Day are going at a loss. The company needs to focus on increasing the cost for the First Class, Second Class and Same Day shipping modes to decrease the loses to some extent. 

#### Profit/Sales by Region
We see East and West having both the highest number of sales and profit whereas Central and South are going at an immense loss with Central incurring the heaviest losses. We thus need to understand the practices in the West region and apply them to the Central and South regions.

#### Profit/Sales by Segment 
We see Consumers having a large amount of profit wherease Corporate and Home Office going at a loss. This can contribute to a number of factors like heavier discount because of bulk buying by the Corporate and Home Office segments.

#### Profit/Sales by Category
Here Sales are in YELLOW and Profit is same as before (-RED->0->+BLUE)
Here we go to the root of it all. We can see while all three of the categories have a desirable amount of sales the losses in the furniture category is extremely concerning. Inorder to understand this we go through our Sub-categories and analyse them by Profit, Sales and Discount on the products. 

#### Profit/Sales/Discount by Subcategories 
Here we have Profit as (-RED->0->+BLUE) Sales as (YELLOW) and Discount as (TEAL BLUE)

Looking into the Furnitures sub-categories we can clearly see that the tables and bookcases are running at losses. This is probably due to a disproportionate amount of discount on the said sub-categories as we can see from the graphs. 

On the other hand we have Office Supplies running at a decent profit with some loss for Supplies.

Our Technology sub-categories as we can see are doing really well however Machines are running at a high discount rate which may be reduced to increase the profit. 

I have more to discuss here, I will do that in the video. 


![dashboard](https://github.com/vxhl/365Days_MachineLearning_DeepLearning/blob/main/Images/dashboard.png)
