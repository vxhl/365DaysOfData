# 365 Days Of Machine Learning and Deep Learning ⚒
⌚ Here I will be documenting my journey from ▶14 June 2021 to 14 June 2022🔚 

### 🏆 Day 01 : 
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

### 📃 Day 02 :
***Preprocessing phase for Twitter Depression Detection Project***
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

