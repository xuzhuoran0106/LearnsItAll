# LearnsItAll
This repo aims to train a deep learning model that learns lifelong, and has infinite possibility to improve itself even may overtake its creator. 
# Train lifelong learning model
Deep learning has a training phase and an inference phase. Feedback signal is provided only at training phase. So learning only happens in the training phase using algorithm like backpropagation. 

But for lifelong learning, feedback signal is provided at both training phase and inference phase. So learning can happen in both training phase and inference phase. Algorithms like backpropagation are used in training phase to learn both knowledge about specific task and some learning skill. The learned learning skill is used in the inference phase to update itself according to feedback signal. Not only task specific knowledge, but even learning skill can be improved in the inference phase .

Such a lifelong learning problem can be defined as below:
```
while not end:
  Take a sample (𝐼,𝐺𝑇)
  for i in [0,...,n]:
    𝑂_𝑖=𝐿𝑆𝑇𝑀(𝐼,𝑁𝑜𝑛𝑒)
    𝐿𝑆𝑇𝑀(𝑁𝑜𝑛𝑒,𝐺𝑇)
    𝐿𝑜𝑠𝑠_𝑖=𝑙𝑜𝑠𝑠(𝑂_𝑖,𝐺𝑇)
Using BP to maximize: 𝐿𝑜𝑠𝑠_𝑛−𝐿𝑜𝑠𝑠_0
```
We use LSTM here because it has potential ability to learn everything. We will explain the reason later.

# A model with infinite talent
Considering a model has infinite talent to learn everything, three fundamental requirements should be meet:
1. The model can read every parts of its knowledge.
2. The model can convert computing result as new knowledge.
3. The model can modify every parts of its knowledge. 

Such a model can be defined as below:
```
The model is a time series model. 𝐼_𝑡 is the input at time step 𝑡. 𝐺𝑇_𝑡 is the feedback at time step 𝑡.

Compute Step:
𝑜_𝑡^1=𝑓_1 (𝐼_𝑡,𝐺𝑇_𝑡,ℎ_(𝑡−1);𝑤_(𝑡−1)^1)
𝑜_𝑡^2=𝑓_2 (𝐼_𝑡,𝐺𝑇_𝑡,ℎ_(𝑡−1);𝑤_(𝑡−1)^2)
…
𝑜_𝑡^𝑛=𝑓_𝑛 (𝐼_𝑡,𝐺𝑇_𝑡,ℎ_(𝑡−1);𝑤_(𝑡−1)^𝑛)


Weights Update:
𝑎_𝑡^(𝑖,0),𝑎_𝑡^(𝑖,1),…,𝑎_𝑡^(𝑖,𝑛)= 𝑆_𝑤(𝑜_𝑡^𝑖;𝑜_𝑡^1,…,𝑜_𝑡^𝑛);      𝑆_𝑤 is a non-parameter function that satisfies ∑𝑎_𝑡^(𝑖,0),𝑎_𝑡^(𝑖,1),…,𝑎_𝑡^(𝑖,𝑛)=1
𝑤_𝑡^1=𝑎_𝑡^1,0 𝑤_𝑡^1+𝑎_𝑡^1,1 𝑜_𝑡^1+…𝑎_𝑡^(1,𝑛)𝑜_𝑡^𝑛
…
𝑤_𝑡^𝑛=𝑎_𝑡^(𝑛,0) 𝑤_𝑡^𝑛+𝑎_𝑡^(𝑛,1) 𝑜_𝑡^1+…𝑎_𝑡^(𝑛,𝑛) 𝑜_𝑡^𝑛

Hidden States Update:
𝑎_𝑡^(1_𝑜),𝑎_𝑡^(1_𝑤)…,𝑎_𝑡^(𝑛_𝑜),𝑎_𝑡^(𝑛_𝑤)=𝑆_ℎ(𝑜_𝑡^1,…,𝑜_𝑡^𝑛);   𝑆_ℎ is a non-parameter function that satisfies ∑𝑎_𝑡^(1_𝑜),𝑎_𝑡^(1_𝑤),…,𝑎_𝑡^(𝑛_𝑜),𝑎_𝑡^(𝑛_𝑤)=1
ℎ_𝑡=𝑎_𝑡^(1_𝑜)𝑜_𝑡^1+𝑎_𝑡^(1_𝑤)𝑤_𝑡^1+…+𝑎_𝑡^(𝑛_𝑜)𝑜_𝑡^𝑛+𝑎_𝑡^(𝑛_𝑤)𝑤_𝑡^𝑛
```

We find this model is equivalent to a LSTM model.
First we notice that Eq.1 is equivalent to Eq.2. 
```
y_0=𝑤_0𝑥_0
𝑤_1=𝑤_0+𝑦_0
𝑦_1=𝑤_1𝑥_1
Eq.1 
```
```
𝑦_1=(𝑤_0+𝑤_0𝑥_0)𝑥_1
Eq.2
```
Next we notice that Eq.3 is equivalent to Eq.2 and compatible with deep learning frameworks. Hence, we can use hidden state to replace the weights update. We also notice that the hidden state naturally satisfies the three requirements mentioned above.  
```
ℎ_𝑠=0 
h_0=(𝑤_0+ℎ_𝑠)𝑥_0=(𝑤_0+ℎ_𝑠)𝑥_0=𝑤_0𝑥_0
ℎ_1=(𝑤_0+ℎ_0)𝑥_1= (𝑤_0+(𝑤_0+ℎ_𝑠)𝑥_0)𝑥_1=(𝑤_0+𝑤_0𝑥_0)𝑥_1
Eq.3
```
Finally Eq.3 can be rewrote as Eq.4. It shows the computation of ℎ𝑥 is independent to 𝑤𝑥. The form of Eq.4 is similar to a lot of existing models, such as LSTM. 
```
ℎ_𝑠=0 
h_0=(𝑤_0+ℎ_𝑠)𝑥_0=𝑤_0𝑥_0+ℎ_𝑠𝑥_0
ℎ_1=(𝑤_0+ℎ_0)𝑥_1=𝑤_0𝑥_1+h_0𝑥_1
Eq.4
```
So a simple LSTM has infinite talent to learn everything. The weights fixed at training phase of LSTM can be considered as instinct. The hidden states at inference phase can be considered as both task specific knowledge and learning skills.


