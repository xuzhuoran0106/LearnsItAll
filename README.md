# LearnsItAll
This repo aims to train a deep learning model that learns lifelong, and has infinite possibility to improve itself even may overtake its creator. 
# Train lifelong learning model
Deep learning has a training phase and an inference phase. Feedback signal is provided only at training phase. So learning only happens in the training phase using algorithm like backpropagation. (For reinforcement learning, we define training phase as all parts can be updated, and inference phase as only some of parts can be updated.)

But for lifelong learning, feedback signal is provided at both training phase and inference phase. So learning can happen in both training phase and inference phase. Algorithms like backpropagation are used in training phase to learn both knowledge about specific task and some learning skill. The learned learning skill is used in the inference phase to update itself according to feedback signal. Not only task specific knowledge, but even learning skill can be improved in the inference phase .

Such a lifelong learning problem can be defined as below:
>while not end: 
&emsp;Take a sample (I,GT)  
&emsp;for i in [0,...,n]:  
&emsp;&emsp;$o_{i}=Model(I,None)$
&emsp;&emsp;$Model(None,GT)$
&emsp;&emsp;$Loss_{i}=loss(o_{i},GT)$
&emsp;&emsp;Using backpropagation to maximize: $Loss_{n}-Loss_{0}$

# A model with infinite talent
Considering a model has infinite talent to learn everything, three fundamental requirements should be meet:
1. The model can read every parts of its knowledge.
2. The model can convert computing result as new knowledge.
3. The model can modify every parts of its knowledge. 

Such a model can be defined as below:  


The model is a time series model. $I_{t}$ is the input at time step t. $GT_{t}$ is the feedback at time step t.  

>Compute Step:  
$o_{t}^{1}=f_{1}(I_{t},GT_{t},h_{t-1};w_{t-1}^1)$
$...$
$o_{t}^{n}=f_{n}(I_{t},GT_{t},h_{t-1};w_{t-1}^n)$

>Weights Update:  
$a_{t}^{i,0},...,a_{t}^{i,n}=S_{w}(o_{t}^{i};o_{t}^{1},...,o_{t}^{n})$
$w_{t}^{1}=a_{t}^{1,0}w_{t-1}^{1}+...+a_{t}^{1,n}w_{t-1}^{n}$
$...$
$w_{t}^{n}=a_{t}^{n,0}w_{t-1}^{n}+...+a_{t}^{n,n}w_{t-1}^{n}$
, where $S_{w}$ is a non-parameter function that satisfies $a_{t}^{i,0}+...+a_{t}^{i,n}=1$

>Hidden States Update:  
$a_{t}^{1_{o}},a_{t}^{1_{w}},...,a_{t}^{n_{o}},a_{t}^{n_{w}}=S_{h}(o_{t}^{1},...,o_{t}^{n})$
$h_{t}=a_{t}^{1_{o}}o_{t}^{1}+a_{t}^{1_{w}}w_{t}^{1}+...+a_{t}^{n_{o}}o_{t}^{n}+a_{t}^{n_{w}}w_{t}^{n}$
, where $S_{h}$ is a non-parameter function that satisfies $a_{t}^{1_{o}}+a_{t}^{1_{w}}+...+a_{t}^{n_{o}}+a_{t}^{n_{w}}=1$  

We find this model is equivalent to a lot of existing models.
First we notice that Eq.1 is equivalent to Eq.2.  

>$y_{0}=w_{0}x_{0}$
$w_{1}=w_{0}+y_{0}$
$y_{1}=w_{1}x_{1}$
Eq.1

>$y_{1}=(w_{0}+w_{0}x_{0})x_{1}$
Eq.2


Next we notice that Eq.3 is equivalent to Eq.2 and compatible with deep learning frameworks. Hence, we can use hidden state to replace the weights update. We also notice that the hidden state naturally satisfies the three requirements mentioned above.  

>$h_{s}=0$
$h_{0}=(w_{0}+h_{s})x_{0}=w_{0}x_{0}$
$h_{1}=(w_{0}+h_{0})x_{1}=(w_{0}+(w_{0}+h_{s})x_{0})x_{1}=(w_{0}+w_{0}x_{0})x_{1}$
Eq.3

Finally Eq.3 can be rewrote as Eq.4. It shows the computation of $hx$ is independent to $wx$. The form of Eq.4 is similar to a lot of existing models, such as LSTM.  

>$h_{s}=0$
$h_{0}=(w_{0}+h_{s})x_{0}=w_{0}x_{0}+h_{s}x_{0}$
$h_{1}=(w_{0}+h_{0})x_{1}=w_{0}x_{1}+h_{0}x_{1}$
Eq.4

So a simple LSTM has infinite talent to learn everything. The weights fixed at training phase of LSTM can be considered as instinct. The hidden states at inference phase can be considered as both task specific knowledge and learning skills.

## A Trick
Consider a fully-connected layer: 
>$y=w_{t-1}\times x + b_{t}$, where the shape of $w_{t-1}$ is $N\times M$, $x$ is $1\times N$.

And a function $f$ can give $dw$ at next time step: 
>$dw_{t}=f(x,y,GT_{t},w_{t-1})$

The input dimension of $f$ is $N+N+N+NM$, and the output dimension of $f$ is $NM$. The input and output dimensions are very large which makes the $f$ is hard to learn. And $f$ must learn gradient descent.

Gradient descent is a very common component to a lot of machine learning models. We hope our model is easy to train and mostly focus on high level knowledge. Hence, we eliminate th learning of gradient descent in our model and use a fixed gradient descent as an internal component. The new model is modified as:
>$Y=f(x,y,GT_{t})$
$dw_{t}=GradientDescent(L_{2}(Y,y))$
Introduce gradient descent into the model blurs the edge of training and inference. Gradient descent at training phase can be considered as learns a high-level learning skill which guide the model modification at inference phase. But the real modification is done by inference phase gradient descent.

# What do we want to let the model learn
As we use gradient descent is our model, we need to answer a question: Does inference phase gradient descent limit the possibility of model to improve itself? Meanwhile, there are a lot of ways to create a neural network. Does using a specific way to create our model limit the possibility of model to improve itself? We think the answer is NO. Just as a computer can be made by vacuum tubes, transistors, or integrated circuits, the low-level mechanism does not affect the learning of high-level knowledge.

We want the model can learn high-level learning skill at inference phase, that is to accumulate old experience and use them to guide new learning process.
Lets use DOTA2 as an example to explain this.
1. The ability of playing Hero "alpha" good is TypeA. The common points of playing many Heroes good is TypeA'.
2. When a new Hero 'beta' is given, TypeA' can be directly used.
3. The ability to learn TypeA of Hero 'beta' quickly is TypeB. TypeB is a kind of high-level learning skill. It can be learned by the model at inference phase. TypeB generates a number of TypeA, and adjust TypeA according to feedback. The common points of playing many games good is TypeB'.
4. When a new game StartCraft2 is given, TypeB' can be directly used. Furthermore, there must be a way to learn TypeB of StartCraft2. The ability to learn TypeB of StartCraft2 quickly is TypeC. TypeC is another kind of high-level learning skill. It can be learned by the model at inference phase.
TypeC generates a number of TypeB for StartCraft2; TypeB generates a number of TypeA for StartCraft2; Feedback of TypeA are collected; TypeC adjust TypeB according to feedback.
5. We find that we need TypeD to learn TypeC, TypeE to learn TypeD, and so on. To end this loop, there must be a TypeX that can learn everything. Enumerating all possibility is TypeX. All types can be updated or newly added at inference phase. The training phase can provide a TypeX' which drives the initial learning at inference phase. TypeX and TypeX' may become disabled by newly learned parts.

## How does the model work at inference phase (ideally)
1. The model have the ability to distinguish different tasks.
2. There are a lot of different TypeA, TypeA', TypeB, TypeB', .etc in the model. The model knows when to use which one.
3. The model knows when to learn new TypeA, TypeA', TypeB, TypeB', .etc. Information that needed by learning is collected by model itself.
4. Learning new TypeA, TypeA', TypeB, TypeB', .etc do not affect other parts of the model.
5. The model decide which neuron stores which knowledge by itself.
6. Even if input or GT is not provided, the model can update itself by revising old experience.
7. TypeX is a internal mechanism. But the parameters for calling TypeX is prepared by the model.
8. ...

This repo holds belief that all those features do not need to be manually designed by human beings. LSTM has potential ability to learn all those features, and training phase backpropagation can actually learn them, in addition, TypeX' is also learned by training phase backpropagation.

## How does the model work at inference phase (simplified)
To simplify the problem, we only consider TypeA-C and do not use TypeX. TypeA and TypeB can be changed at inference phase. TypeC is fixed at inference phase. Three types in the model are assigened manually, in stead of letting algorithm to automatically determine them.

# The experiments
We use a modified MNIST experiment to evaluate our idea. We encode the ground truth in a way that the decoding must be done by trial and error. The feedback can tell the correctness of decoded class.
For a specific encoding method, TypeA is a MLP that maps image to class. TypeB is a RNN that can decode the class by minimum trial and error. TypeC is another RNN that can generate TypeB RNN. 


The ground truth for MNIST is 0-9. We split number 0-9 to N groups. We only give the group index to the model. The model must guess the real groud truth and finish the training. The model can know the correctness of the guess from feedback.
For example, we have group [0-3],[4-6],[7-9]. We take a number 5, and give group index 2 to the model. The model may guess 4 at the first time, and receive a 'wrong' feedback. Then the model can guess 5 and receive a 'right' feedback.

The TypeA network is a MLP. It takes image as input, and output the class of the image (y). The feedback is calculated using 'y' and the real groud truth (0-9).
It has two kinds of weights: $w_{A}$ and $h_{A}$. $w_{A}$ is fixed at traning phase; $h_{A}$ can be updated at inference phase. 
Let $o_{A}=h_{A}x_{A}$. The target of $o_{A}$ is given by TypeB. So $h_{A}$ can be updated by backpropagation.

The TypeB network is a LSTM. It must know two things:
1. the mapping from group index to numbers. This is encoded as $h_{B}$ which is given by TypeC.
2. the trail and error process, and how to generate $h_{A}$. This is encoded as $w_{B}$.
So its input sequence is [group, feedback, ..., feedback]. Its output sequence is [$target_{A}$, ..., $target_{A}$].

The TypeC network is also a LSTM. It must know to guess the mapping from group index to numbers by trail and error process.
So its input sequence is [[group,target_{A},feedback,...,target_{A},feedback], ..., [group,target_{A},feedback,...,target_{A},feedback]]. Its output sequence is [$target_{B}$, ..., $target_{A}$].

$$
y=MLP(image;w_{A},h_{A})\\
target_{A}=LSTM_{B}(group,feedback;w_{B},h_{B})\\
target_{B}=LSTM_{C}([group,target_{A},feedback,...,target_{A},feedback];w_{C})
$$

When the mapping is fixed, we can use TypeB to train new image (update TypeA). 
```
while true:
  take a sample (image, GT)
  map GT to GroupID

  y=MLP(image;w_{A},h_{A})
  feedback_{0}=loss(y,GT)
  target_{A}=LSTM_{B}(group,feedback_{0};w_{B},h_{B})
  update h_{A} using target_{A}

  ...

  y=MLP(image;w_{A},h_{A})
  feedback_{i}=loss(y,GT)
  target_{A}=LSTM_{B}(group,feedback_{i};w_{B},h_{B})
  update h_{A} using target_{A}

  maximize: feedback_{0}-feedback_{i} // If we want to train TypeB directly

```

When a new mapping is given, we need to run TypeC to update TypeB. If a group is shared by all mappings, we can learn a TypeB'. 
```
while true:
  take samples for every class (image_{0}, GT_{0}, ..., image_{9}, GT_{9})
  map GT to GroupID
  for step in [0:S]:
    for i in [0:9]:
      image = image_{i}
      GT = GT_{i}

      y=MLP(image;w_{A},h_{A})
      feedback_{0}=loss(y,GT)
      target_{A}=LSTM_{B}(none,feedback_{0};w_{B},h_{B})
      update h_{A} using target_{A}

      ...

      y=MLP(image;w_{A},h_{A})
      feedback_{i}=loss(y,GT)
      target_{A}=LSTM_{B}(none,feedback_{i};w_{B},h_{B})
      update h_{A} using target_{A}
  
    target_{B}=LSTM_{C}([
      [group,target_{A},feedback_{0},...,target_{A},feedback_{i}],
      ...,
      [group,target_{A},feedback_{0},...,target_{A},feedback_{i}]
                        ];w_{C})
    update h_{B}
    step_loss_{step}=sum(feedback_{0}+...+feedback_{i})
  maximize:step_loss_{0}-step_loss_{S}
```

# The ideal model and TypeX
The above model has many manually designed parts. The "for loop" is fixed; the calling sequence of LSTM is fixed; the input of LSTM is prepared manually; the update is fixed. In a ideal model, all of those procedure should be learned by model itself.

We also notice that the trial and error happens multiple times in the above model. The "for loop", "calling sequence", and "update" form the trial and error procedure. This procedure can be shared across many tasks. If we consider trial and error is a inductive bias, then this procedure is a TypeX. But not all procedure is TypeX. The concept of TypeX is relative to inductive bias. 

## The ideal model (Only consider TypeA and TypeB)
Assuming we has three sub-models: $f_{A}, f_{B}, f_{aux}$. They receives same input and hidden states(h), and calculate their outputs independently. A non-parameter function selects the final hidden state from the 3 outputs. Another non-parameter functions decides the update of a specific sub-model. The training phase must to let the three sub-models learn to cooperate with each other to form the trial and error procedure.

input:$img_{1}$
h:none
$y_{1}=f_{A}(img_{1}, none)-->h$

input:GT
h:$y_{1}$
$cls_{1}=f_{B}(GT, y_{1})-->h$          $f_{B}$ memory: $GT, cls_{1}$

input:$feedback_{1}$
h:$cls_{1}$
$cls_{2}=f_{B}(feedback_{1}, cls_{1})-->h$          $f_{B}$ memory: $GT, cls_{1}, feedback_{1}, cls_{2}$

input:$feedback_{2}$
h:$cls_{2}$
$img_{1}=f_{aux}(feedback_{2}, cls_{2})-->h$          $f_{B}$ memory: $img_{1}, y_{1}, cls_{2}, feedback_{2}$
update $f_{A}$ with $img_{1}, y_{1}, cls_{2}$

input:none
h:$img_{1}$
$y_{1}^{'}=f_{A}(none, img_{1})$

input:$img_{2}$
h:none
$y_{2}=f_{A}(img_{2}, none)-->h$


# Conclusion
1. We need a fully differentiable model that can generate "interal program" and has a "interal backpropagation step" to update itself.
2. Only update hidden states is enough for infinite possibility.
3. A training process that focuses on learning ability.