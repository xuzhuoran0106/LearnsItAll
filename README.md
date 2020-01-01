# LearnsItAll
This repo aims to train a deep learning model that learns lifelong, and has infinite possibility to improve itself even may overtake its creator. 
# Train lifelong learning model
Deep learning has a training phase and an inference phase. Feedback signal is provided only at training phase. So learning only happens in the training phase using algorithm like backpropagation. 

But for lifelong learning, feedback signal is provided at both training phase and inference phase. So learning can happen in both training phase and inference phase. Algorithms like backpropagation are used in training phase to learn both knowledge about specific task and some learning skill. The learned learning skill is used in the inference phase to update itself according to feedback signal. Not only task specific knowledge, but even learning skill can be improved in the inference phase .

Such a lifelong learning problem can be defined as below:

>while not end: 
&ensp;&ensp;Take a sample (I,GT)  
&ensp;&ensp;for i in [0,...,n]:  
&ensp;&ensp;&ensp;&ensp;
<a href="https://www.codecogs.com/eqnedit.php?latex=o_{i}=LSTM(I,None)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?o_{i}=LSTM(I,None)" title="o_{i}=LSTM(I,None)" /></a>  
&ensp;&ensp;&ensp;&ensp;
<a href="https://www.codecogs.com/eqnedit.php?latex=LSTM(None,GT)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?LSTM(None,GT)" title="LSTM(None,GT)" /></a>  
&ensp;&ensp;&ensp;&ensp;
<a href="https://www.codecogs.com/eqnedit.php?latex=Loss_{i}=loss(o_{i},GT)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Loss_{i}=loss(o_{i},GT)" title="Loss_{i}=loss(o_{i},GT)" /></a>  
&ensp;&ensp;&ensp;&ensp;Using backpropagation to maximize: 
<a href="https://www.codecogs.com/eqnedit.php?latex=Loss_{n}-Loss_{0}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Loss_{n}-Loss_{0}" title="Loss_{n}-Loss_{0}" /></a>  

We use LSTM here because it has potential ability to learn everything. We will explain the reason later.

# A model with infinite talent
Considering a model has infinite talent to learn everything, three fundamental requirements should be meet:
1. The model can read every parts of its knowledge.
2. The model can convert computing result as new knowledge.
3. The model can modify every parts of its knowledge. 

Such a model can be defined as below:  
>The model is a time series model. 𝐼_𝑡 is the input at time step 𝑡. 𝐺𝑇_𝑡 is the feedback at time step 𝑡.  

>Compute Step:  
<a href="https://www.codecogs.com/eqnedit.php?latex=\\o_{t}^{1}=f_{1}(I_{t},GT_{t},h_{t-1};w_{t-1}^1)&space;\\&space;...&space;\\&space;o_{t}^{n}=f_{n}(I_{t},GT_{t},h_{t-1};w_{t-1}^n)&space;\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\o_{t}^{1}=f_{1}(I_{t},GT_{t},h_{t-1};w_{t-1}^1)&space;\\&space;...&space;\\&space;o_{t}^{n}=f_{n}(I_{t},GT_{t},h_{t-1};w_{t-1}^n)&space;\\" title="\\o_{t}^{1}=f_{1}(I_{t},GT_{t},h_{t-1};w_{t-1}^1) \\ ... \\ o_{t}^{n}=f_{n}(I_{t},GT_{t},h_{t-1};w_{t-1}^n) \\" /></a>  

>Weights Update:  
<a href="https://www.codecogs.com/eqnedit.php?latex=\\a_{t}^{i,0},...,a_{t}^{i,n}=S_{w}(o_{t}^{i};o_{t}^{1},...,o_{t}^{n})\\&space;w_{t}^{1}=a_{t}^{1,0}w_{t-1}^{1}&plus;...&plus;a_{t}^{1,n}w_{t-1}^{n}\\&space;...\\&space;w_{t}^{n}=a_{t}^{n,0}w_{t-1}^{n}&plus;...&plus;a_{t}^{n,n}w_{t-1}^{n}\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\a_{t}^{i,0},...,a_{t}^{i,n}=S_{w}(o_{t}^{i};o_{t}^{1},...,o_{t}^{n})\\&space;w_{t}^{1}=a_{t}^{1,0}w_{t-1}^{1}&plus;...&plus;a_{t}^{1,n}w_{t-1}^{n}\\&space;...\\&space;w_{t}^{n}=a_{t}^{n,0}w_{t-1}^{n}&plus;...&plus;a_{t}^{n,n}w_{t-1}^{n}\\" title="\\a_{t}^{i,0},...,a_{t}^{i,n}=S_{w}(o_{t}^{i};o_{t}^{1},...,o_{t}^{n})\\ w_{t}^{1}=a_{t}^{1,0}w_{t-1}^{1}+...+a_{t}^{1,n}w_{t-1}^{n}\\ ...\\ w_{t}^{n}=a_{t}^{n,0}w_{t-1}^{n}+...+a_{t}^{n,n}w_{t-1}^{n}\\" /></a>  
, where <a href="https://www.codecogs.com/eqnedit.php?latex=S_{w}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S_{w}" title="S_{w}" /></a> is a non-parameter function that satisfies <a href="https://www.codecogs.com/eqnedit.php?latex=a_{t}^{i,0}&plus;...&plus;a_{t}^{i,n}=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_{t}^{i,0}&plus;...&plus;a_{t}^{i,n}=1" title="a_{t}^{i,0}+...+a_{t}^{i,n}=1" /></a>  

Hidden States Update:  
<a href="https://www.codecogs.com/eqnedit.php?latex=\\a_{t}^{1_{o}},a_{t}^{1_{w}},...,a_{t}^{n_{o}},a_{t}^{n_{w}}=S_{h}(o_{t}^{1},...,o_{t}^{n})\\&space;h_{t}=a_{t}^{1_{o}}o_{t}^{1}&plus;a_{t}^{1_{w}}w_{t}^{1}&plus;...&plus;a_{t}^{n_{o}}o_{t}^{n}&plus;a_{t}^{n_{w}}w_{t}^{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\a_{t}^{1_{o}},a_{t}^{1_{w}},...,a_{t}^{n_{o}},a_{t}^{n_{w}}=S_{h}(o_{t}^{1},...,o_{t}^{n})\\&space;h_{t}=a_{t}^{1_{o}}o_{t}^{1}&plus;a_{t}^{1_{w}}w_{t}^{1}&plus;...&plus;a_{t}^{n_{o}}o_{t}^{n}&plus;a_{t}^{n_{w}}w_{t}^{n}" title="\\a_{t}^{1_{o}},a_{t}^{1_{w}},...,a_{t}^{n_{o}},a_{t}^{n_{w}}=S_{h}(o_{t}^{1},...,o_{t}^{n})\\ h_{t}=a_{t}^{1_{o}}o_{t}^{1}+a_{t}^{1_{w}}w_{t}^{1}+...+a_{t}^{n_{o}}o_{t}^{n}+a_{t}^{n_{w}}w_{t}^{n}" /></a>  
, where <a href="https://www.codecogs.com/eqnedit.php?latex=S_{h}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S_{h}" title="S_{h}" /></a> is a non-parameter function that satisfies <a href="https://www.codecogs.com/eqnedit.php?latex=a_{t}^{1_{o}}&plus;a_{t}^{1_{w}}&plus;...&plus;a_{t}^{n_{o}}&plus;a_{t}^{n_{w}}=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_{t}^{1_{o}}&plus;a_{t}^{1_{w}}&plus;...&plus;a_{t}^{n_{o}}&plus;a_{t}^{n_{w}}=1" title="a_{t}^{1_{o}}+a_{t}^{1_{w}}+...+a_{t}^{n_{o}}+a_{t}^{n_{w}}=1" /></a>


We find this model is equivalent to a LSTM model.
First we notice that Eq.1 is equivalent to Eq.2.   

<a href="https://www.codecogs.com/eqnedit.php?latex=\\y_{0}=w_{0}x_{0}\\&space;w_{1}=w_{0}&plus;y_{0}\\&space;y_{1}=w_{1}x_{1}\\&space;Eq.1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\y_{0}=w_{0}x_{0}\\&space;w_{1}=w_{0}&plus;y_{0}\\&space;y_{1}=w_{1}x_{1}\\&space;Eq.1" title="\\y_{0}=w_{0}x_{0}\\ w_{1}=w_{0}+y_{0}\\ y_{1}=w_{1}x_{1}\\ Eq.1" /></a>  

<a href="https://www.codecogs.com/eqnedit.php?latex=\\y_{1}=(w_{0}&plus;w_{0}x_{0})x_{1}\\&space;Eq.2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\y_{1}=(w_{0}&plus;w_{0}x_{0})x_{1}\\&space;Eq.2" title="\\y_{1}=(w_{0}+w_{0}x_{0})x_{1}\\ Eq.2" /></a>  

Next we notice that Eq.3 is equivalent to Eq.2 and compatible with deep learning frameworks. Hence, we can use hidden state to replace the weights update. We also notice that the hidden state naturally satisfies the three requirements mentioned above.  

<a href="https://www.codecogs.com/eqnedit.php?latex=\\h_{s}=0\\&space;h_{0}=(w_{0}&plus;h_{s})x_{0}=w_{0}x_{0}\\&space;h_{1}=(w_{0}&plus;h_{0})x_{1}=(w_{0}&plus;(w_{0}&plus;h_{s})x_{0})x_{1}=(w_{0}&plus;w_{0}x_{0})x_{1}\\&space;Eq.3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\h_{s}=0\\&space;h_{0}=(w_{0}&plus;h_{s})x_{0}=w_{0}x_{0}\\&space;h_{1}=(w_{0}&plus;h_{0})x_{1}=(w_{0}&plus;(w_{0}&plus;h_{s})x_{0})x_{1}=(w_{0}&plus;w_{0}x_{0})x_{1}\\&space;Eq.3" title="\\h_{s}=0\\ h_{0}=(w_{0}+h_{s})x_{0}=w_{0}x_{0}\\ h_{1}=(w_{0}+h_{0})x_{1}=(w_{0}+(w_{0}+h_{s})x_{0})x_{1}=(w_{0}+w_{0}x_{0})x_{1}\\ Eq.3" /></a>  

Finally Eq.3 can be rewrote as Eq.4. It shows the computation of ℎ𝑥 is independent to 𝑤𝑥. The form of Eq.4 is similar to a lot of existing models, such as LSTM.   

<a href="https://www.codecogs.com/eqnedit.php?latex=\\h_{s}=0\\&space;h_{0}=(w_{0}&plus;h_{s})x_{0}=w_{0}x_{0}&plus;h_{s}x_{0}\\&space;h_{1}=(w_{0}&plus;h_{0})x_{1}=w_{0}x_{1}&plus;h_{0}x_{1}\\&space;Eq.4" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\h_{s}=0\\&space;h_{0}=(w_{0}&plus;h_{s})x_{0}=w_{0}x_{0}&plus;h_{s}x_{0}\\&space;h_{1}=(w_{0}&plus;h_{0})x_{1}=w_{0}x_{1}&plus;h_{0}x_{1}\\&space;Eq.4" title="\\h_{s}=0\\ h_{0}=(w_{0}+h_{s})x_{0}=w_{0}x_{0}+h_{s}x_{0}\\ h_{1}=(w_{0}+h_{0})x_{1}=w_{0}x_{1}+h_{0}x_{1}\\ Eq.4" /></a>  

So a simple LSTM has infinite talent to learn everything. The weights fixed at training phase of LSTM can be considered as instinct. The hidden states at inference phase can be considered as both task specific knowledge and learning skills.


