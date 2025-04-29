#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math


X = [
    [1,0,0,0],  #التارجت 4 كلمات 
    [0,1,0,0],  
    [0,0,1,0]   
]
y_true = [0, 0, 0, 1]  

hidden_size = 5
learning_rate = 0.1
epochs = 100

Wxh = [[0.1]*4 for _ in range(hidden_size)]  
Whh = [[0.1]*hidden_size for _ in range(hidden_size)]  
Why = [[0.1]*hidden_size for _ in range(4)]  
bh = [0.1]*hidden_size  
by = [0.1]*4  

h_prev = [0]*hidden_size  
hidden_states = []

for word in X:
    h = [
        math.tanh(
            sum([Wxh[i][j] * word[j] for j in range(4)]) +  
            sum([Whh[i][k] * h_prev[k] for k in range(hidden_size)]) +  
            bh[i]
        ) 
        for i in range(hidden_size)
    ]
    hidden_states.append(h)
    h_prev = h

output = [
    sum([Why[i][j] * h_prev[j] for j in range(hidden_size)]) + by[i]
    for i in range(4)
]

exp_output = [math.exp(o) for o in output]
softmax = [e / sum(exp_output) for e in exp_output]

loss = -math.log(softmax[3])  


dWhy = [[0.0]*hidden_size for _ in range(4)]
dby = [0.0]*4
for i in range(4):
    grad = softmax[i] - (1 if i == 3 else 0)  
    for j in range(hidden_size):
        dWhy[i][j] += grad * h_prev[j]
    dby[i] += grad

for i in range(4):
    for j in range(hidden_size):
        Why[i][j] -= learning_rate * dWhy[i][j]
    by[i] -= learning_rate * dby[i]

print("Predicted Results:", softmax)
print("loss:", loss)

