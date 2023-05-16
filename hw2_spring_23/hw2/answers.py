r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

1. 
A. the shape of the tensor is [N x in_features x N x out_features] = [64 x 1024 x 64 x 512]
B. Yes, most of the elements are zeros, all the cells [i][j][d][f] where $j != f$


2.
A. the shape of the tensor is [in_features x out_features x N x out_features] = [1024 x 512 x 64 x 512]
B. Yes, most of the elements are zeros, all the cells [i][j][d][f] where $j != f$


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**

No, backprop is not required for training neural networks, after bulding the network we can calculate the gradient for each parameter separately without using the chain rule. the problems with that method is that we will need to calculate for the gradients again every time we changing our network and it much slower because we calculate the same calculating for many parameters.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 2
    lr = 0.05
    reg = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.2
    lr_vanilla = 0.043
    lr_momentum = 0.0035
    lr_rmsprop = 0.0001
    reg = 0.005
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wastd = 0.2
    lr = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. we can see from the graphs that the one's without dropout has better result both in test and train sets. 
the dropout techniqe is used to reduce the overffit by damaging the learning process so it wont be to tilted to the train set. that is why we are expected the model without dropout will do better on the train test but worse on the test set.
we can see from the graphs that on the train sets our assumptions were true, and the model without the dropout was much better. but on the test set the model with the dropout (the orenge one) wasn't better than the model without dropout as we assumed.
we also see that the acuracy of the orenge graph is about the same in both the train and the test.?????????????????? maybe remove the line?????
we think the problem is the that dropout precentage was to big and needed to modify to around 0.1.

2. for the green graph represent the model with dropout 0.8 we can see only small changing in the accuracy witch means the model wasn't able to learn much from the data and this is expected because doupout 0.8 means 80% of the network connections are loss, and that is damaging the learning process alot. 
there for as we expected the orenge, the model with dropout 0.4 as mutch better results than the green one.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**

yes it is possible, because the cross-entropy function is a measure of the difference between the predicted probability distribution and the actual probability distribution and not the numbers that the model predict worng.
there can be a situation were the biggest probability of some samples is changing witch make the accuracy to increse but in the same time the probability of other calss is changing to and making the cross-entropy loss to increas.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**

1. GD is an learning algorithm for getting to the minimum loss by stepping small stepts for the oposite direction of the gradient in each point. GD using the gradient for the algirithm, GD don't tell us how to calculate the gradients.
backprop is a relative fast way to calculate the gradient for each parameter means $\delta L/\delta p$ when L is the loss function and p in the parameter.

2. the difference is that in GD for each step we calculate the parameters gradients on all our sampels and sum them up.
but in SDG we on each step we calculate the gradients on a 1 sample from the data and updating our params base on that gradients.
each step in GD in more accurate toward the minimum that we want to get to, but calculate the gradients on all the sample take a long long time. 


3. like we said before SGD is much faster espessaly in Deep Learning when we are training on **ALOT** of samples so it is almost impossible to use GD as it is. 
another reason is that in DL sometimes our samples are very big and we can't laod are all data samples to the computer RAM so it will take even longer to calculate the gradient on all the samples.   
as we seen in calss the loss functions in DL are not convex so they have a local minimun points, by using SGD we dont go directly to the minimum and that helps avoiding those local minimum points.

4. 
A. Yes, in both options the gradient will be the same.
In GD we do a forward pass with all the samples and sum up the loss form all the samples and then backprop and calculate the gradients, in the new method we do a forward pass on baches of the samples and then sum up the loss. 

Knowing that loss is a linear operation we can say that: 
$(\sum \nabla L = \nabla \sum L)$ and therefore both expression are Equivalent.
So the backprop is on the same loss sum so we will get the same gradients.

B.  To calculate the gradient in each layer we need the gradients from the next layer that we get using backprop, and we also need the forward calculation on the layer that we storage in memory for each layer for each batch. because we have a lot of layers and a lot of samples even after doing forward on each batch separatly we still need to keep the result on each layer for all the batches so the computer memory will run out of space. 



Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 4
    hidden_dims = 8
    activation = "relu"
    out_activation = "relu"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.01
    weight_decay = 0.005
    momentum = 0.7
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
1. our model doesn't have high optimization error, we can see that from the decision boundary plot that almost all of the dots are in the right decision boundary so our model accuracy is high.

2. the model Generalization error is also low and that we can see from the accuracy and loss plots where the test and train scores are very close so the model is doing good on unseen data - accuracy on the test set is over 90%.

3. same in Approximation error is good enough..................................................................

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**

We didn't expect the FPR and FNR to be higher because as we said in the previous answer our model preform prety good so we predict the FPR and FNR to be low as there are. But we think it is odd the there is a big defference between those two - FPR is 0.02 and FNR is 0.075 more than 3 times, it is odd because the threshold is 0.5 so we expected that they both will be roughly the sime.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**

1. In this case we would like to avoid as mutch FPR scores as possible to reduce the number of expensive test, because a person with the disease will develop non-lethal symptoms that immediately confirm the diagnosis we can accept more FNR to reduce the FTR so we would choose a higher treshold.

2. In this case we think that getting FNR is worse because then that person chance to die are high, so we would like to avoid FNR as much as possible, not changing the threshold that we get from the ROC curve is pretty good because it give a low FNR (0.02) and still minimaiz the FPR.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**
1.
For the columns, when depth is fixed and width is varies, we can see that only when depth=4, the test accuracy increase while the width is increase to, in the other cases, we get the best test accuracy for width=8, and when the width became bigger the test accuracy decrease a little.

2.
For the columns, when depth is fixed and width is varies, we can see that for all thw rows we don't get a monotonic connection while the depth is increase so the test accuracy increase too.
But in tow rows we get that the best accuracy is for the higest depth.

3.
The results that we get on the model with depth=1 width=32 is smaller than the model with depth=4, width=8,
We can assume that we get this result because the first model has only one level of depth. while the second model is depper.
The both models have the same number of total parameters, but the deeper model capture hierarchical representations and complex patterns.

4.
As we can see in the tests results, we get better results when we use the optimal threshold.
When we use the optimal threshold, we choose the treshold to be the balances between the FPR and FNR, as a result we get that the model improve prediction accuracy on the test set.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.01
    weight_decay = 0.003
    momentum = 0.7
    loss_fn = torch.nn.CrossEntropyLoss() 
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""