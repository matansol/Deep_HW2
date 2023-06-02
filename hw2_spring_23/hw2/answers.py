"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

1. 
A.  The shape of the tensor is [N x out_features, N x in_features] = $[64\cdot512 , 64\cdot1024]$

B.  The matrix would not be sparse because the layer is fully connected, we have a connection between every two elements in X and Y.
As a result, we will get a weight parameter that isnt a zero value in each element in jacobian.

C.  No we don't have to materialize this jacobian tensor in order to calculate $\delta\mat{X}. 
we will use a matrix multiplation and chain rules $\delta\mat{X} =\pderiv{L}{\mat{Y}}W^T$
As a result instead of materialization the whole jacobian matrix we can only use the expression that appear above because we know the values of $\pderiv{L}{\mat{Y}}$ and $W^T$.

2.
A.  The shape of the tensor is [N x out_features, in_features x out_features] = $[64\cdot512 , 512\cdot1024]$

B.  The matrix would be sparse because only derivatives between some Yi element and weights that
represents connection between same Yi and some Xj will not be zero. 
So since many elements do not hold this connection most of the matrix will be zeros.

C. No we don't have to materialize this jacobian tensor in order to calculate $\delta\mat{X}. 
Because of the same reasons that we write in the previous section.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**

No, backprop is **not required** for training neural networks, after bulding the network we can calculate the gradient for each parameter separately without using the chain rule. the problems with that method is that we will need to calculate for the gradients again every time we changing our network and it much slower because we calculate the same calculating for many parameters.


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
    lr_momentum = 0.007
    lr_rmsprop = 0.0002
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
    wstd = 0.1
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. We can see from the graphs that the one's without dropout has the better accuracy in the train sets - overfit and a bad result on the test set, while as we expected the graphs with the dropout get better results with a higher accuracy on the test set because of the model has better generalization and less overfit on the train set.
The dropout techniqe is used to reduce the overffit by damaging the learning process so it won't be to tilted to the train set. that is why we are expected the model without dropout will do better on the train test but worse on the test set.
Also we can see that a model with a droput, has not only a better accuracy on the test, but also lower loss on the test set, while without droput the loss is higher.
we think the problem is the that dropout precentage was to big and needed to modify to around 0.15.

2. for the green graph represent the model with dropout 0.8 we can see only small changing in the accuracy wich means the model wasn't able to learn much from the data and this is expected because doupout 0.8 means 80% of the network connections are loss, and that is damaging the learning process alot, there for as we expected the orenge, the model with dropout 0.4 as mutch better results than the green one.


"""

part2_q2 = r"""
**Your answer:**

yes it is possible, because the cross-entropy function is a measure of the difference between the predicted probability distribution and the actual probability distribution and not the numbers that the model predict worng.
there can be a situation were the biggest probability of some samples is changing witch make the accuracy to increse but in the same time the probability of other calss is changing to and making the cross-entropy loss to increas.


"""

part2_q3 = r"""
**Your answer:**

1. GD is an learning algorithm for getting to the minimum loss by stepping small stepts for the oposite direction of the gradient in each point. GD using the gradient for the algirithm, GD don't tell us how to calculate the gradients.
backprop is a relative fast way to calculate the gradient for each parameter means $\delta L/\delta p$ when L is the loss function and p in the parameter.


2. the difference is that in GD for each step we calculate the parameters gradients on all our sampels and sum them up.
While in SDG on each step we calculate the gradients only on 1 sample from the data and updating our params base on those gradients.
Each step in GD in more accurate toward the minimum that we want to get to, but calculate the gradients on all the sample take a long long time. 


3. like we said before SGD is much faster especially in DL(Deep Learning) when we are training on **ALOT** of samples so it is almost impossible to use GD as it is. 
another reason is that in DL sometimes our samples are very big and we can't load all of our samples to the computer RAM so it will take even longer to calculate the gradient on all the samples.   
as we seen in calss the loss functions in DL are not convex so they have a local minimun points, by using SGD we dont go directly to the minimum and that helps avoiding those local minimum points.

4. 
A. Yes, in both options the gradient will be the same.
In GD we do a forward pass with all the samples and sum up the loss form all the samples and then backprop and calculate the gradients, in the new method we do a forward pass on baches of the samples and then sum up the loss. 
Knowing that loss is a linear operation we can say that: 
$(\sum \nabla L = \nabla \sum L)$ and therefore both expression are Equivalent.
So the backprop is on the same loss sum so we will get the same gradients.

B.  To calculate the gradient in each layer we need the gradients from the next layer that we get using backprop, and we also need the forward calculation on the layer that we storage in memory for each layer for each batch. because we have a lot of layers and a lot of samples even after doing forward on each batch separatly we still need to keep the result on each layer for all the batches so the computer memory will run out of space. 



"""

part2_q4 = r"""
**Your answer:**

To reduce the memory complexity for evaluating the gradient $\nabla f(x_0)$ using forward mode automatic differentiation (AD), we can exploit the fact that each function $f_i:ℝ → ℝ$ is easy to evaluate and differentiate.

1. A In forward mode AD, we compute the derivative of each function 𝑓𝑖 with respect to its input while evaluating the function itself. we can avoid storing all intermediate values during the computation, and store only the derivative values for each function at each step. This allows us to compute the gradient without storing the intermediate values of the function evaluations, resulting in a memory complexity of O(n), where n is the number of functions in the composition.

1.B To compute the gradient using backward mode AD, we can use a similar approach to reduce memory complexity. In backward mode AD, we compute the derivative of the final output with respect to each intermediate variable. Instead of storing all intermediate values and their derivatives, we can calculate the gradients while traversing the computational graph in reverse order. This way, we only need to store the derivatives at each step, reducing the memory complexity to O(n), similar to forward mode AD.

2. These techniques can be generalized for arbitrary computational graphs. In both forward and backward mode AD, we only need to store the derivatives at each step, which allows us to compute the gradient without storing the intermediate values. As long as the functions in the computational graph are differentiable and easy to evaluate, we can apply these memory-saving techniques to reduce the memory complexity while maintaining the computation cost of O(n).

3. When applied to deep architectures like VGGs or ResNets, these techniques can significantly benefit the backpropagation algorithm. These architectures often involve a large number of layers, resulting in a high memory requirement when storing intermediate values during the backward pass. By using forward or backward mode AD with reduced memory complexity, we can effectively handle deep architectures with limited memory resources. This can enable efficient training and inference in deep neural networks without excessive memory consumption.

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
    hidden_dims = 12
    activation = "relu"
    out_activation = "softmax"
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
    lr = 0.05
    weight_decay = 0.005
    momentum = 0.8
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
1.
Optimization refers to the process of adjusting the parameters of a model to minimize the difference between the predicted output and the actual output.
Meaning that we want to find the best set of parameters that minimize the error or maximize the performance of the model.
Our model doesn't have high Optimization error, we can see that from the decision boundary plot that almost all of the dots are in the right decision boundary means our model accuracy is high.

2.
Generalization error is the difference between the model's performance on the training data and its performance on new data - the test set.
We think our model Generalization error is also low, because as we can see the accuracy and loss plots on the test and the train scores are very close, and the model is doing good on unseen data - the accuracy on the test set is over 90%.

3.
Approximation error refers to the discrepancy between an exact value and some approximation to it.
As we can see from the plots, the results we get are not so far from the optimal one.
But when we look at decision boundary plot, we can see that there are some areas in which the model predict failes. 
As a result approximation error that we do have is probably caused by the fact that our model could not identify well those areas of true class.
To improve the Approximation error he can add some non linear features, and by that we will get a bigger family models witch can get better results.

"""

part3_q2 = r"""
**Your answer:**

We didn't expect the FPR and FNR to be higher because as we said in the previous answer our model preform prety good so we predict the FPR and FNR to be low as there are. But we think it is odd the there is a big defference between those two - FPR is 0.02 and FNR is 0.075 more than 3 times, it is odd because the threshold is 0.5 so we expected that they both will be roughly the sime.


"""

part3_q3 = r"""
**Your answer:**

1. In this case we would like to avoid as mutch FPR scores as possible to reduce the number of expensive tests, because a person with the disease will develop non-lethal symptoms that immediately confirm the diagnosis we can accept more FNR to reduce the FTR so we would choose a higher threshold.

2. In this case we think that getting FNR is worse because then that person chance to die are high, so we would like to avoid FNR as much as possible, not changing the threshold that we get from the ROC curve is pretty good because it give a low FNR (0.02) and still minimaiz the FPR.


"""


part3_q4 = r"""
**Your answer:**
1.
For the columns, when depth is fixed and width is varies, we can see that while the width is increased we get a better results in the validation and the test accuracy, until we get around 90% of accuracy.
Also we can notice that the decision boundaries become more complex as long as the width is increased.
For width=2 in columns 1 and 2, the decision boundries are close to a striaght line and in the others width we get more complex seperate.

2.
For the rows, when the width is fixed and the depth is varies, as long as the depth is increase, we get that the seperates have sharper curves.
We can see that when width=2 as long as the depth is increase the model result is improved, but for the others, the model results while increasing the depth dont improving so much. 

3.
The results that we get on the model with depth=1 width=32 are better than the model with depth=4, width=8.
We can guess that when the model is wider, it increases its representational capacity, allowing it to potentially capture more complex patterns in the data.

4.
As we can see in the tests results, we get better results when we use the optimal threshold.
When we use the optimal threshold, we choose the treshold to be the balances between the FPR and FNR, by reducing both types of non-classifications(FPR, FNR).
As a result we get that the model improve prediction accuracy on the test set..

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

1.
The convolution **without bottleneck** we have $64 \cdot ((256 \cdot 3 \cdot 3) + 1) + $256 \cdot ((64 \cdot 3 \cdot 3) + 1) = 295,232 $ parameters
The convolution **with bottleneck** we have $(256+1)\cdot64 + 64\cdot(3\cdot3\cdot64 + 1) + (64 + 1)\cdot256 = 70,016$ parameters
We can see that using bottleneck is **reducing** a lot of parameters.

2.
In bottleneck block, we need to consider the height and the width dimensions when we calculate 1x1 convolution: the output has 256 chunnels while the input has 64.
Every element in the ouput is computed by a 1X1 kernel.
For, 3x3 convolution: The input has 64 channels, and the output has 64 channels too. Every element in the output is computed by a 3x3 kernel.
The final computation is ((1 * 1 * 256 * 64) + (3 * 3 * 64 * 64) + (1 * 1 * 256 * 64)) * (H * W) = (16,384 + 36,864 + 16,384) * (H * W) = 69,632 * (H * W)

In regular block, we need to consider the height and the width dimensions in the computation of 3x3 convolution: The both the input and the output has 256 channels.
Every element in the output is computed by a 3x3 kernel. Every dot product involves 3x3 multiplications and more 8 additions.
For 3x3 convolution. The input has 256 channels, and the output has a different number of channels, we assume 64 channels.The same like the first convolution, every element of the output is computed by a 3x3 kernel.
So, we will get the next number:
((3 * 3 * 256 * 64) + (3 * 3 * 64 * 256)) * (W * H) = (294,912 + 294,912) * (W * H) = 589,824 * (W * H)

3. Spatial
   In the regular block we have two convolution layers of 3x3 therefore the respective field is 5x5.
   In the bottleneck block we have two convolution layers of 1x1 and one convolution layer of 3x3 therefore the
   respective field is 3x3. We can conclude that the regular block combine the input better in terms of spatial.
   
   
   Across feature map
   In bottleneck block the first layer to smaller dimension because we reduce the number of input channels,
   So, not all of the input channels have the same influence across feature maps.
   On the other hand in the regular block since we don't project the input have the same influence across feature map.



"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

1. By analyzing the graphs we can see that K=32 and K=64 graphs are quite simillar so we will talk on the different in the depth(L).
The first thing we can see is that in L=16 the model failed to learn and we will talk on it in part 2 of the answer. 
As we can expect the deeper the network the better accuracy the model will get to, L=8 get the best results when L=2 get the worst (when we dont count L=16).

We can see that on K=32, L=4 graph in the train set is sharper means he learn faster the train data but in the test set the accuracy isn't raising 

2. As we said in before of the answer the L value that the network isn't traiable is L=16, one of the main reasons is vanishing or exploding gradients - during backpropagation, gradients can become very small or very large, therefore, the network fails to update the lower layers resulting in shallow learning.
The first way to deal with that problem is to use residule blockes, by creating shortcuts that skip few layers by adding the original input to the output of a residual block, the network can learn residual functions that don't suffer from vanishing gradient.
Another way is to use ReLU or Leaky ReLU as our activations function, ReLU and Leaky ReLU allows positive gradients to pass through, which helps prevent gradients from vanishing.
We also added pooling, batch normalization to train the model easier when using high-depths models


"""

part5_q2 = r"""
**Your answer:**

By analyzing the graphs we can see that the number of parameters in each layer (K) has a small impact on the accuracy, it mainly effect on the model learning rate - to bigger the K we get higher training rate. 
In the other Hand, like in experemet 1.1 the dipper model (larger L) get to better accuracy and lower loss

The result for K=128 in this experement are slightly better than the results in experement 1, because wider layers give the model better representation capacity and generalization.


"""

part5_q3 = r"""
**Your answer:**

By analyzing the graphs accuracy we can see that the models learning on the train sets are the same and they get to the same accuracy. But on the test set we see that model with L=3 and L=4 got better accuracy than the on with L=2. 

All models suffer from overffit but L=2 model sufer more propebly because he has less layers and we know that deeper networks get better representation capacity and generalization of the data so they get better accuracy on the test set.


"""

part5_q4 = r"""
**Your answer:**

As we can see on the results we get, when k=32, all the models have almost the same accuracy, and the accuracy graphs are almost identical.
By comparing this experement to 1.1 and 1.3, we can see that the test accuracy of the models in this experemnt is higher, in exp 1.1 we run a cnn model with the values of k=32 and L=8 and get a lower accuracy than now.
We can guess that we get a better accuracy because we using a resent while using a dropout that reduce the overfit and residualblock that help learning more spesific gradient.
Using resent can allow us to use much deep network without the problem of vanishing gradient.

For the models with the values of k=(64, 128, 256), we see that the model with L=2 the accuracy is lower than the others model that almost the same, we can assume that as we see before, deeper model has a better results.
For L=4/8 we get the best results that we get from all the experements, with more a deeper and widther network we can get better representation capacity and better generalization capabilities.

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
1.
The model detect not very well, we can see in the first picture that altough the model idetefiy the objects, when one object overlap aonther one, the model little mistake in the bounding box size.
Morover, in the second picture, we can see that again when one object overlap another one, not only the bounding box size is not very well, the model miss one of the object (the cat) in that picture.
Also the model fail in classifier most of the objects in both pictures.

2.
The reason that can possible be of the model's failures in the calssifier of the objects, is that myabe the model didnt train on a huge dataset of dolphins and most of the examples were of person, and that can lead him to make a incorrent classify.
Another reason is that the first picture is very rare because there are dolphins in the air while probably in most of the pictures they appear in the sea, and that can casue to confusing of the model.
Morover, another reason is that we use the model yolov5s, which has less parameters, therfore it is less accorate, when we try to use a diffrent model like yolov5m, it has a better results in the second picture.


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
We chose to demostrate the next object detection pitfalls: occlusion, model bias and Deformation.
Occlusion:
The picture that demostrate an occlusion is the first picture that it is a pictuare of a road full of cars, that many cars occlude the cars that appear in front of them.
We can see that the model detect the cars that appear more closely in the picture very well but most of the cars that appear further, the model doesnt recognize them and doesnt detect them at all.
We can assume that the reasons of the model detection on that picture are that first of all the cars that appear closely are much bigger and clearer, and we can notice that the bounding box size are very good.
Morover, we can see that the model detect well the cars that are hidden by the previos cars although those cars are not appear fully, probably because that the model assume that those objects suitable to be cars, and the model determine the bounding box size of them well.
Also we can see that the score of car reduce as long as continue forward in the road until the model stop to detact the further objects.

Model bias:
we chose to demostrate a picture with model bias with scarecrow that appears in that picture.
First we can see that the model detect well the bounding box size only half of the scarecrows but the model detact them as a person.
We can see that the fronted scarecrow gets a very high score of 0.91 from the model, we can assume that the model recognize the scarecrow as a person because of the body shape of the object - having two hands and two legs.
We can guss that the model detact the objects as a person and not as a scarecrow probabaly because the data train of the model doesnt contain a large samples of scarecrow.

Deformation:
We chose a picture that contains several vegetables that undergo deformation, for examples we can see a bent cucumber and potato, two carrots that merge and etc.
We can see that almost a half of the objects the model detact well altough there are not in their normal shape, but the vegetbales with extremly rare shape like the cucumber and the potato, the model detact wrong as a banana becasue of the similar shape to it, altough that the cucumber has a diffrent color, green and not yellow and even that the potato's shape doesnt look like a regular banana.
The vegetables that have a little deformation like the apple and the orange get a high score and currect, but the model give to the others a low score and wrong on thier determintaion.
Also need to say that the model doesnt detact at all the tomato.

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