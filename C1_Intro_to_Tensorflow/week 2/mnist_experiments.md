

## Parametric Experiments
**Outline**
1. Experiment: just FC Network. `Adjusting neurons`. Smallest underfitting network
2. Experiment: just CNN Network. `Adjusting filters`. Smallest underfitting network
- [ ] Experiment: just CNN Network. `All datasets` & `adjusting filters` . Smallest underfitting network for each
- [ ] Experiment: Just CNN Network. compare `Optimizers` , same Dataset. 
- [ ] Experiment: Just FC Network. compare `Optimizers`, same Dataset.

#### 1. Adjusting just FC Neurons
Parameter	: Flatten -> FC 3 -> FC 10  
Observation	: Reaches 53% acc by epoch 3. 74% by epoch 10. Platues to 76% around 25 epochs  
Analysis	: Network learns. Not overfitting for sure because max is 76%.  

Change		: FC 2 -> FC 10  
Observation	: Reaches 37% by epoch 3. Platues to 53% by epoch 19  
Analysis	: Completely underfitting network because max learning is 53%. Lowerbound of minimum parameters. FC 3 platues around 76%. Starts to learn. Minimum learning network.

Change		: FC 5 -> 6 -> 10  
Observation	: Increases upper limit of accuracy, in same number of epochs across experiments  

Imp Observations
- First few epoch accuracy should be good representation of model capacity / complexity. and data pattern complexity. 
- Plotting increasing neurons and accuracy graph wrt epochs for all models on same graph. Can see, how increasing model capacity leads to increasing ability to extract complex patterns. (Andrew ng's slide)
- Need to standardize num of epochs = 10 or 20. But don't time or resources in retraining. For quick experiments, epochs = 5 is a good number. Understand initial behaviour quickly.
- Impact of improvements is most seen on barely learning network. Whether it improves drastically or not.

Tensorboard: Visualize scalars of metrics & loss, weights histogram, graph of NN, 
- [x] tensorboard needs each run into a different folder. same for WandB. use `datetime.datetime.now().strftime("%Y%m%d-%H%M%S")`
- [ ] tensorboad doesn't show loss updates wrt batch no. Figure out those plots as well
- [ ] doesn't have ability to add custom graph

#### 2. Fully CNN Network
Parameter: `minimum filters` of barely learning network  
Change: (5,10,10,40) from (5,10,15,25)  
Observation: 1st epoch. avg accuracy 62. 2nd epoch 93% accuracy
Analysis: Still overfitting. 
Next: May be use 3 blocks instead of 4.

Parameter: `blocks = 3` (5,10,40)
Observation: 1st Epoch acc: 0.46. 2nd epoch acc: 0.88
Analysis: Still overfitting. How to get accuracy of 50%?

Change: `reducing neurons in decision layer` (10*15 -> 40)
Observation: accuracy per epochs(0.41, 0.88, 0.95). params 37k

Change: `10 neurons everywhere & 3 blocks`
Observation: accuracy/epoch (0.21, 0.78, 0.92). params 11k

TODO
- [ ] train & validation data. to check at what epoch we actually start overfitting. plot those graphs
  - [ ] do different architectures start overfitting at different epochs?

Base NN: 
```python
model = keras.models.Sequential([
    keras.layers.Input(shape=(28,28,1)),
    keras.layers.Conv2D(5,7,activation="relu"),
    keras.layers.Conv2D(10,7,activation= "relu"),
    keras.layers.Conv2D(15,7,activation= "relu"),
    keras.layers.Conv2D(25,7,activation= "relu"),
    keras.layers.Conv2D(10,1,activation="relu"),
    keras.layers.Conv2D(10*15,3,activation="relu"),
    keras.layers.Conv2D(10,1,activation="relu"),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10,activation="softmax", use_bias=False),
])
```

Observations:
1. Network capacity > pattern complexity. avg 46% accuracy reached in first epoch. means, last values are much closer to double. Hence 70% accuracy, after looking at each data just once. Network is just learning
2. FC model parameters vs CNN model parameters. 10k vs 40k

**IMP**
1. model's one time behaviour vs behaviour multiple times. its important to observe one time training behaviour thoroughly. and also average of at least 3 runs should be checked later.
2. save model architecture, model parameters, model training history, model logs. *if you can save all these*, then there is no need to train frequently. 
4. Improve network. Reinitialize with extra flags. `(use_bias=False)`. Easy parameter calculations, confirmable

#### 3. Just CNN: MNIST, CIFAR10, CIFAR100
```python
from tensorflow.keras.layers import Conv2D as Conv2d
# MNIST
model = keras.Sequential([
    Conv2d(10, 7, input_shape = (28,28,1)),
    Conv2d(20, 7),
    Conv2d(30, 7),
    Conv2d(50, 7),

    Conv2d(10, (1,1)),
    Conv2d(10*15, 4-1),
    Conv2d(10, (1,1)),

    layers.Dense(10)
])

# CIFAR10
model = keras.Sequential([
    Conv2d(10, 8, input_shape=(32,32,3)),
    Conv2d(20, 8),
    Conv2d(30, 8),
    Conv2d(50, 8),

    Conv2d(10, (1,1)),
    Conv2d(10*25, 4-1),
    Conv2d(10, (1,1)),

    layers.Dense(10),
])

# CIFAR100
model = keras.Sequential([
    Conv2d(100, 8, input_shape=(32,32,3)),
    Conv2d(200, 8),
    Conv2d(400, 8),
    Conv2d(500, 8),

    Conv2d(100, (1,1)),
    Conv2d(100*15, 4-1),
    Conv2d(100, (1,1)),

    layers.Dense(10),

])
```
