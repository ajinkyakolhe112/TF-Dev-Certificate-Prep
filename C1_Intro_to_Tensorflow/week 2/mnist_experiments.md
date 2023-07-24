

#### Adjusting just FC Neurons. 
Parameter	: Flatten -> FC 3 -> FC 10. 
Observation	: Reaches 53% acc by epoch 3. 74% by epoch 10. Platues to 76% around 25 epochs
Analysis	: Network learns. Not overfitting for sure because max is 76%. 

Change		: FC 2 instead of FC 3 -> FC 10
Observation	: Reaches 37% by epoch 3. Platues to 53% by epoch 19
Analysis	: Completely underfitting network because max learning is 53%. Lowerbound of minimum parameters. FC 3 platues around 76%. Underfitting to Fitting better

Change		: FC 5 -> 6 -> 10
Observation	: Increases upper limit of accuracy, in same number of epochs across experiments

Imp Observations
- First few epoch accuracy should be good representation of model capacity / complexity. and data pattern complexity. 
- Plotting increasing neurons and accuracy graph wrt epochs for all models on same graph. Can see, how increasing model capacity leads to increasing ability to extract complex patterns. (Andrew ng's slide)
- Need to standardize num of epochs

Future Improvements
- [x] tensorboard needs runs of each run, into a different folder
- [ ] tensorboad doesn't show loss updates wrt batch no
- [ ] don't have ability to add custom graph

#### Comparing optimizers
Base NN?

#### Fully CNN Network
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
1. Network capacity > pattern complexity. avg 46% accuracy reached in first epoch. means, last values are much closer to double. Hence 70% accuracy, after looking at each data just once. 
2. FC model parameters vs CNN model parameters. 10k vs 40k
