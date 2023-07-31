```python
    keras.layers.Conv2D(40,3,activation="relu"),
    keras.layers.Conv2D(10,1,activation="relu"), # Should do filtering, but also final activation & decision making
    keras.layers.Conv2D(10,1),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Activation("softmax")
```