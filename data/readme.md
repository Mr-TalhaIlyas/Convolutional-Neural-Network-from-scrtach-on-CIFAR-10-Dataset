You can easily download the data by using

```python
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

if you dont want to use keras at all then just download the data by using above line 1st and then save the downloaded arrays via numpy like

```python
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
np.save('/data/train_data).npy', x_train)
np.save('/data/train_label).npy', y_train)
np.save('/data/test_data).npy', x_test)
np.save('/data/test_label).npy', y_test)
```

and load like

```python
import numpy as np


train_data = np.load('/data/train_data).npy') 
train_label = np.load('/data/train_label).npy') 
test_data = np.load('/data/test_data).npy') 
test_label = np.load('/data/test_label).npy') 
```