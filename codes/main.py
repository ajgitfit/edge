# installation files
pip install tensorflow
pip install scipy
pip install tensorflow-addons
pip install -q -U tensorflow_addons

from keras.models import load_model
from tensorflow_addons.optimizers import CyclicalLearningRate
import numpy as np
import os
import LRFinder
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow as tf

python -c 'import tensorflow as tf; print(tf.__version__)'
python -c 'import keras; print(keras.__version__)'
# load model
model = load_model('my_model.h5')
#perform processing using model
f=open("experiment_values.csv","a+")
str1=" "
for root,path,file in  os.walk("/home/user"):
	for f in file:	
		for i in range(1,5):
			d={1:"m1.h5",2:"m2.h5",3:"m3.h5",4:"m4.h5"}
			print(d[i])
			str1+="var"+";"
			img=Image.open(filename)
     	img=img.resize((100,100))
     	img_batch = image.img_to_array(img)
     	predictions=predict(img_batch)
     	classbinary=.predict_classes(img_batch)
		str1+="\n"
		print(str1)
		f.write(str1)
#find suitable learning rate
#find LR range
lr_finder = LRFinder(min_lr=1e-5, max_lr=1e-2, steps_per_epoch=np.ceil(epoch_size/batch_size), epochs=3)

model.fit(X_train, Y_train, callbacks=[lr_finder])
lr_finder.plot_loss()
#apply the min and max ranges
#Set initial learning rate
#Set Maximal learning rate
#Set Step size
cyclical_learning_rate = CyclicalLearningRate( initial_learning_rate=3e-7,
 maximal_learning_rate=3e-5, step_size=2000, scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
 scale_mode='cycle')

steps_per_epoch = len(x_train) // BATCH_SIZE
clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=INIT_LR,
    maximal_learning_rate=MAX_LR,
    scale_fn=lambda x: 1/(2.**(x-1)),
    step_size=2 * steps_per_epoch
)
optimizer = tf.keras.optimizers.SGD(clr)
#define cyclical learning strategies

def cyclical_lr_np(iter, step_size, base_lr, max_lr):
  cycle = np.floor(1 + (iter / (2*step_size)))
  x = np.abs((iter/step_size) - 2*cycle + 1)

  lr = base_lr + (max_lr-base_lr) * (np.max((0, (1-x))))

  return lr

cosine_LR = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
model.fit(X_train, Y_train, epochs=100, callbacks=[cosine_LR])
