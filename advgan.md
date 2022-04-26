# ADVGAN

## 知识蒸馏

### 什么是知识蒸馏？

近年来，神经模型在几乎所有领域都取得了成功，包括极端复杂的问题。然而，这些模型体积巨大，有数百万(甚至数十亿)个参数，因此不能部署在边缘设备上。

知识蒸馏指的是模型压缩的思想，**通过一步一步地使用一个较大的已经训练好的网络去教导一个较小的网络确切地去做什么**。“软标签”指的是大网络在每一层卷积后输出的feature map。然后，通过尝试复制大网络在每一层的输出(不仅仅是最终的损失)，小网络被训练以学习大网络的准确行为。

来张图：

![img](https://cdn.jsdelivr.net/gh/junsuyiji/tuchang@master/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9LWVNEVG1PVlp2cVM5eURJYlVNNlJ2Q2djN2IxSlJzeGVvTlhKZjBNM1QwNUJ4TzlmTlRPSGRnNGtQQkVLbWliZ2N6ZlNpYWhSV2ptMjR4Vll2WkYxVFhBLzY0MA)![img](https://cdn.jsdelivr.net/gh/junsuyiji/tuchang@master/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2pwZy9LWVNEVG1PVlp2cVM5eURJYlVNNlJ2Q2djN2IxSlJzeDJCZkpmTUhma29sVEVxaWFjVzBnRW9pYzBMc1NmZmlhYlA5UThIUGRGWm5XZXhqWExBb01Lc045Zy82NDA)



为什么需要这样做？

**也许是因为没钱，也许是因为要变小**



#### **网络流程**

1. **训练教师网络**：首先使用完整数据集分别对高度复杂的教师网络进行训练。这个步骤需要高计算性能，因此只能在离线(在高性能gpu上)完成。
2. **构建对应关系**：在设计学生网络时，需要建立学生网络的中间输出与教师网络的对应关系。这种对应关系可以直接将教师网络中某一层的输出信息传递给学生网络，或者在传递给学生网络之前进行一些数据增强。
3. **通过教师网络前向传播**：教师网络前向传播数据以获得所有中间输出，然后对其应用数据增强(如果有的话)。
4. **通过学生网络反向传播**：现在利用教师网络的输出和学生网络中反向传播误差的对应关系，使学生网络能够学会复制教师网络的行为。

**小总结**：

知识蒸馏，可以将一个网络的知识转移到另一个网络，两个网络可以是同构或者异构。做法是先训练一个teacher网络，然后使用这个teacher网络的输出和数据的真实标签去训练student网络。知识蒸馏，可以用来将网络从大网络转化成一个小网络，并保留接近于大网络的性能；也可以将多个网络的学到的知识转移到一个网络中，使得单个网络的性能接近emsemble的结果





**简单易懂的栗子**

```python
#载入数据
#使用fashion_mnist数据集，输入图像大小为28*28，共分为10类。
import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images/255
test_images = test_images/255
train_labels = tf.one_hot(train_labels, depth=10)
test_labels = tf.one_hot(test_labels, depth=10)


"""
教师模型
使用一个4层MLP来作为教师模型。
训练过程中，模型最后使用softmax层来计算损失值。
训练结束后，更改最后的softmax层，以便生成软标签，其中T=2。同时，为了防止误操作，将教师模型冻结。
需要注意的是，虽然更改后教师模型不再进行训练，但仍需要使用compile函数进行配置，否则无法调用predict函数。
"""

# 构建并训练教师模型
inputs = keras.layers.Input(shape=(28,28))
x = keras.layers.Flatten()(inputs)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dense(10)(x)
outputs = keras.layers.Softmax()(x)

t_model = keras.Model(inputs, outputs)
t_model.summary()

callback = [keras.callbacks.EarlyStopping(patience=10 ,restore_best_weights=True)]
t_model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

t_model.fit(train_images, train_labels, epochs=500, validation_data=(test_images, test_labels),callbacks=callback)

# 更改教师模型以便后续生成软标签
x = t_model.get_layer(index=-2).output
outputs = keras.layers.Softmax()(x/3)
Teacher_model = keras.Model(t_model.input, outputs)
Teacher_model.summary()
Teacher_model.trainable = False

Teacher_model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


"""
学生模型
本文使用一个2层MLP作为学生模型。
学生模型构建完成后不进行训练，在后续的蒸馏过程中进行训练。
需要注意的是，学生模型最后一层不加Softmax层。
"""

inputs = keras.layers.Input(shape=(28,28))
x = keras.layers.Flatten()(inputs)
x = keras.layers.Dense(128, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)

Student_model = keras.Model(inputs, outputs)
Student_model.summary()

"""
知识蒸馏过程
学生模型进行蒸馏时，损失函数包括两部分：
Loss1：学生模型softmax输出值与真实标签的之间的损失（交叉熵）；
Loss2：学生模型软化后的softmax输出值（T=2）与教师模型生成的软标签之间的损失（KL散度）。
则，Loss = 0.1*Loss1 + 0.9*Loss2。
本文通过重写Model类来实现。

"""

class Distilling(keras.Model):
  def __init__(self, student_model, teacher_model, T, alpha):
    super(Distilling, self).__init__()
    self.student_model = student_model
    self.teacher_model = teacher_model
    self.T = T
    self.alpha = alpha

  def train_step(self, data):
    x, y = data
    softmax = keras.layers.Softmax()
    kld = keras.losses.KLDivergence()
    with tf.GradientTape() as tape:
      logits = self.student_model(x)
      soft_labels = self.teacher_model(x)
      loss_value1 = self.compiled_loss(y, softmax(logits))
      loss_value2 = kld(soft_labels, softmax(logits/self.T))
      loss_value = self.alpha* loss_value2 + (1-self.alpha) * loss_value1
    grads = tape.gradient(loss_value, self.student_model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.student_model.trainable_weights))
    self.compiled_metrics.update_state(y, softmax(logits))
    return {'sum_loss':loss_value, 'loss1': loss_value1, 'loss2':loss_value2, }
  
  def test_step(self, data):
    x, y = data
    softmax = keras.layers.Softmax()
    logits = self.student_model(x)
    loss_value = self.compiled_loss(y, softmax(logits))
    return {'loss':loss_value}

  def call(self, inputs):
    return self.student_model(inputs)
 
    
"""
蒸馏过程加入早停止机制，监视val_loss。
"""
    

distill = Distilling(Student_model, Teacher_model, 2, 0.9)
distill.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False))

callback = [keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)]

distill.fit(train_images, train_labels, epochs=500, validation_data=(test_images, test_labels), callbacks=callback)


```



## 对抗攻击

### 简介

**对抗攻击是机器学习与计算机安全的结合（Intersection）：对输入样本故意添加一些人无法察觉的细微的干扰，导致模型以高置信度给出一个错误的输出。**

为什么会有对抗攻击？我们面临的问题是什么？

**以前设计的机器学习模型在面对攻击者精心设计的对抗攻击时往往会达不到预期的准确度**

**对抗攻击的分类**

- 白盒攻击，称为White-box attack，也称为open-box 对**模型和训练集完全了解**，这种情况比较简单，但是和实际情况不符合。
- 黑盒攻击，称为Black-box attack，对模型不了解，**对训练集不了解或了解很少**。这种攻击和实际情况比较符合，主要也是主要研究方向。
- 定向攻击，称为targeted attack，**对于一个多分类网络，把输入分类误判到一个指定的类上**
- 非定向攻击，称为non-target attack，**只需要生成对抗样本来欺骗神经网络**，可以看作是上面的一种特例。
- 来张图![uTools_1649504313806](https://cdn.jsdelivr.net/gh/junsuyiji/tuchang@master/uTools_1649504313806.png)

**对抗攻击的目标**

- 减少置信度：减小输入分类的置信度，从而引起歧义。
- 无目标分类：将输出分类更改为与原始类不同的任何类。
- 有目标分类：强制将输出分类为特定的目标类。
- 源到目的分类：强制将特定的输入的输出分类为特定的目标类

**对抗攻击分类总结:**

1. 按照攻击者是否知道目的网络的结构参数，可以将对抗攻击分为**白盒攻击**和**黑盒攻击**。
2. 实际中，根据目的网络最终得到的分类结果是否是攻击者预先设计好的，将对抗攻击分为**目标攻击**和**非目标攻击**。

**目前，在对抗攻击防御上存在三个主要方向：**

1）在学习过程中修改训练过程或者修改的输入样本。

2）修改网络，比如：添加更多层/子网络、改变损失/激活函数等。

3）当分类未见过的样本时，用外部模型作为附加网络。

第一个方法没有直接处理学习模型。另一方面，另外两个分类是更加关心神经网络本身的。这些方法可以被进一步细分为两种类型：（a）完全防御；（b）仅探测（detection only）。「完全防御」方法的目标是让网络将对抗样本识别为正确的类别。另一方面，「仅探测」方法意味着在对抗样本上发出报警以拒绝任何进一步的处理

### 栗子



**小小的总结**

对抗攻击可以实现，其本质**一方面是因为神经网络乃至深度学习可以实现分类和预测目的的原理还比较模糊，因此可以利用这种不确定性来混淆模型**；另一方面是因为**数据本身就不能按照抽取的特征得到固定的分类结果，每一个个体具有比较大的误差因素**。因此，**模型容易受到对抗攻击是因为模型的泛化能力不够**，在处理非训练数据时容易得到错误的结果。提高模型的泛化能力才是最好的防御策略。

## ADVGAN

**为什么用GAN**

GAN网络在图像生成和操作方面的效果很好。对抗损失和图像到图像的网络结构来学习从原始图像到扰动的映射，使产生的扰动和原始图像分不开，而且保证了攻击的效果。

AdvGAN 的核心思想**是将干净样本通过 GAN 的生成器映射成对抗扰动**，**然后加在对应的干净样本中，判别器负责判别输入的样本是否为对抗样本。**

![uTools_1649504536426](https://cdn.jsdelivr.net/gh/junsuyiji/tuchang@master/uTools_1649504536426.png)



上图为 AdvGAN 的总体架构，主要由三部分组成：**生成器 G、判别器 D 和目标神经网络 C**。**将干净样本 x 输入到 G 中生成对抗扰动 G(x)。然后将 x+G(x) 发送给判别器 D，用于区分生成的样本和原始干净的样本**，判别器 D 的目的是**鼓励生成的实例与原始类中的数据不可区分。**

为了实现愚弄学习模型的目标，再将生成的数据 x+G(x) 输入到目标分类模型 C 中，其中输出的损失为$L_{GAN}$ ，$L_{hinge}$表示预测与目标类别 t 目标攻击之间的距离。优化目标损失函数，当模型达到最优时，G(x) 即为对抗扰动。



**换张图，再来一遍**





![img](https://cdn.jsdelivr.net/gh/junsuyiji/tuchang@master/ae6758d6be874adabb69a78d766fa186.png)

给定一个输入x，generator网络生成扰动G(x)，扰动G(x)和输入x相加，一方面x把和G(X)送入discriminator网络训练，另外一方面x+G(x)将送入被攻击的网络训练。G网络用来输出噪声，而D网络的目标函数代表x和x+G(x)的距离度量，而f网络的目标函数代表分类结果和攻击的标签的损失值。
$$
L_{GAN} = E_xlogD(x) + E_xlog(1 - D(x + G(x)))
$$

$$
L_{adv}^{f} = E_xl_f(x + G(x),t)
$$

$l_f$是原网络f的目标函数，t是攻击的标签。

为了限制扰动的大小，作者加入了额外的损失$L_{hinge}$


$$
L_{hinge} = E_xmax(0,||G(x)||_2 - c)
$$
c是一个常数，控制扰动的大小的。

最终的目标函数L表示为
$$
L = L_{adv}^f + {\alpha}L_{GAN} + {\beta }L_{hinge}
$$



**怎么样才算是成功的攻击？**

**个人见解：：生成的x+G(x)在肉眼上与x无分辨，但是却可以扰乱NN的判断**

**问题1：每次攻击都要去判断一下生成的X+G（X）是否与X相同？？**



**ADVGAN的缺点是什么？**

首先看上面问题一

二 GAN有的缺点他都有。。。

三蒸馏的f时候真的能和黑盒中的NN无限接近??

**代码**

```python
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import to_categorical
from keras import layers, Model
from keras.layers import Input, Dense, Dropout, Flatten, Reshape, Activation, Lambda, LeakyReLU
from keras.layers import Conv2D, AveragePooling2D, Conv2DTranspose, BatchNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam, SGD
from keras.metrics import binary_accuracy
from keras import backend as K
import os, cv2, re, random
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow import set_random_seed

class DCGAN():

    def __init__(self):
        #input image dimensions
        self.img_width = 28
        self.img_height = 28
        self.input_shape = (self.img_width, self.img_height, 1) #1 channel for grayscale

        optimizer_g = Adam(0.0002)
        optimizer_d = SGD(0.01)

        inputs = Input(shape=self.input_shape)
        outputs = self.build_generator(inputs)
        self.G = Model(inputs, outputs)
        self.G.summary()

        outputs = self.build_discriminator(self.G(inputs))
        self.D = Model(inputs, outputs)
        self.D.compile(loss=keras.losses.binary_crossentropy, optimizer = optimizer_d, metrics=[self.custom_acc])
        self.D.summary()

        outputs = self.build_target(self.G(inputs))
        self.target = Model(inputs, outputs)
        self.target.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        self.target.summary()

        self.stacked = Model(inputs=inputs, outputs=[self.G(inputs), self.D(self.G(inputs)), self.target(self.G(inputs))])
        self.stacked.compile(loss=[self.generator_loss, keras.losses.binary_crossentropy, keras.losses.binary_crossentropy], optimizer = optimizer_g)
        self.stacked.summary()

    def generator_loss(self, y_true, y_pred):
        return K.mean(K.maximum(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) - 0.3, 0), axis=-1)
        #||G(x) - x||_2 - c, where c is user-defined. Here it is set to 0.3

    def custom_acc(self, y_true, y_pred):
        return binary_accuracy(K.round(y_true), K.round(y_pred))

    #build the cnn
    def build_discriminator(self, inputs):

        D = Conv2D(32, 4, strides=(2,2))(inputs)
        D = LeakyReLU()(D)
        D = Dropout(0.4)(D)
        D = Conv2D(64, 4, strides=(2,2))(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)
        D = Dropout(0.4)(D)
        D = Flatten()(D)
        D = Dense(64)(D)
        D = BatchNormalization()(D)
        D = LeakyReLU()(D)
        D = Dense(1, activation='sigmoid')(D)
        return D


    def build_generator(self, inputs):
        #c3s1-8
        G = Conv2D(8, 3, padding='same')(inputs)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        #d16
        G = Conv2D(16, 3, strides=(2,2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        #d32
        G = Conv2D(32, 3, strides=(2,2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        residual = G
        #four r32 blocks
        for _ in range(4):
            G = Conv2D(32, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = Activation('relu')(G)
            G = Conv2D(32, 3, padding='same')(G)
            G = BatchNormalization()(G)
            G = layers.add([G, residual])
            residual = G

        #u16
        G = Conv2DTranspose(16, 3, strides=(2,2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        #u8
        G = Conv2DTranspose(8, 3, strides=(2,2), padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)

        #c3s1-3
        G = Conv2D(1, 3, padding='same')(G)
        G = InstanceNormalization()(G)
        G = Activation('relu')(G)
        G = layers.add([G, inputs])

        return G

    def build_target(self, inputs):
        f = Conv2D(64, 5, padding='same', activation='relu')(inputs)
        f = Conv2D(64, 5, padding='same', activation='relu')(f)
        f = Dropout(0.25)(f)
        f = Flatten()(f)
        f = Dense(128, activation='relu')(f)
        f = Dropout(0.5)(f)
        f = Dense(2, activation='softmax')(f)
        return f



    def get_batches(self, start, end, x_train, y_train):
        x_batch = x_train[start:end]
        Gx_batch = self.G.predict_on_batch(x_batch)
        y_batch = y_train[start:end]
        return x_batch, Gx_batch, y_batch


    def train_D_on_batch(self, batches):
        x_batch, Gx_batch, _ = batches

        #for each batch:
            #predict noise on generator: G(z) = batch of fake images
            #train fake images on discriminator: D(G(z)) = update D params per D's classification for fake images
            #train real images on disciminator: D(x) = update D params per classification for real images

        #Update D params
        self.D.trainable = True
        d_loss_real = self.D.train_on_batch(x_batch, np.random.uniform(0.9, 1.0, size=(len(x_batch), 1)) ) #real=1, positive label smoothing
        d_loss_fake = self.D.train_on_batch(Gx_batch, np.zeros((len(Gx_batch), 1)) ) #fake=0
        d_loss = 0.5*np.add(d_loss_real, d_loss_fake)

        return d_loss #(loss, accuracy) tuple


    def train_stacked_on_batch(self, batches):
        x_batch, _, y_batch = batches
        flipped_y_batch = 1.-y_batch

        #for each batch:
            #train fake images on discriminator: D(G(z)) = update G params per D's classification for fake images

        #Update only G params
        self.D.trainable = False
        self.target.trainable = False
        stacked_loss = self.stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)), to_categorical(flipped_y_batch)] )
        #stacked_loss = self.stacked.train_on_batch(x_batch, [x_batch, np.ones((len(x_batch), 1)), to_categorical(y_batch)] )
        #input to full GAN is original image
        #output 1 label for generated image is original image
        #output 2 label for discriminator classification is real/fake; G wants D to mark these as real=1
        #output 3 label for target classification is 1/3; g wants to flip these so 1=1 and 3=0
        return stacked_loss #(total loss, hinge loss, gan loss, adv loss) tuple


    def train_GAN(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = (x_train *2./255 - 1).reshape((len(x_train), 28, 28, 1)) #pixel values in range [-1., 1.] for D
        binary_indices = np.where(y_train == 1)
        x_ones = x_train[binary_indices][:6000]
        y_ones = np.zeros((6000, 1))
        binary_indices = np.where(y_train == 3)
        x_threes = x_train[binary_indices][:6000]
        y_threes = np.ones((6000, 1))
        x_train = np.concatenate((x_ones, x_threes)) #(12000, 28, 28, 1)
        y_train = np.concatenate((y_ones, y_threes)) #1=0, 3=1
        zipped = list(zip(x_train,y_train))
        np.random.shuffle(zipped)
        x_train, y_train = zip(*zipped)
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        self.target.fit(x_train, to_categorical(y_train), epochs=5) #pretrain target

        epochs = 50
        batch_size = 128
        num_batches = len(x_train)//batch_size
        if len(x_train) % batch_size != 0:
            num_batches += 1

        for epoch in range(epochs):
            print("Epoch " + str(epoch))
            batch_index = 0

            for batch in range(num_batches - 1):
                start = batch_size*batch_index
                end = batch_size*(batch_index+1)
                batches = self.get_batches(start, end, x_train, y_train)
                self.train_D_on_batch(batches)
                self.train_stacked_on_batch(batches)
                batch_index += 1


            start = batch_size*batch_index
            end = len(x_train)
            x_batch, Gx_batch, y_batch = self.get_batches(start, end, x_train, y_train)

            (d_loss, d_acc) = self.train_D_on_batch((x_batch, Gx_batch, y_batch))
            (g_loss, hinge_loss, gan_loss, adv_loss) = self.train_stacked_on_batch((x_batch, Gx_batch, y_batch))

            target_acc = self.target.test_on_batch(Gx_batch, to_categorical(y_batch))[1]
            target_predictions = self.target.predict_on_batch(Gx_batch) #(96,2)

            misclassified = np.where(y_batch.reshape((len(x_train) % batch_size, )) != np.argmax(target_predictions, axis=1))[0]
            print(np.array(misclassified).shape)
            print(misclassified)

            print("Discriminator -- Loss:%f\tAccuracy:%.2f%%\nGenerator -- Loss:%f\nHinge Loss: %f\nTarget Loss: %f\tAccuracy:%.2f%%" %(d_loss, d_acc*100., gan_loss, hinge_loss, adv_loss, target_acc*100.))

            if epoch == 0:
                self.save_generated_images("orig", x_batch, 'images')
            if epoch % 5 == 0:
                self.save_generated_images(str(epoch), Gx_batch, 'images')
                self.save_generated_images(str(epoch), Gx_batch[misclassified], 'misclass')


    def save_generated_images(self, filename, batch, dir):
        batch = batch.reshape(batch.shape[0], self.img_width, self.img_height)
        rows, columns = 5, 5

        fig, axs = plt.subplots(rows, columns)
        cnt = 0
        for i in range(rows):
            for j in range(columns):
                axs[i,j].imshow((batch[cnt] + 1)/2., interpolation='nearest', cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("%s/%s.png" % (dir, filename))
        plt.close()




if __name__ == '__main__':
    seed(5)
    set_random_seed(1)
    dcgan = DCGAN()
    dcgan.train_GAN()
```



学习的文章：

[https://blog.csdn.net/hehhehea1/article/details/121630220?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_default&utm_relevant_index=2](https://blog.csdn.net/hehhehea1/article/details/121630220?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1.pc_relevant_default&utm_relevant_index=2)

https://zhuanlan.zhihu.com/p/58089323

https://www.sohu.com/a/405844314_500659?_f=index_pagefocus_4

https://github.com/mathcbc/advGAN_pytorch

[https://www.zhihu.com/search?type=content&q=%E5%AF%B9%E6%8A%97%E6%94%BB%E5%87%BB](https://www.zhihu.com/search?type=content&q=对抗攻击)

知识蒸馏

https://zhuanlan.zhihu.com/p/435919414

https://github.com/dvlab-research/ReviewKD

https://zhuanlan.zhihu.com/p/81467832

https://zhuanlan.zhihu.com/p/85507223

https://zhuanlan.zhihu.com/p/102038521

https://github.com/mathcbc/advGAN_pytorch

https://blog.csdn.net/For_learning/article/details/117304450