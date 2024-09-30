from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 程序最多只能占用指定gpu5%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.compat.v1.Session(config=config)
from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop
from keras.models import Model
from keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from keras.callbacks import TensorBoard
from tensorflow import keras
from keras import losses


def sampling(args):
    z_mean, z_log_var = args
    epsilon = 0.1*K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]))
    return z_mean + K.exp(z_log_var / 2) * epsilon
def concatenate(inputs):
    [x, y] = inputs
    z = K.concatenate([x, y], axis=1)
    return z
def equal(x):
    return x
def vae_loss(y,x):
    xent_loss = K.binary_crossentropy(x, y)
    kl_loss = - 0.5 * K.mean(K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
    vae_loss = xent_loss + kl_loss
    return vae_loss
def mse_loss(y,x):
    return 5*losses.mean_squared_error(x, y)
def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
    if epoch % 600 == 0 and epoch != 0:
        lr = K.get_value(vae.optimizer.lr)
        K.set_value(vae.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(vae.optimizer.lr)


# 可改变变量
batch_size = 128
epochs = 4800
fre_dims = 401
latent_dims = 8
S11_dims = 401
mean_dims = S11_dims + latent_dims
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
fre_train = np.load('fre_train.npy')
S11 = np.load('S11_train.npy')


# 输入
x_input = Input(shape=(12, 12,), name='x_input')
freq_input = Input(shape=(fre_dims,), name='freq_input')

###### encoder网络 ###########
encoder = Sequential(name='encoder')
encoder.add(keras.layers.Reshape([12, 12, 1]))
encoder.add(keras.layers.Conv2D(32, (3, 3), activation='selu', padding='same',
                                kernel_initializer=keras.initializers.he_normal(seed=5),
                                bias_initializer='zeros'))

encoder.add(keras.layers.Conv2D(32, (3, 3), activation='selu', padding='same',
                                kernel_initializer=keras.initializers.he_normal(seed=5),
                                bias_initializer='zeros'))
encoder.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
encoder.add(keras.layers.Flatten())

encoder.add(Dense(128, activation='selu',
            kernel_initializer=keras.initializers.he_normal(seed=5),
            bias_initializer='zeros'))
encoder_h = encoder(x_input)

z_mean = Dense(latent_dims, activation='selu', name='z_mean',
               kernel_initializer=keras.initializers.he_normal(seed=5),
               bias_initializer='zeros')(encoder_h)
z_log_var = Dense(latent_dims, activation='selu', name='z_var',
                  kernel_initializer=keras.initializers.he_normal(seed=5),
                  bias_initializer='zeros')(encoder_h)

###### 重参与堆叠 ##############
z = Lambda(sampling, output_shape=(latent_dims,), name='resample')([z_mean, z_log_var])

encoder1 = Model(x_input, [z_mean, z_log_var, z], name='encoder')
encoder1.summary()

# 堆叠层，重参z和fre堆叠
latent = Lambda(concatenate, name='concatenate')([z_mean, freq_input])  # 维度是1+fre_dims

######### predictor网络 #########
predictor_input = Input(shape=(mean_dims,), name='predictor_input')
predictor = Sequential(name='predictor')
predictor.add(Dense(500, activation='relu',
                    kernel_initializer=keras.initializers.he_normal(seed=5),
                    bias_initializer='zeros'))
predictor.add(Dense(1000, activation='relu',
                    kernel_initializer=keras.initializers.he_normal(seed=5),
                    bias_initializer='zeros'))
predictor.add(Dense(2000, activation='relu',
                    kernel_initializer=keras.initializers.he_normal(seed=5),
                    bias_initializer='zeros'))
predictor.add(Dense(1000, activation='relu',
                    kernel_initializer=keras.initializers.he_normal(seed=5),
                    bias_initializer='zeros'))
predictor.add(Dense(500, activation='relu',
                    kernel_initializer=keras.initializers.he_normal(seed=5),
                    bias_initializer='zeros'))
predictor.add(Dense(250, activation='relu',
                    kernel_initializer=keras.initializers.he_normal(seed=5),
                    bias_initializer='zeros'))

predictor_h = predictor(predictor_input)

# predictor输出
predictor_output = Dense(S11_dims, activation='sigmoid')(predictor_h)
predictor1 = Model(predictor_input, predictor_output, name='predictor')
predictor1.summary()

S11_output = predictor1(latent)
S11_output = Lambda(equal, name='S11_output')(S11_output)

###### decoder网络 ############
decoder_input = Input(shape=(latent_dims,), name='decoder_input')
decoder = Sequential(name='decoder')
decoder.add(Dense(128, activation='selu'))
decoder.add(Dense(6*6*32, activation='selu'))
decoder.add(keras.layers.Reshape((6, 6, 32)))
decoder.add(keras.layers.UpSampling2D((2, 2)))
decoder.add(keras.layers.Conv2D(32, (3, 3), activation='selu', padding='same'))
# decoder.add(keras.layers.UpSampling2D((2, 2)))
decoder.add(keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
decoder.add(keras.layers.Reshape([12, 12]))
decoder_h = decoder(decoder_input)

# decoder输出
decoder1 = Model(decoder_input, decoder_h, name='decoder')
decoder1.summary()
y = decoder1(z)
y = Lambda(equal, name='y_output')(y)

####### vae模型 ############
vae = Model([x_input, freq_input], [y, S11_output])
optimizer = RMSprop(lr=0.001)

vae.compile(optimizer=optimizer,
            loss={'y_output': vae_loss, 'S11_output': mse_loss},
            experimental_run_tf_function=False)
vae.summary()
tbCallBack = TensorBoard(log_dir=r'loss',  # log 目录
                         histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                         write_graph=True,  # 是否存储网络结构图
                         write_grads=True,  # 是否可视化梯度直方图
                         write_images=True,  # 是否可视化参数
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)
vae.fit({'x_input': x_train, 'freq_input': fre_train}, {'y_output': y_train, 'S11_output': S11},
        epochs=epochs,
        shuffle=True,
        batch_size=batch_size,
        callbacks=[tbCallBack])

encoder1.save_weights(r'encoder_weights.h5')
predictor1.save_weights(r'predictor_weights.h5')
decoder1.save_weights(r'decoder_weights.h5')
vae.save_weights(r'vae_weights.h5')



