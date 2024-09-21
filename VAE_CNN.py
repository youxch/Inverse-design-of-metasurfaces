import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, losses, callbacks, optimizers, backend as K
import random

# ==================== GPU 配置 ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 设置 GPU 显存占用
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 限制 TensorFlow 只使用指定的 GPU，并按需分配显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # 可选：限制 GPU 显存使用上限（例如 30%）
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpu,
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=你的显存上限)])
    except RuntimeError as e:
        print(e)

# ==================== 导入必要的库 ====================
import matplotlib.pyplot as plt
from scipy.stats import norm

# ==================== 定义辅助函数 ====================
def read_input_file(path):
    """读取输入文件并转换为矩阵。"""
    with open(path) as f:
        data = [list(map(float, line.strip().split())) for line in f]
    matrix = np.array(data)
    return matrix.transpose()

def read_s_file(path):
    """读取 S 参数文件并转换为矩阵。"""
    with open(path) as f:
        data = [list(map(complex, line.strip().split(','))) for line in f]
    matrix = np.array(data)
    return matrix.transpose()

def sampling(args):
    """重参数技巧采样函数。"""
    z_mean, z_log_var = args
    epsilon = 0.001 * K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

def concatenate(inputs):
    """将两个张量在指定轴上连接。"""
    x, y = inputs
    return K.concatenate([x, y], axis=1)

def vae_loss(y_true, y_pred):
    """定义 VAE 的损失函数，包括重构损失和 KL 散度。"""
    reconstruction_loss = 0.2 * losses.binary_crossentropy(y_true, y_pred)
    kl_loss = -0.5 * K.mean(
        K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    ) / 144
    return reconstruction_loss + kl_loss

def mse_loss(y_true, y_pred):
    """定义均方误差损失函数。"""
    return 20 * losses.mean_squared_error(y_true, y_pred)

def scheduler(epoch, lr):
    """学习率调度函数，每隔 600 个 epoch 学习率减小为原来的 1/10。"""
    if epoch % 600 == 0 and epoch != 0:
        new_lr = lr * 0.1
        print(f"学习率已更新为：{new_lr}")
        return new_lr
    return lr

# ==================== 设置超参数 ====================
batch_size = 128
epochs = 4800
num_samples = 4031
img_dims = 12 * 12
fre_dims = 401
latent_dims = 8
S21_dims = 401
mean_dims = S21_dims + latent_dims

# ==================== 数据预处理 ====================
def load_data():
    """加载并预处理数据。"""
    input_data = read_input_file(r'E:\project\AWPL\data\str\strinput.txt')[:num_samples, :]
    phase = read_input_file(r'E:\project\AWPL\data\s11\s11phase.txt') / 360  # 相位归一化
    freq = (read_input_file(r'E:\project\AWPL\data\s11\freq.txt') - 8) / 4  # 频率归一化
    freq = np.repeat(freq.reshape(1, fre_dims), num_samples, axis=0)
    input_data = input_data.reshape(num_samples, 12, 12)
    return input_data, input_data, freq, phase

x_train, y_train, freq_train, S21 = load_data()

# ==================== 构建模型 ====================
# ----- 编码器 -----
x_input = keras.Input(shape=(12, 12), name='x_input')
freq_input = keras.Input(shape=(fre_dims,), name='freq_input')

encoder = models.Sequential(name='encoder')
encoder.add(layers.Reshape((12, 12, 1), input_shape=(12, 12)))
encoder.add(layers.Conv2D(32, (3, 3), activation='selu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros'))
encoder.add(layers.Conv2D(32, (3, 3), activation='selu', padding='same',
                          kernel_initializer='he_normal', bias_initializer='zeros'))
encoder.add(layers.MaxPooling2D((2, 2)))
encoder.add(layers.Flatten())
encoder.add(layers.Dense(128, activation='selu',
                         kernel_initializer='he_normal', bias_initializer='zeros'))

encoder_output = encoder(x_input)

# 均值和方差层
z_mean = layers.Dense(latent_dims, activation='selu', name='z_mean',
                      kernel_initializer='he_normal', bias_initializer='zeros')(encoder_output)
z_log_var = layers.Dense(latent_dims, activation='selu', name='z_log_var',
                         kernel_initializer='he_normal', bias_initializer='zeros')(encoder_output)

# 采样层
z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])

# 编码器模型
encoder_model = keras.Model(x_input, [z_mean, z_log_var, z], name='encoder')
encoder_model.summary()

# ----- 预测器 -----
# 将潜在向量和频率连接
latent = layers.Lambda(concatenate, name='concatenate')([z_mean, freq_input])

predictor = models.Sequential(name='predictor')
predictor.add(layers.Dense(500, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros'))
predictor.add(layers.Dense(1000, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros'))
predictor.add(layers.Dense(2000, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros'))
predictor.add(layers.Dense(1000, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros'))
predictor.add(layers.Dense(500, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros'))
predictor.add(layers.Dense(250, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros'))
predictor_output = predictor(latent)
predictor_output = layers.Dense(S21_dims, activation='sigmoid', name='predictor_output')(predictor_output)

# 预测器模型
predictor_model = keras.Model([z_mean, freq_input], predictor_output, name='predictor')
predictor_model.summary()

# ----- 解码器 -----
decoder_input = keras.Input(shape=(latent_dims,), name='decoder_input')

decoder = models.Sequential(name='decoder')
decoder.add(layers.Dense(128, activation='selu', input_shape=(latent_dims,)))
decoder.add(layers.Dense(6 * 6 * 32, activation='selu'))
decoder.add(layers.Reshape((6, 6, 32)))
decoder.add(layers.UpSampling2D((2, 2)))
decoder.add(layers.Conv2D(32, (3, 3), activation='selu', padding='same'))
decoder.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
decoder.add(layers.Reshape((12, 12)))

decoder_output = decoder(decoder_input)

# 解码器模型
decoder_model = keras.Model(decoder_input, decoder_output, name='decoder')
decoder_model.summary()

# ----- VAE 模型 -----
vae_output = decoder_model(z)
vae = keras.Model(inputs=[x_input, freq_input], outputs=[vae_output, predictor_output], name='vae')

# 编译模型
optimizer = optimizers.RMSprop(learning_rate=0.001)
vae.compile(optimizer=optimizer,
            loss={'decoder': vae_loss, 'predictor': mse_loss})

vae.summary()

# ==================== 模型训练 ====================
# 回调函数
tensorboard_callback = callbacks.TensorBoard(log_dir=r'E:\project\AWPL\model\loss',
                                             histogram_freq=0, write_graph=True, write_images=True)
lr_scheduler = callbacks.LearningRateScheduler(scheduler)

vae.fit({'x_input': x_train, 'freq_input': freq_train},
        {'decoder': y_train, 'predictor': S21},
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[tensorboard_callback, lr_scheduler])

# ==================== 保存模型权重 ====================
encoder_model.save_weights(r'E:\project\AWPL\model\loadmodel\encoder_weights.h5')
predictor_model.save_weights(r'E:\project\AWPL\model\loadmodel\predictor_weights.h5')
decoder_model.save_weights(r'E:\project\AWPL\model\loadmodel\decoder_weights.h5')
vae.save_weights(r'E:\project\AWPL\model\loadmodel\vae_weights.h5')
