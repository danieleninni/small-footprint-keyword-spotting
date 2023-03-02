# definition of tensorflow models 
import tensorflow as tf
import types

from keras.models import Model
from keras import layers as L
from keras import backend as K

# from kapre.utils import Normalization2D


def cnn_trad_fpool3(input_shape):

    model = tf.keras.Sequential(name='cnn_trad_fpool3')

    model.add(L.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1)))
 
    model.add(L.Conv2D(64, (20, 8), strides=(1, 1), padding='same', activation='relu'))
    model.add(L.MaxPool2D((1, 3)))

    model.add(L.Conv2D(64, (10, 4), strides=(1, 1), padding='same', activation='relu'))
    model.add(L.MaxPool2D((1, 1)))

    model.add(L.Flatten())

    model.add(L.Dense(32, activation='relu'))

    model.add(L.Dense(128, activation='relu'))

    model.add(L.Dense(35, activation='softmax'))

    return model


def cnn_one_fpool3(input_shape):

    model = tf.keras.Sequential(name='cnn_one_fpool3')

    model.add(L.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1)))

    model.add(L.Conv2D(54, (32, 8), strides=(1, 1), padding='same', activation='relu'))
    model.add(L.MaxPool2D((1, 3)))

    model.add(L.Flatten())

    model.add(L.Dense(32, activation='relu'))

    model.add(L.Dense(128, activation='relu'))

    model.add(L.Dense(128, activation='relu'))

    model.add(L.Dense(35, activation='softmax'))

    return model


def cnn_one_fstride4(input_shape):

    model = tf.keras.Sequential(name='cnn_one_fstride4')

    model.add(L.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1)))

    model.add(L.Conv2D(186, (32, 8), strides=(1, 4), padding='same', activation='relu'))

    model.add(L.Flatten())

    model.add(L.Dense(32, activation='relu'))

    model.add(L.Dense(128, activation='relu'))

    model.add(L.Dense(128, activation='relu'))

    model.add(L.Dense(35, activation='softmax'))

    return model


def cnn_one_fstride8(input_shape):

    model = tf.keras.Sequential(name='cnn_one_fstride8')

    model.add(L.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1)))

    model.add(L.Conv2D(336, (32, 8), strides=(1, 8), padding='same', activation='relu'))

    model.add(L.Flatten())
    
    model.add(L.Dense(32, activation='relu'))
    
    model.add(L.Dense(128, activation='relu'))
    
    model.add(L.Dense(128, activation='relu'))
    
    model.add(L.Dense(35, activation='softmax'))

    return model


def custom_cnn_simple(
    input_shape, 
    bo_params_redefined = [48, 32, 8, 3, 2, 0.3, 128, 64, 5, 5, 3, 1, 0.3, 0.2, 0.01]
    ):
    '''
      4 cnn blocks, 2x pooling, 2x pooling in time
    '''
    
    nf_sp_1, nf_sp_2, nk_sp_l, nk_sp_r, mp_sp, dp_sp, nf_tp_1, nf_tp_2, nk_tp_l, nk_tp_r, mp_tp_1, mp_tp_2, dp_tp, dp_fc, lr = bo_params_redefined

    model = tf.keras.models.Sequential(name='custom_cnn')

    model.add(L.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1)))
    model.add(L.BatchNormalization())

    filters_pool = [nf_sp_1, nf_sp_2]

    for num_filters in filters_pool:

      model.add(L.Conv2D(num_filters, kernel_size=(nk_sp_l, nk_sp_r), padding='same'))
      model.add(L.BatchNormalization())
      model.add(L.Activation('relu'))

      model.add(L.MaxPooling2D(pool_size=(mp_sp, mp_sp)))
      model.add(L.Dropout(dp_sp))

    filters_pool_in_time = [nf_tp_1, nf_tp_2]

    for p, num_filters in zip([mp_tp_1, mp_tp_2], filters_pool_in_time):

      model.add(L.Conv2D(num_filters, kernel_size=(nk_tp_l, nk_tp_r), padding='same'))
      model.add(L.BatchNormalization())
      model.add(L.Activation('relu'))

      model.add(L.MaxPooling2D(pool_size=(p, 1)))
      model.add(L.Dropout(dp_tp))

    model.add(L.Flatten())
    model.add(L.Dropout(dp_fc))
    model.add(L.Dense(256))
    model.add(L.BatchNormalization())
    model.add(L.Activation('relu'))
    model.add(L.Dense(35, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              metrics=["sparse_categorical_accuracy"])

    return model


def custom_cnn(input_shape):

    model = tf.keras.models.Sequential(name='custom_cnn')

    model.add(L.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1)))
    model.add(L.BatchNormalization())

    filters_pool = [48, 32]

    for num_filters in filters_pool:

        model.add(L.Conv2D(num_filters, kernel_size=(7, 3), padding='same'))
        model.add(L.BatchNormalization())
        model.add(L.Activation('relu'))

        model.add(L.Conv2D(num_filters, kernel_size=(7, 3), padding='same'))
        model.add(L.BatchNormalization())
        model.add(L.Activation('relu'))

        model.add(L.Conv2D(num_filters, kernel_size=(7, 3), padding='same'))
        model.add(L.BatchNormalization())
        model.add(L.Activation('relu'))

        model.add(L.MaxPooling2D(pool_size=(2, 2)))
        model.add(L.Dropout(0.3))

    filters_pool_in_time = [128, 64]

    for p, num_filters in zip([3, 3], filters_pool_in_time):

        model.add(L.Conv2D(num_filters, kernel_size=(5, 7), padding='same'))
        model.add(L.BatchNormalization())
        model.add(L.Activation('relu'))

        model.add(L.MaxPooling2D(pool_size=(p, 1)))
        model.add(L.Dropout(0.3))

    model.add(L.Flatten())
    model.add(L.Dense(512, name='features512'))
    model.add(L.BatchNormalization())
    model.add(L.Activation('relu'))
    model.add(L.Dropout(0.4))
    model.add(L.Dense(256, name='features256'))
    model.add(L.BatchNormalization())
    model.add(L.Activation('relu'))
    model.add(L.Dense(35, activation='softmax'))

    return model


def residual_block(x_input, n_filt, ks=5, downsample=False):
  x = L.Conv2D(filters=n_filt, kernel_size=(ks,ks), strides=(1 if not downsample else 2), padding='same', activation='relu')(x_input)
  x = L.BatchNormalization()(x)
  x = L.Conv2D(filters=n_filt, kernel_size=(ks,ks), strides=1, padding='same', activation='relu')(x)
  x = L.BatchNormalization()(x)

  if downsample:
    x_input = L.Conv2D(kernel_size=1,
               strides=2,
               filters=n_filt,
               padding="same")(x)

  out = L.Add()([x_input, x])
  out = L.ReLU()(out)
  out = L.BatchNormalization()(out)

  return out

def resnet(input_shape=(99, 40), output_shape=35, n_filters=45, num_blocks=3, ks=5, downsample=False, triplet_loss=False):  
  # reshape  
  input_layer = L.Input(shape=input_shape)
  reshape_layer = L.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1))(input_layer)

  # expand input for residual block
  x = L.BatchNormalization()(reshape_layer)

  x = L.Conv2D(filters=n_filters, kernel_size=(ks,ks), strides=(1,1), padding='same')(x)
  x = L.ReLU()(x)
  x = L.BatchNormalization()(x)
  x = L.MaxPooling2D(pool_size=(2,1))(x)
  
  # residual blocks
  for _ in range(num_blocks):
      x = residual_block(x, n_filt=n_filters, ks=ks, downsample=downsample)

  # one cnn block
  x = L.Conv2D(filters=n_filters, kernel_size=(ks,ks), strides=1, padding='same', dilation_rate=8)(x)
  x = L.ReLU()(x)
  x = L.BatchNormalization()(x)

  if not triplet_loss:
    # final part of the network
    x = L.GlobalAveragePooling2D()(x) 
    x = L.Flatten()(x)
    x = L.Dropout(0.4)(x)
    x = L.Dense(output_shape, activation='softmax')(x)

  else:
    # final part of the network

    x = L.MaxPooling2D(pool_size=4)(x)
    x = L.Dropout(0.3)(x)
    x = L.Flatten()(x)
    x = L.Dense(output_shape, activation=None)(x) # No activation on final dense layer
    x = L.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x) # L2 normalize embeddings

  model = Model(inputs=input_layer, outputs=x, name='Resnet')

  return model


def RNNSpeechModel(input_shape=(99, 40), output_shape=35):
    # simple LSTM
    input_layer = L.Input(shape=input_shape)
    x = L.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1))(input_layer)
    
    # x = Normalization2D(int_axis=0)(x)
    # x = L.Permute((2, 1, 3))(x)

    x = L.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.3)(x)
    
    # x = Reshape((125, 80)) (x)
    # keras.backend.squeeze(x, axis)
    x = L.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = L.Bidirectional(L.CuDNNLSTM(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]
    x = L.Bidirectional(L.CuDNNLSTM(64))(x)

    x = L.Dense(64, activation='relu')(x)
    x = L.Dropout(0.3)(x)

    output = L.Dense(output_shape, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output, name='RNNSpeechModel')

    return model


def AttRNNSpeechModel(input_shape=(99, 40), output_shape=35, rnn_func=L.LSTM):
    # simple LSTM
    input_layer = L.Input(shape=input_shape)
    x = L.Reshape(input_shape=input_shape, target_shape=(input_shape[0], input_shape[1], 1))(input_layer)

    # x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)
    # x = L.Permute((2, 1, 3))(x)

    x = L.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Dropout(0.3)(x)

    # x = Reshape((125, 80)) (x)
    # keras.backend.squeeze(x, axis)
    x = L.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = L.Bidirectional(rnn_func(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]
    x = L.Bidirectional(rnn_func(64, return_sequences=True))(x)  # [b_s, seq_len, vec_dim]

    xFirst = L.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = L.Dense(128)(xFirst)

    # dot product attention
    attScores = L.Dot(axes=[1, 2])([query, x])
    attScores = L.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = L.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = L.Dense(64, activation='relu')(attVector)
    x = L.Dropout(0.3)(x)
    
    output = L.Dense(output_shape, activation='softmax', name='output')(x)

    model = Model(inputs=input_layer, outputs=output, name='AttRNNSpeechModel')

    return model


################################################
# NOTE: define all the models above this line! #
################################################


models = [f for f in globals().values() if type(f) == types.FunctionType]
models_names = [str(f).split()[1] for f in models]


def available_models():

    print('Available models:')
    for name in models_names:
        print(name)


def select_model(model_name, input_shape):

    model_index = models_names.index(model_name)
    model = models[model_index](input_shape)
    print('Selected model:', model_name)

    return model