import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout, LSTM, ConvLSTM2D, TimeDistributed, Reshape, Input, GlobalAveragePooling2D

def create_cnn_lstm_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # ConvLSTM directly processes spatial-temporal data
    x = ConvLSTM2D(32, (3, 3), padding='same', return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=False)(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

    inputs = Input(shape=input_shape)
    
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    
    x = TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)

    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64, return_sequences=False)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_lstm_only_model(input_shape, num_classes):
    """
    LSTM-only model for thermal gesture recognition.
    Memory efficient - no TimeDistributed Conv2D layers.
    
    Args:
        input_shape: (timesteps, height, width, channels) e.g., (100, 24, 32, 1)
        num_classes: Number of output classes
    
    Architecture:
        Input (100, 24, 32, 1) 
        -> Reshape to (100, 768)  [flatten spatial dimensions]
        -> LSTM(256) with return_sequences=True
        -> Dropout(0.3)
        -> LSTM(128) 
        -> Dense(128) + Dropout(0.5)
        -> Output(num_classes)
    
    Memory usage: ~2-3 GB (vs ~10 GB for CNN-LSTM)
    """
    inputs = Input(shape=input_shape)
    
    # Flatten spatial dimensions (24, 32, 1) -> 768
    # Shape: (batch, 100, 24, 32, 1) -> (batch, 100, 768)
    timesteps = input_shape[0]
    spatial_dim = input_shape[1] * input_shape[2] * input_shape[3]  # 24 * 32 * 1 = 768
    x = Reshape((timesteps, spatial_dim))(inputs)
    
    # First LSTM layer - processes flattened spatial-temporal data
    x = LSTM(512, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(x)
    x = Dropout(0.1)(x)
    
    # Second LSTM layer - aggregates temporal information
    x = LSTM(256, dropout=0.1, recurrent_dropout=0.1)(x)
    
    # Dense layers for classification
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_lightweight_cnn_lstm(input_shape, num_classes):

    inputs = Input(shape=input_shape)
    

    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(inputs)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    

    x = TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
    

    x = LSTM(64, return_sequences=False)(x)
    

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
