from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_model(input_shape, num_classes):
    """
    1D CNN for handwriting writer identification on sequence data
    input_shape: (sequence_length, feature_dim, 1)
    """
    # Remove channel dimension for Conv1D
    input_shape_1d = (input_shape[0], input_shape[1])

    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape_1d),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(64, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
