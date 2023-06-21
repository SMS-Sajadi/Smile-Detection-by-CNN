import tensorflow as tf
import matplotlib.pyplot as plt


class SmileDetectionModel:
    AUTOTUNE = tf.data.AUTOTUNE
    INITIAL_EPOCH = 10
    FINE_TUNE_AT = 20
    history = []

    def __init__(self, DATA_PATH, batchsize, shuffle, test_rate, validation_rate):

        # Loading the dataset
        self.dataset = tf.keras.utils.image_dataset_from_directory(DATA_PATH, batch_size=batchsize, shuffle=shuffle,
                                                                   image_size=(64, 64))

        test_batches = tf.data.experimental.cardinality(self.dataset)
        self.test_dataset = self.dataset.take(int(test_batches // test_rate))
        train_dataset = self.dataset.skip(int(test_batches // test_rate))
        val_batches = tf.data.experimental.cardinality(train_dataset)
        self.validation_dataset = self.dataset.take(int(val_batches // validation_rate))
        self.train_dataset = self.dataset.skip(int(val_batches // validation_rate))

        self.train_dataset = self.train_dataset.prefetch(buffer_size=self.AUTOTUNE)
        self.validation_dataset = self.validation_dataset.prefetch(buffer_size=self.AUTOTUNE)
        self.test_dataset = self.test_dataset.prefetch(buffer_size=self.AUTOTUNE)

    def data_augmentation(self, filp_augment=True, brightness_augment=True):
        layers = []

        if filp_augment:
            layers.append(tf.keras.layers.RandomFlip('horizontal'))
        if brightness_augment:
            layers.append(tf.keras.layers.RandomBrightness(0.3, value_range=(0, 255)))

        return tf.keras.Sequential(layers)

    def compile_model(self, data_augmentation):

        preprocess_input = tf.keras.applications.resnet50.preprocess_input
        base_model = tf.keras.applications.ResNet50(input_shape=(64, 64)+(3,), include_top=False, weights='imagenet')

        for layer in base_model.layers[:self.FINE_TUNE_AT]:
            layer.trainable = False

        Dense1 = tf.keras.layers.Dense(15)
        Dense2 = tf.keras.layers.Dense(30)
        outputlayer = tf.keras.layers.Dense(1)

        inputs = tf.keras.Input(shape=(64, 64)+(3,))
        x = data_augmentation(inputs) if data_augmentation else inputs
        x = preprocess_input(x)
        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = Dense1(x)
        x = tf.keras.activations.relu(x)
        x = Dense2(x)
        x = tf.keras.activations.relu(x)
        outputs = outputlayer(x)
        self.model = tf.keras.Model(inputs, outputs)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self):
        new_history = self.model.fit(self.train_dataset, epochs=self.INITIAL_EPOCH,
                            validation_data=self.validation_dataset)

        self.history.append(new_history)

    def test(self):
        loss, accuracy = self.model.evaluate(self.dataset)
        return loss, accuracy

    def report(self):
        acc = self.history[-1].history['accuracy']
        val_acc = self.history[-1].history['val_accuracy']

        loss = self.history[-1].history['loss']
        val_loss = self.history[-1].history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()
