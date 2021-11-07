from keras.models import Sequential
from keras.layers import Merge, Activation


class WassersteinGAN():

    def __init__(self, discriminator, generator):
        self.d = discriminator
        self.g = generator
        self.d_on_g = self.generator_containing_discriminator(self.d, self.g)

    def train(self, X, y):
        raise NotImplementedError

    def train_d_on_batch(self):
        raise NotImplementedError

    def train_g_on_batch(self):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def generator_containing_discriminator(self, discriminator, generator):
        model_input = Sequential()
        model_gen = Sequential()

        model_input.add(Activation('linear', input_shape=(
        self.params['tile_size'][2], self.params['tile_size'][0], self.params['tile_size'][1], len(self.params['channels']))))
        model_gen.add(generator)
        merge = Merge([model_input, model_gen], mode='concat', concat_axis=-1)

        sequential = Sequential()

        sequential.add(merge)
        discriminator.trainable = False
        sequential.add(discriminator)
        return sequential