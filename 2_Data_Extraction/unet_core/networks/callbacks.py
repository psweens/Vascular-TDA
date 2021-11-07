from keras.callbacks import Callback, BaseLogger, History
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import scipy

class OutputHistory(History):

    def __init__(self, plot_name, data_name, metrics=None, save_graphs=True, plot_logs=True):
        self.plot_name = plot_name
        self.data_name = data_name

        splits = os.path.split(plot_name)
        self.log_plot_name = os.path.join(splits[0], 'log_' + splits[1])

        self.metrics = metrics
        self.legend = []
        self.nb_samples = 1
        self.batch_history = {}
        self.epoch_history = {}
        self.batch = []
        self.batch_counter = 0
        self.epoch_counter = 0
        self.save_graphs = save_graphs
        self.plot_logs = plot_logs
        self.calibrated=False
        super(OutputHistory, self).__init__()

    def on_train_begin(self, logs={}):
        if self.calibrated is False:
            super(OutputHistory, self).on_train_begin(logs=logs)
            if 'samples' in self.params.keys():
                self.nb_samples = self.params['samples']
            else:
                self.nb_samples = self.params['steps']

            if self.metrics is None:
                self.metrics = self.params['metrics']
            self.calibrated = True

    def on_train_end(self, logs={}):
        super(OutputHistory, self).on_train_end(logs=logs)
        # self.save_tracked_values()
        # plt.clf()

    def on_epoch_end(self, epoch, logs={}):
        super(OutputHistory, self).on_epoch_end(epoch, logs=logs)
        for k, v in logs.items():
            self.epoch_history.setdefault(k, []).append(v)

        self.save_tracked_values()

        if self.save_graphs:
            self.save_value_plot()

        self.batch_counter = 0
        self.epoch_counter += 1

    def on_batch_end(self, batch, logs={}):
        batch_size = logs.get('size', 0)

        if len(self.epoch) >= 0:
            batch_frac = (self.epoch_counter - 1) + float(self.batch_counter)/float(self.nb_samples)
            self.batch.append(batch_frac)
            for k, v in logs.items():
                self.batch_history.setdefault(k, []).append(v)

        self.batch_counter += batch_size

    def save_tracked_values(self):
        json.dump(self.history, open(self.data_name, 'w'), sort_keys=True, indent=4)

    def plot_tracked_values(self, log_plot=False):

        for t in self.metrics:
            epoch_x = range(1, len(self.epoch_history[t])+1)
            plt.plot(epoch_x, self.epoch_history[t], label=t)
            if t in self.batch_history:
                batch_x = np.arange(len(self.batch_history[t]))/len(self.batch_history[t]) * len(self.epoch_history[t])
                plt.plot(batch_x, self.batch_history[t], color='k', alpha=0.2)
                plt.plot(batch_x, scipy.ndimage.filters.median_filter(self.batch_history[t], [51]), color='k', alpha=0.5)

        self.legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(b=True, which='major', linestyle='--')
        plt.xlabel('epoch')

        if log_plot:
            plt.yscale('log')

    def save_value_plot(self):
        self.plot_tracked_values()
        plt.savefig(self.plot_name, dpi=300, bbox_extra_artists=(self.legend, ), bbox_inches='tight')
        plt.cla()

        if self.plot_logs:
            self.plot_tracked_values(log_plot=True)
            plt.savefig(self.log_plot_name, dpi=300, bbox_extra_artists=(self.legend, ), bbox_inches='tight')
            plt.cla()


class TestImageCallback(Callback):

    def __init__(self, output_name, test_case, test_mask=None, frequency=10):
        super(TestImageCallback, self).__init__()
        self.frequency = frequency
        self.output_name = output_name
        self.test_case = test_case
        self.test_mask = test_mask
        self.nb_saves = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == 0:
            test = self.model.predict(self.test_case)

            if isinstance(test,list):
                test = np.concatenate(test, axis=-3)

            if len(test.shape)==4:

                im = test[0,:,:,1]
                test = im.reshape(self.test_case.shape)
                im /= np.max(im)

                # if self.test_mask is not None:
                #     test = np.concatenate([self.test_mask] + [self.test_case] + test ,axis=-1)
                # else:
                #     test = np.concatenate([self.test_case] + test ,axis=-1)

            else:
                if self.test_mask is not None:
                    test = np.concatenate([self.test_mask] + [self.test_case] + [test], axis=-1)
                else:
                    test = np.concatenate([self.test_case] + [test], axis=-1)

            plt.imsave(self.output_name.format(self.nb_saves), test.reshape(test.shape[1:3]), cmap=plt.get_cmap('viridis'))
            self.nb_saves += 1
