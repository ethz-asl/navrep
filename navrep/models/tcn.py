from __future__ import print_function
import numpy as np
import tensorflow as tf
from collections import namedtuple
import json
import pickle

HyperParams = namedtuple(
    "HyperParams",
    [
        "max_seq_len",
        "seq_width",
        "h_size",
        "batch_size",
        "grad_clip",
        "num_mixture",
        "restart_factor",
        "learning_rate",
        "decay_rate",
        "min_learning_rate",
        "use_layer_norm",
        "dropout_prob",
        "is_training",
        "differential_z",
        "layer_channels",
        "layer_dilations",
        "kernel_size",
    ],
)


def default_hps():
    return HyperParams(
        max_seq_len=1000,  # train on sequences of 1000
        seq_width=32,  # width of our data (32)
        h_size=512,  # number of rnn cells
        batch_size=100,  # minibatch sizes
        grad_clip=1.0,
        num_mixture=5,  # number of mixtures in MDN
        restart_factor=10.0,  # factor of importance for restart=1 rare case for loss.
        learning_rate=0.001,
        decay_rate=0.99999,
        min_learning_rate=0.00001,
        use_layer_norm=0,  # set this to 1 to get more stable results (less chance of NaN), but slower
        dropout_prob=0.05,
        is_training=1,
        differential_z=1,
        layer_channels=[32, 32, 32, 512], # [32, 32, 32, 32, 32, 32, 32, 512],  # last channel size is h_size
        layer_dilations=[1, 2, 4, 8], # [1, 2, 4, 8, 16, 32, 64, 128],
        kernel_size=3, # 7,
    )


default_hps_params = default_hps()
sample_hps_params = default_hps_params._replace(
    batch_size=1, is_training=0
)


def reset_graph():
    if "sess" in globals() and sess:  # noqa
        sess.close()  # noqa
    tf.reset_default_graph()


# MDN-TCN model tailored for doomrnn
class MDNTCN:
    def __init__(self, hps, gpu_mode=True, reuse=False):
        self.hps = hps
        if not gpu_mode:
            with tf.device("/cpu:0"):
                print("model using cpu")
                self.g = tf.Graph()
                with self.g.as_default():
                    self.build_model(hps)
        else:
            print("model using gpu")
            self.g = tf.Graph()
            with self.g.as_default():
                self.build_model(hps)
        self.init_session()

    def build_model(self, hps):

        self.num_mixture = hps.num_mixture
        KMIX = self.num_mixture  # 5 mixtures
        ACTION_WIDTH = 3
        LENGTH = self.hps.max_seq_len - 1  # 999 timesteps

        if hps.is_training:
            self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.sequence_lengths = LENGTH  # assume every sample has same length.
        self.batch_z = tf.placeholder(
            dtype=tf.float32,
            shape=[self.hps.batch_size, self.hps.max_seq_len, hps.seq_width],
        )
        self.batch_action = tf.placeholder(
            dtype=tf.float32,
            shape=[self.hps.batch_size, self.hps.max_seq_len, ACTION_WIDTH],
        )
        self.batch_restart = tf.placeholder(
            dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len]
        )
        # TODO: add latent goal info

        self.input_z = self.batch_z[:, :LENGTH, :]
        self.input_action = self.batch_action[:, :LENGTH, :]
        self.input_restart = self.batch_restart[:, :LENGTH]

        if hps.differential_z:
            self.target_z = (
                self.batch_z[:, 1:, :] - self.batch_z[:, :LENGTH, :]
            )  # delta Z prediction
        else:
            self.target_z = self.batch_z[:, 1:, :]
        self.target_restart = self.batch_restart[:, 1:]

        self.input_seq = tf.concat(
            [
                self.input_z,
                self.input_action,
                tf.reshape(self.input_restart, [self.hps.batch_size, LENGTH, 1]),
            ],
            axis=2,
        )

        # OUTPUT = TCN(INPUT) --------------------------------------------------------
        with tf.variable_scope("MDNTCN"):
            x = self.input_seq  # [batch, seq_length, 32 + 3 + 1]

            padding = "causal"
            dropout_rate = self.hps.dropout_prob
            kernel_size = self.hps.kernel_size
            layer_channels = self.hps.layer_channels
            layer_dilations = self.hps.layer_dilations
            for nb_filters, dilation_rate in zip(layer_channels, layer_dilations):
                init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

                # block1
                conv1 = tf.keras.layers.Conv1D(
                    filters=nb_filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    padding=padding,
                    kernel_initializer=init,
                )
                batch1 = tf.keras.layers.BatchNormalization(axis=-1)
                ac1 = tf.keras.layers.Activation("relu")
                drop1 = tf.keras.layers.Dropout(rate=dropout_rate)

                # block2
                conv2 = tf.keras.layers.Conv1D(
                    filters=nb_filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    padding=padding,
                    kernel_initializer=init,
                )
                batch2 = tf.keras.layers.BatchNormalization(axis=-1)
                ac2 = tf.keras.layers.Activation("relu")
                drop2 = tf.keras.layers.Dropout(rate=dropout_rate)

                downsample = tf.keras.layers.Conv1D(
                    filters=nb_filters,
                    kernel_size=1,
                    padding="same",
                    kernel_initializer=init,
                )
                ac3 = tf.keras.layers.Activation("relu")

                prev_x = x
                x = conv1(x)
                x = batch1(x)
                x = ac1(x)
                x = drop1(x) if self.hps.is_training else x

                x = conv2(x)
                x = batch2(x)
                x = ac2(x)
                x = drop2(x) if self.hps.is_training else x

                if prev_x.shape[-1] != x.shape[-1]:  # match the dimention
                    prev_x = downsample(prev_x)
                assert prev_x.shape == x.shape

                x = ac3(prev_x + x)  # skip connection

            output = x
        # ----------------------------------------------------------------------------
        # output = h = [batch, length, hps.h_size]
        self.h = output

        output = tf.reshape(output, [-1, self.hps.h_size])
        # now [batch * seq_length, hps.h_size]

        NOUT = hps.seq_width * KMIX * 3 + 1  # plus 1 to predict the restart state.

        with tf.variable_scope("MDNTCN"):
            output_w = tf.get_variable("output_w", [self.hps.h_size, NOUT])
            output_b = tf.get_variable("output_b", [NOUT])

        output = tf.reshape(output, [-1, hps.h_size])
        output = tf.nn.xw_plus_b(output, output_w, output_b)
        # now [batch * seq_length, 32 * KMIX * 3 + 1]

        self.out_restart_logits = output[:, 0]  # [batch * seq_length,]
        output = output[:, 1:]

        output = tf.reshape(output, [-1, KMIX * 3])
        # now [batch * seq_length * 32, KMIX * 3]

        logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))

        def tf_lognormal(y, mean, logstd):
            return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

        def get_lossfunc(logmix, mean, logstd, y):
            v = logmix + tf_lognormal(y, mean, logstd)
            v = tf.reduce_logsumexp(v, 1, keepdims=True)
            return -tf.reduce_mean(v)

        def get_mdn_coef(output):
            logmix, mean, logstd = tf.split(output, 3, 1)
            logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
            return logmix, mean, logstd

        out_logmix, out_mean, out_logstd = get_mdn_coef(output)
        # each [batch * seq_length * 32, 1]

        self.out_logmix = out_logmix
        self.out_mean = out_mean
        self.out_logstd = out_logstd

        # reshape target data so that it is compatible with prediction shape
        flat_target_data = tf.reshape(self.target_z, [-1, 1])

        lossfunc = get_lossfunc(out_logmix, out_mean, out_logstd, flat_target_data)

        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(out_mean - flat_target_data)))
        self.z_cost = tf.reduce_mean(lossfunc)

        flat_target_restart = tf.reshape(self.target_restart, [-1, 1])

        self.r_cost = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=flat_target_restart,
            logits=tf.reshape(self.out_restart_logits, [-1, 1]),
        )

        factor = tf.ones_like(self.r_cost) + flat_target_restart * (
            self.hps.restart_factor - 1.0
        )

        self.r_cost = tf.reduce_mean(tf.multiply(factor, self.r_cost))

        #         self.cost = self.rmse + self.r_cost
        self.cost = self.z_cost + self.r_cost

        if self.hps.is_training == 1:
            self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)

            gvs = optimizer.compute_gradients(self.cost)
            capped_gvs = [
                (tf.clip_by_value(grad, -self.hps.grad_clip, self.hps.grad_clip), var)
                for grad, var in gvs
            ]
            self.train_op = optimizer.apply_gradients(
                capped_gvs, global_step=self.global_step, name="train_step"
            )

        # initialize vars
        self.init = tf.global_variables_initializer()

        t_vars = tf.trainable_variables()
        self.assign_ops = {}
        for var in t_vars:
            if var.name.startswith("MDNTCN"):
                pshape = var.get_shape()
                pl = tf.placeholder(tf.float32, pshape, var.name[:-2] + "_placeholder")
                assign_op = var.assign(pl)
                self.assign_ops[var] = (assign_op, pl)

    def init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                if var.name.startswith("MDNTCN"):
                    param_name = var.name
                    p = self.sess.run(var)
                    model_names.append(param_name)
                    params = np.round(p * 10000).astype(np.int).tolist()
                    model_params.append(params)
                    model_shapes.append(p.shape)
        return model_params, model_shapes, model_names

    def set_model_params(self, params):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                if var.name.startswith("MDNTCN"):
                    pshape = tuple(var.get_shape().as_list())
                    p = np.array(params[idx])
                    assert pshape == p.shape, "inconsistent shape"
                    assign_op, pl = self.assign_ops[var]
                    self.sess.run(assign_op, feed_dict={pl.name: p / 10000.0})
                    idx += 1

    def get_random_model_params(self, stdev=0.5):
        # get random params.
        _, mshape, _ = self.get_model_params()
        rparam = []
        for s in mshape:
            # rparam.append(np.random.randn(*s)*stdev)
            rparam.append(np.random.standard_cauchy(s) * stdev)  # spice things up!
        return rparam

    def set_random_params(self, stdev=0.5):
        rparam = self.get_random_model_params(stdev)
        self.set_model_params(rparam)

    def load_json(self, jsonfile="tcn.json"):
        with open(jsonfile, "r") as f:
            params = json.load(f)
        self.set_model_params(params)

    def save_json(self, jsonfile="tcn.json"):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, "wt") as outfile:
            json.dump(
                qparams, outfile, sort_keys=True, indent=0, separators=(",", ": ")
            )
        with open(jsonfile.strip(".json")+".hyperparams.pckl", "wb") as f:
            pickle.dump(self.hps, f)
        print("{} written.".format(jsonfile))


def get_pi_idx(x, pdf):
    # samples from a categorial distribution
    N = pdf.size
    accumulate = 0
    for i in range(0, N):
        accumulate += pdf[i]
        if accumulate >= x:
            return i
    random_value = np.random.randint(N)
    # print('error with sampling ensemble, returning random', random_value)
    return random_value


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
