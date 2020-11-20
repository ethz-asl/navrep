import json
import numpy as np
import os
import tensorflow as tf


def reset_graph():
    if "sess" in globals() and sess:  # noqa
        sess.close()  # noqa
    tf.reset_default_graph()


class SIREN1D(object):
    def __init__(
        self,
        z_size=32,
        batch_size=1,
        learning_rate=0.0001,
        kl_tolerance=0.5,
        is_training=False,
        reuse=False,
    ):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.kl_tolerance = kl_tolerance
        self.reuse = reuse
        with tf.variable_scope("conv_vae", reuse=self.reuse):
            self._build_graph()
        self._init_session()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():

            N_RAYS = 1080
            self.o = tf.placeholder(tf.float32, shape=[None, N_RAYS, 1])
            self.x = tf.placeholder(tf.float32, shape=[None, 1])
            # TODO: 3 channels with different features?

            # missing:
            # encode o -> z (256)
            # bake z -> theta (256*1 + 256 + 256*256 + 256 + 256*256 + 256 + 256*256 + 256 + 256*1 + 1) (WTF?)
            # use theta instead of weights.

            # encode o -> z
            h = tf.layers.conv1d(
                self.o, 32, 8, strides=4, activation=tf.nn.relu, name="enc_conv1"
            )
            h = tf.layers.conv1d(
                h, 64, 9, strides=4, activation=tf.nn.relu, name="enc_conv2"
            )
            h = tf.layers.conv1d(
                h, 128, 6, strides=4, activation=tf.nn.relu, name="enc_conv3"
            )
            h = tf.layers.conv1d(
                h, 256, 4, strides=4, activation=tf.nn.relu, name="enc_conv4"
            )
            h = tf.reshape(h, [-1, 4 * 256])
            self.mu = tf.layers.dense(h, self.z_size, name="enc_fc_mu")
            self.logvar = tf.layers.dense(h, self.z_size, name="enc_fc_log_var")
            self.sigma = tf.exp(self.logvar / 2.0)
            self.epsilon = tf.random_normal([self.batch_size, self.z_size])
#             if self.is_training:
#                 self.z = self.mu + self.sigma * self.epsilon
#             else:
#                 self.z = self.mu
            self.z = self.mu + self.sigma * self.epsilon

            # bake z -> theta
            # w1     + b1  + w2      + b2  + w3      + b3  + w4      + b4  + w5    + b5
            # (256*1 + 256 + 256*256 + 256 + 256*256 + 256 + 256*256 + 256 + 256*1 + 1)
            h = tf.layers.dense(self.z, 256)  # TODO set to RELu
            self.theta = tf.layers.dense(h, 198145)

            self.params = tf.split(
                self.theta,
                [256*1, 256, 256*256, 256, 256*256, 256, 256*256, 256, 256*1, 1],
                axis=1,
            )
            self.params = [tf.reshape(p, (self.batch_size,) + shape) for p, shape in zip(
                self.params,
                [(1, 256), (256,), (256, 256), (256,), (256, 256), (256,), (256, 256), (256,), (256,1), (1,)]
            )]

            self.params = [[self.params[i], self.params[i+1]] for i in range(0, len(self.params), 2)]

            # 5 SIREN layers
            y = self.x
            for weights, biases in self.params:
                y = tf.reshape(y, (self.batch_size, 1, y.shape[-1]))  # add middle dim for matmul
                y = tf.math.sin(tf.reshape(
                    tf.matmul(y, weights), (self.batch_size, weights.shape[-1])) + biases)
            self.y = y  # (batch, 1)

            # train ops
            if self.is_training:
                self.y_true = tf.placeholder(tf.float32, shape=[None, 1])
                self.global_step = tf.Variable(0, name="global_step", trainable=False)

                # reconstruction loss
                self.r_loss = tf.reduce_sum(
                    tf.square(self.y_true - self.y), reduction_indices=[1]
                )
                self.r_loss = tf.reduce_mean(self.r_loss)

                # augmented kl loss per dim
                self.kl_loss = -0.5 * tf.reduce_sum(
                    (1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)),
                    reduction_indices=1,
                )
                self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
                self.kl_loss = tf.reduce_mean(self.kl_loss)

                self.loss = self.r_loss + self.kl_loss

                # training
                self.lr = tf.Variable(self.learning_rate, trainable=False)
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                grads = self.optimizer.compute_gradients(self.loss)  # can potentially clip gradients here.

                self.train_op = self.optimizer.apply_gradients(
                    grads, global_step=self.global_step, name="train_step"
                )

            # initialize vars
            self.init = tf.global_variables_initializer()

            t_vars = tf.trainable_variables()
            self.assign_ops = {}
            for var in t_vars:
                # if var.name.startswith('conv_vae'):
                pshape = var.get_shape()
                pl = tf.placeholder(tf.float32, pshape, var.name[:-2] + "_placeholder")
                assign_op = var.assign(pl)
                self.assign_ops[var] = (assign_op, pl)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def encode(self, obs):
        """ obs: (n_scans, 1080, 1) normalized 0-1
        output: (n_scans, 32)  -inf to inf
        """
        n_scans, _, _ = obs.shape
        z = self.sess.run(self.z, feed_dict={self.o: obs})
        return z

    def encode_mu_logvar(self, obs):
        """ obs: (n_scans, 1080, 1) normalized 0-1 """
        (mu, logvar) = self.sess.run([self.mu, self.logvar], feed_dict={self.o: obs})
        return mu, logvar

    def decode(self, z):
        """ z: (n_scans, 32) """
        n_scans, _z = z.shape
        z = np.repeat(z.reshape(n_scans, 1, _z), 1080, axis=1).reshape(n_scans*1080, _z)
        # TODO divide X by 1080
        x = np.repeat(np.arange(1080).reshape(1, 1080, 1), n_scans, axis=0).reshape(n_scans*1080, 1)
        y_pred = self.sess.run(self.y, feed_dict={self.x: x, self.z: z})
        return y_pred.reshape(n_scans, 1080, 1)

    def encode_decode(self, obs, PER_POINT_SAMPLING=True):
        """ obs: (n_scans, 1080, 1) normalized 0-1
        output: (n_scans, 1080, 1) 0-1
        """
        n_scans, _, _ = obs.shape
        if self.is_training:
            raise NotImplementedError("encode_decode() with is_training true is disabled.")
        if PER_POINT_SAMPLING:
            z = self.encode(np.repeat(obs, 1080, axis=0))  # different z sample for every point
            # TODO divide X by 1080
            x = np.repeat(np.arange(1080).reshape(1, 1080, 1), n_scans, axis=0).reshape(n_scans*1080, 1)
            y_pred = self.sess.run(self.y, feed_dict={self.x: x, self.z: z})
        else:
            z = self.encode(obs)
            y_pred = self.decode(z)
        return y_pred.reshape(n_scans, 1080, 1)

    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            for var in t_vars:
                # if var.name.startswith('conv_vae'):
                param_name = var.name
                p = self.sess.run(var)
                model_names.append(param_name)
                params = np.round(p * 10000).astype(np.int).tolist()
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names

    def get_random_model_params(self, stdev=0.5):
        # get random params.
        _, mshape, _ = self.get_model_params()
        rparam = []
        for s in mshape:
            # rparam.append(np.random.randn(*s)*stdev)
            rparam.append(np.random.standard_cauchy(s) * stdev)  # spice things up
        return rparam

    def set_model_params(self, params):
        with self.g.as_default():
            t_vars = tf.trainable_variables()
            idx = 0
            for var in t_vars:
                # if var.name.startswith('conv_vae'):
                pshape = tuple(var.get_shape().as_list())
                p = np.array(params[idx])
                assert pshape == p.shape, "inconsistent shape"
                assign_op, pl = self.assign_ops[var]
                self.sess.run(assign_op, feed_dict={pl.name: p / 10000.0})
                idx += 1

    def load_json(self, jsonfile="vae.json"):
        with open(jsonfile, "r") as f:
            params = json.load(f)
        self.set_model_params(params)

    def save_json(self, jsonfile="vae.json"):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, "wt") as outfile:
            json.dump(
                qparams, outfile, sort_keys=True, indent=0, separators=(",", ": ")
            )

    def set_random_params(self, stdev=0.5):
        rparam = self.get_random_model_params(stdev)
        self.set_model_params(rparam)

    def save_model(self, model_save_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join(model_save_path, "vae")
        tf.logging.info("saving model %s.", checkpoint_path)
        saver.save(sess, checkpoint_path, 0)  # just keep one

    def load_checkpoint(self, checkpoint_path):
        sess = self.sess
        with self.g.as_default():
            saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print("loading model", ckpt.model_checkpoint_path)
        tf.logging.info("Loading model %s.", ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
