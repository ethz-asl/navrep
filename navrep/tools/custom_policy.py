import tensorflow as tf

from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc

_Z = 32
_RS = 5
_64 = 64
_C = 64  # controller FC layer size
ARCH = "VCARCH"

# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor
class Custom1DPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(Custom1DPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            x = self.processed_obs[:, :-_RS]
            state_features = tf.reshape(self.processed_obs[:, -_RS:], (-1, _RS))

            h = tf.layers.conv1d(
                x, 32, 8, strides=4, activation=tf.nn.relu, name="enc_conv1"
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
            h = tf.layers.dense(h, _Z, name="enc_fc_mu")
            extracted_features = tf.concat([h, state_features], axis=1)

            pi_h = extracted_features
            for i, layer_size in enumerate([_C, _C]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = extracted_features
            for i, layer_size in enumerate([_C, _C]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            x = self.processed_obs[:, :-_RS]
            x = tf.reshape(x, (-1, _64, _64, 1))
            state_features = tf.reshape(self.processed_obs[:, -_RS:], (-1, _RS))

            # Encoder
            h = tf.layers.conv2d(
                x, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1"
            )
            h = tf.layers.conv2d(
                h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2"
            )
            h = tf.layers.conv2d(
                h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3"
            )
            h = tf.layers.conv2d(
                h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4"
            )
            h = tf.reshape(h, [-1, 2 * 2 * 256])
            h = tf.layers.dense(h, _Z, name="enc_fc_mu")
            extracted_features = tf.concat([h, state_features], axis=1)

            pi_h = extracted_features
            for i, layer_size in enumerate([_C, _C]):
                pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            pi_latent = pi_h

            vf_h = extracted_features
            for i, layer_size in enumerate([_C, _C]):
                vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            value_fn = tf.layers.dense(vf_h, 1, name='vf')
            vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
