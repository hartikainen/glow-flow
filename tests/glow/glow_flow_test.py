import numpy as np
import tensorflow as tf

from glow.glow_flow import GlowFlow


tf.enable_eager_execution()

batch_size = 1
event_dims = (5,)

flow = GlowFlow(num_layers=2,
                event_ndims=1,
                event_dims=event_dims,
                validate_args=False)

x = np.random.rand(batch_size, *event_dims)

z = flow.forward(x)
print('z:', z.numpy())

x_ = flow.inverse(z)
print('x_', x_.numpy())

tf.assert_equal(x, x_)
