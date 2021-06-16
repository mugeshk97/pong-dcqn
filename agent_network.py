import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)


def network(input_dims, n_actions):

	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=2, activation='relu',
	                 input_shape=(*input_dims,), data_format='channels_first'))
	model.add(tf.keras.layers.MaxPooling2D( pool_size=2, data_format='channels_first'))
	model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=2, activation='relu',
	                 data_format='channels_first'))
	model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, strides=2, activation='relu',
	                 data_format='channels_first'))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(32, activation='relu'))
	model.add(tf.keras.layers.Dense(n_actions))

	model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
	
	return model