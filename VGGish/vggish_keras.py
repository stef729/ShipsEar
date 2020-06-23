from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
import vggish_params

def get_vggish_keras():
	input_shape = (vggish_params.NUM_FRAMES,vggish_params.NUM_BANDS,1)

	img_input = Input( shape=input_shape)
	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1')(img_input)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)

	# Block fc
	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1_1')(x)
	x = Dense(4096, activation='relu', name='fc1_2')(x)
	x = Dense(vggish_params.EMBEDDING_SIZE, activation='relu', name='fc2')(x)


	model = Model(img_input, x, name='vggish')
	return model
