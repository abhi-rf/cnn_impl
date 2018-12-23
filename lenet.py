import numpy as np

'''
For conv layer

output_size = 1+ (input_size - field + 2*pad)/stride

'''

def conv_layer(input_array,weight_array,bias_array,filter_size,field_size,stride,pad):

	input_size = input_array.shape[1]
	input_depth = input_array.shape[0]

	output_size = 1 + (input_size - field_size + 2*pad)/stride


	# 2*pad because adding it to starting and ending position 
	padded_size = input_size + 2*pad 
	
	padded_input_array = np.zeros((input_depth,padded_size,padded_size))

	#array to perform operations on. Would be equal to input_array if pad=0
	for depth in range(input_depth):
		#padded_input_array[pad:(padded_size-pad),pad:(padded_size-pad),depth] = input_array[:,:,depth]
		padded_input_array[depth,pad:(padded_size-pad),pad:(padded_size-pad)] = input_array[depth]
		
	
	output_array = np.zeros((filter_size,output_size,output_size))
	
	for filter in range(filter_size):
		for i in range(output_size):
			for j in range(output_size):
				for depth in range(input_depth):
					output_array[filter,i,j] += np.sum(padded_input_array[depth,i*stride:i*stride+field_size,j*stride:j*stride+field_size]*weight_array[filter*3+depth])

				output_array[filter,i,j] += bias_array[filter]




	return (output_array)



def pool_layer(input_array,field_size,stride):

	input_size = input_array.shape[1]
	input_depth = input_array.shape[0]
	output_size = 1 + (input_size - field_size)/stride
	output_depth = input_depth

	output_array = np.zeros((output_depth,output_size,output_size))

	for depth in range(output_depth):
		for i in range(output_size):
			for j in range(output_size):
				output_array[depth,i,j] = np.amax(input_array[depth,i*stride:i*stride+field_size,j*stride:j*stride+field_size])

	return (output_array)



#image_array = np.matrix([[[0,1,2,0,0], [2,1,0,1,0],[2,1,1,1,0],[2,2,2,1,2],[0,1,2,1,2]],[[2,1,1,2,2],[1,0,0,0,0],[0,1,0,2,2],[0,2,2,2,0],[2,0,0,1,0]],[[1,0,1,2,0],[2,0,1,2,0],[1,2,2,2,1],[2,1,0,1,2],[2,0,1,0,2]]])
image_size = 5

img1 = np.matrix([[0,1,2,0,0], [2,1,0,1,0],[2,1,1,1,0],[2,2,2,1,2],[0,1,2,1,2]])
img2 = np.matrix([[2,1,1,2,2],[1,0,0,0,0],[0,1,0,2,2],[0,2,2,2,0],[2,0,0,1,0]])
img3 = np.matrix([[1,0,1,2,0],[2,0,1,2,0],[1,2,2,2,1],[2,1,0,1,2],[2,0,1,0,2]])

image_array = [img1,img2,img3]
image_array = np.asarray(image_array)

#weight_array = np.matrix([[[1,-1,1],[1,1,-1],[0,1,1]],[[0,0,-1],[1,-1,-1],[0,1,-1]],[[0,-1,1],[0,1,0],[0,1,-1]]])

w1 = np.matrix([[1,-1,1],[1,1,-1],[0,1,1]])
w2 = np.matrix([[0,0,-1],[1,-1,-1],[0,1,-1]])
w3 = np.matrix([[0,-1,1],[0,1,0],[0,1,-1]])

w4 = np.matrix([[0,1,1],[-1,1,1],[-1,1,0]])
w5 = np.matrix([[1,-1,0],[-1,-1,-1],[0,1,0]])
w6 = np.matrix([[0,1,0],[0,1,0],[0,0,1]])



weight_array = [w1,w2,w3,w4,w5,w6]
weight_array = np.asarray(weight_array)
#print(weight_array)
bias = [1,0]
conv1 = conv_layer(image_array,weight_array,bias,2,3,2,1)

pool1 = pool_layer(conv1,2,2)

