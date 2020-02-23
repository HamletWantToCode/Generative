export mnist_image

using Flux.Data.MNIST

function mnist_image(num_train, batch_size, num_test)
	X = float.(hcat(vec.(MNIST.images())...)) .> 0.5
	N = size(X, 2)
	I_train = rand(1:N, num_train); train_X = X[:, I_train]
	I_remain = [i for i in 1:N if !(i∈I_train)]
	I_test = rand(I_remain, num_test); test_X = X[:, I_test]
	batch_data = [train_X[:,i] for i in Iterators.partition(1:num_train, batch_size)]
	batch_data, test_X
end


