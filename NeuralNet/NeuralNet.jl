#=
NeuralNet:
- Julia version: 1.1.0
- Author: endoumasafumi
- Date: 2019-12-14
=#

using LinearAlgebra

module NeuralNet

    # Function as a constructor
    function init(node_input, node_hidden, node_output)
        # Initialize variables
        global weight_hidden = rand(node_hidden, node_input+1)
        global weight_output = rand(node_output, node_hidden+1)
        global momentum_hidden = zeros(node_hidden, node_input+1)
        global momentum_output = zeros(node_output, node_hidden+1)
    end

    # Public function
    # Function for training neural network
    function train(X_train, T_train, X_valid, T_valid, batch_size, lr, mu, epoch)

        error_train = zeros(epoch)
        error_valid = zeros(epoch)

        # Loop for updating weights
        for i in 1:epoch

            # Shuffle data with keeping relationship
            idx_train = shuffle(length(X_train))
            X_train = X_train[idx_train]
            T_train = T_train[idx_train]
            idx_valid = shuffle(length(X_valid))
            X_valid = X_valid[idx_valid]
            T_valid = T_valid[idx_valid]

            # Loop for batch-sized data
            for j = 1:batch_size
                x = X_train[j, :]
                t = T_train[j, :]

                # Update weights by back-propagation
                weight_output, weight_hidden = __backprop(x, t, lr, mu)
            end
            # Keep errors for drawing error history
            error_train[i] = __calc_error(X_train, T_train)
            error_valid[i] = __calc_error(X_valid, T_valid)
        end
        return weight_output, weight_hidden, error_train, error_valid
    end

    # Function for prediction
    function predict(X, T, weight_output, weight_hidden)

        # Initialize each variable
        N = size(X, 1)
        C = zeros(N)
        Y = zeros(N, size(X, 2))
        _T = zeros(N)

        # Loop for computing output (z: output from hidden layer, y: output from output layer)
        for i in 1:N

            x = X[i, :]
            z = __sigmoid(dot(weight_hidden, x))
            y = __sigmoid(dot(weight_output, z))

            t = T[i, :]
            if t[1] == 0
                _T[i] = 1
            elseif t[1] == 1
                _T[i] = 0
            end

            Y[i] = y
            C[i] = argmax(y)
        end

        # Loop for computing accuracy
        for j in 1:N
            if C[j] == _T[j]
                TPN += 1
            end
        end

        # Compute accuracy
        acc = TPN / N
        return acc
    end

    function __sigmoid(input)
        sigmoid(x) = 1.0 / (1.0 + exp(-x))
        output = sigmoid(input)
        return output
    end
    
    function __forward(x)
        # Compute output (z: output from hidden layer, y: output from output layer)
        z = __sigmoid(dot(weight_hidden, x))
        y = __sigmoid(dot(weight_output, z))
        return z, y
    end
    
    function __backprop()
        # Compute output
        
    end
    
    function __calc_error()
        
    end

end