
class AtomicNetwork:
    def __init__(self, num_inputs, num_hidden):  # Eqn 1 is being coded out here
        
        self.input_to_hidden_weights=[
            [random.uniform(-0.1, 0.1) for i in range(num_inputs)]
            for i in range(num_hidden)
        ]
        
        self.hidden_biases = [0.0 for i in range(num_hidden)]
        self.hidden_to_output_weights = [
            random.uniform(-0.1, 0.1) for i in range(num_hidden)
        ]
        self.output_bias = 0.0

    def forward(self, symmetry_vector):
        hidden_linear = []
        hidden_activation = []
        for j in range(len(self.input_to_hidden_weights)):
            
            linear_sum = sum(w*x for w,x in zip(self.input_to_hidden_weights[j], symmetry_vector))+self.hidden_biases[j]
            
            hidden_linear.append(linear_sum)
            
            hidden_activation.append(math.tanh(linear_sum))
        
        
        atomic_energy = sum(w*h for w,h in zip(self.hidden_to_output_weights, hidden_activation))+self.output_bias
        
        cache = {
            "inputs":symmetry_vector,
            "hidden_linear":hidden_linear,
            "hidden_activation":hidden_activation,
            "output":atomic_energy
        }
        
        return atomic_energy,cache

    def backward_and_update(self, cache, dL_d_output, learning_rate): # Eqn 14
        # This is basically coding out all the four derivative equations.
        hidden_activation = cache["hidden_activation"]
        dW_output = [dL_d_output * h for h in hidden_activation]
        db_output = dL_d_output
        dL_d_hidden_activation = [dL_d_output * w for w in self.hidden_to_output_weights]
        dL_d_hidden_linear = [dh * (1.0 - (h * h)) for dh, h in zip(dL_d_hidden_activation, hidden_activation)]
        inputs = cache["inputs"]
        dW_input = [[dz * x for x in inputs] for dz in dL_d_hidden_linear]
        db_hidden = dL_d_hidden_linear
        
        for j in range(len(self.hidden_to_output_weights)):
            self.hidden_to_output_weights[j] -= learning_rate * dW_output[j]
        self.output_bias -= learning_rate * db_output
        for j in range(len(self.input_to_hidden_weights)):
            for i in range(len(self.input_to_hidden_weights[j])):
                self.input_to_hidden_weights[j][i] -= learning_rate * dW_input[j][i]
            self.hidden_biases[j] -= learning_rate * db_hidden[j]
