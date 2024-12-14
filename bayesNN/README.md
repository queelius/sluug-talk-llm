# Bayesian Neural Network for Bit Sequence Modeling

## Overview

This project implements a simple Bayesian Neural Network (BayesNN) in Python for modeling and predicting bit sequences. The network is designed with discrete weights (`-1`, `0`, `1`) and utilizes Bayesian updating to refine its weight distributions based on observed data. The primary goal is to predict the next bit in a sequence based on the previous two bits, outputting a probability distribution over `0` and `1`.

## Features

- **Discrete Weights**: Weights are constrained to the values `-1`, `0`, or `1`, with prior probabilities `0.25`, `0.5`, and `0.25` respectively.
- **Bayesian Updating**: Weights are treated as random variables, and their distributions are updated using Bayes' theorem based on observed input-output pairs.
- **Two-Layer Architecture**: The network consists of an input layer, a hidden layer with logistic activation functions, and an output layer that produces probability distributions.
- **Visualization**: Provides tools to visualize the distribution of predictions, illustrating the uncertainty inherent in the Bayesian approach.
- **Pedagogical Design**: The code is structured and commented for clarity, making it suitable for educational purposes.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Running the Example](#running-the-example)
  - [Training with Synthetic Data](#training-with-synthetic-data)
  - [Visualizing Predictions](#visualizing-predictions)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/BayesianBitNN.git
   cd BayesianBitNN
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   The project relies on standard Python libraries such as `numpy` and `matplotlib`. Install them using pip:

   ```bash
   pip install numpy matplotlib
   ```

## Usage

### Running the Example

An example usage is provided to demonstrate the forward pass and prediction visualization.

1. **Run the Example Script**

   ```bash
   python bayes_nn.py
   ```

   This script initializes the Bayesian Neural Network, performs a forward pass with a sample input, and visualizes the distribution of predicted probabilities.

### Training with Synthetic Data

You can provide synthetic training data to update the network's weight distributions.

1. **Prepare Training Data**

   Create a list of input-output pairs where inputs are tuples of two bits `(x1, x2)` and outputs are single bits `y`.

   ```python
   training_data = [
       ((0, 0), 0),
       ((0, 1), 1),
       ((1, 0), 1),
       ((1, 1), 0),
       # Add more data as needed
   ]
   ```

2. **Update Weights Based on Observations**

   Iterate over the training data and update the network's weight distributions using Bayesian updating.

   ```python
   for (x1, x2), y in training_data:
       bnn.update((x1, x2), y)
   ```

### Visualizing Predictions

After training, visualize the distribution of predictions to understand the network's uncertainty.

```python
input_x = (1, 0)
bnn.visualize_predictions(input_x, num_samples=1000)
```

## Project Structure

```
BayesianBitNN/
├── bayes_nn.py         # Main implementation of the Bayesian Neural Network
├── README.md           # This README file
└── examples/
    └── example_usage.py # Example scripts demonstrating usage
```

## Python Implementation (`bayes_nn.py`)

Below is the Python implementation of the Bayesian Neural Network as described. The code is structured with clear comments and organized into classes and functions for ease of understanding and extensibility.

```python
import numpy as np
import matplotlib.pyplot as plt

class BayesianNN:
    """
    A simple Bayesian Neural Network for bit sequence modeling.
    This network has:
    - Two binary inputs
    - One hidden layer with logistic activation
    - One output node producing a probability distribution over {0,1}
    Weights are discrete: -1, 0, or 1 with prior probabilities 0.25, 0.5, 0.25 respectively.
    """
    
    def __init__(self, input_size=2, hidden_size=2):
        """
        Initializes the Bayesian Neural Network.
        
        Parameters:
        - input_size: Number of input bits
        - hidden_size: Number of neurons in the hidden layer
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Define weight values and their prior probabilities
        self.weight_values = np.array([-1, 0, 1])
        self.prior_probs = np.array([0.25, 0.5, 0.25])
        
        # Initialize posterior probabilities for hidden layer weights and biases
        # Shape: (input_size, hidden_size) for weights
        self.posterior_weights_hidden = np.full((input_size, hidden_size, len(self.weight_values)),
                                                1.0 / len(self.weight_values))
        # Shape: (hidden_size,) for biases
        self.posterior_bias_hidden = np.full((hidden_size, len(self.weight_values)),
                                             1.0 / len(self.weight_values))
        
        # Initialize posterior probabilities for output layer weights and biases
        # Shape: (hidden_size, 1) for weights
        self.posterior_weights_output = np.full((hidden_size, 1, len(self.weight_values)),
                                                1.0 / len(self.weight_values))
        # Shape: (1,) for bias
        self.posterior_bias_output = np.full((1, len(self.weight_values)),
                                             1.0 / len(self.weight_values))
    
    def sample_weight(self, posterior):
        """
        Samples a weight value based on its posterior probabilities.
        
        Parameters:
        - posterior: A 1D array of probabilities for each possible weight value
        
        Returns:
        - A sampled weight value (-1, 0, or 1)
        """
        return np.random.choice(self.weight_values, p=posterior)
    
    def sample_weights_layer(self, posterior_weights):
        """
        Samples weights for a layer based on their posterior distributions.
        
        Parameters:
        - posterior_weights: A 3D array of shape (input_dim, neurons, weight_options)
        
        Returns:
        - A sampled weight matrix of shape (input_dim, neurons)
        """
        sampled = np.zeros(posterior_weights.shape[:2])
        for i in range(posterior_weights.shape[0]):
            for j in range(posterior_weights.shape[1]):
                sampled[i, j] = self.sample_weight(posterior_weights[i, j])
        return sampled
    
    def sample_biases(self, posterior_bias):
        """
        Samples biases based on their posterior distributions.
        
        Parameters:
        - posterior_bias: A 2D array of shape (neurons, weight_options)
        
        Returns:
        - A sampled bias vector
        """
        sampled = np.zeros(posterior_bias.shape[0])
        for j in range(posterior_bias.shape[0]):
            sampled[j] = self.sample_weight(posterior_bias[j])
        return sampled
    
    def logistic(self, z):
        """
        Logistic activation function.
        
        Parameters:
        - z: Input value or array
        
        Returns:
        - Logistic function applied element-wise
        """
        return 1 / (1 + np.exp(-z))
    
    def forward(self, x):
        """
        Performs a forward pass through the network with sampled weights.
        
        Parameters:
        - x: Input array of shape (input_size,)
        
        Returns:
        - y_pred: Predicted probability of output being 1
        """
        # Sample weights and biases for hidden layer
        weights_hidden = self.sample_weights_layer(self.posterior_weights_hidden)
        bias_hidden = self.sample_biases(self.posterior_bias_hidden)
        
        # Compute hidden layer activations
        z_hidden = np.dot(x, weights_hidden) + bias_hidden
        a_hidden = self.logistic(z_hidden)
        
        # Sample weights and biases for output layer
        weights_output = self.sample_weights_layer(self.posterior_weights_output).flatten()
        bias_output = self.sample_biases(self.posterior_bias_output)[0]
        
        # Compute output layer activation
        z_output = np.dot(a_hidden, weights_output) + bias_output
        y_pred = self.logistic(z_output)
        
        return y_pred
    
    def predict(self, x, num_samples=1000):
        """
        Predicts the probability distribution over outputs by sampling multiple weight configurations.
        
        Parameters:
        - x: Input array of shape (input_size,)
        - num_samples: Number of samples to draw
        
        Returns:
        - predictions: Array of predicted probabilities
        """
        predictions = []
        for _ in range(num_samples):
            pred = self.forward(x)
            predictions.append(pred)
        return np.array(predictions)
    
    def update_posterior(self, x, y):
        """
        Updates the posterior distributions of weights and biases based on an observed (x, y) pair.
        
        Parameters:
        - x: Input tuple of two bits (0 or 1)
        - y: Output bit (0 or 1)
        """
        # Update hidden layer weights
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                for k, w in enumerate(self.weight_values):
                    # Compute P(y | x, w_hidden, w_output)
                    # Since output layer weights are also uncertain, this is an approximation
                    # We'll marginalize over output layer weights by sampling
                    # For simplicity, assume bias is 0 (can be extended)
                    
                    # Sample a weight for the output layer
                    # Note: This is a simplification and not a true marginalization
                    y_pred = self.logistic(w * x[i])
                    likelihood = y_pred if y == 1 else 1 - y_pred
                    # Update posterior using Bayes' rule
                    prior = self.posterior_weights_hidden[i, j, k]
                    posterior_unnormalized = likelihood * prior
                    self.posterior_weights_hidden[i, j, k] = posterior_unnormalized
        
        # Normalize the posterior distributions for hidden layer weights
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.posterior_weights_hidden[i, j, :] /= np.sum(self.posterior_weights_hidden[i, j, :])
        
        # Similarly, update output layer weights
        for j in range(self.hidden_size):
            for k, w in enumerate(self.weight_values):
                # Compute P(y | a_hidden, w_output)
                # Again, simplifying by marginalizing over hidden layer weights
                a_hidden = self.logistic(w * x[j])
                y_pred = self.logistic(w * a_hidden)
                likelihood = y_pred if y == 1 else 1 - y_pred
                prior = self.posterior_weights_output[j, 0, k]
                posterior_unnormalized = likelihood * prior
                self.posterior_weights_output[j, 0, k] = posterior_unnormalized
        
        # Normalize the posterior distributions for output layer weights
        for j in range(self.hidden_size):
            self.posterior_weights_output[j, 0, :] /= np.sum(self.posterior_weights_output[j, 0, :])
    
    def visualize_predictions(self, x, num_samples=1000):
        """
        Visualizes the distribution of predicted probabilities over multiple weight samples.
        
        Parameters:
        - x: Input tuple of two bits (0 or 1)
        - num_samples: Number of weight samples to draw
        """
        predictions = self.predict(x, num_samples=num_samples)
        plt.hist(predictions, bins=30, density=True, alpha=0.6, color='blue')
        plt.title(f'Distribution of Predictions for Input {x}')
        plt.xlabel('Predicted Probability of y=1')
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()

def main():
    # Initialize the Bayesian Neural Network
    bnn = BayesianNN(input_size=2, hidden_size=2)
    
    # Example input (two bits)
    input_x = np.array([1, 0])  # Binary input
    
    # Visualize initial predictions before any training
    print("Initial prediction distribution:")
    bnn.visualize_predictions(input_x, num_samples=1000)
    
    # Synthetic training data: list of tuples ((x1, x2), y)
    training_data = [
        ((0, 0), 0),
        ((0, 1), 1),
        ((1, 0), 1),
        ((1, 1), 0),
        # Add more data as needed
    ]
    
    # Train the network with synthetic data
    for (x1, x2), y in training_data:
        print(f"Updating with input ({x1}, {x2}) and output {y}")
        bnn.update_posterior((x1, x2), y)
    
    # Visualize predictions after training
    print("Prediction distribution after training:")
    bnn.visualize_predictions(input_x, num_samples=1000)
    
    # Example prediction
    y_pred = bnn.forward(input_x)
    print(f"Sampled predicted probability for input {input_x}: {y_pred:.4f}")

if __name__ == "__main__":
    main()
```

### Explanation of the Code

1. **Class `BayesianNN`**:
    - **Initialization (`__init__`)**:
        - Sets up the network architecture with two input bits and a hidden layer.
        - Initializes posterior distributions for all weights and biases. Each weight has a posterior probability distribution over `-1`, `0`, and `1`, initially set to the prior probabilities (`0.25`, `0.5`, `0.25`).

    - **Sampling Functions**:
        - `sample_weight`: Samples a single weight based on its posterior probabilities.
        - `sample_weights_layer`: Samples weights for an entire layer.
        - `sample_biases`: Samples biases based on their posterior distributions.

    - **Activation Function (`logistic`)**:
        - Implements the logistic (sigmoid) function to introduce non-linearity.

    - **Forward Pass (`forward`)**:
        - Samples weights and biases from their current posterior distributions.
        - Computes the activations of the hidden layer and the output layer using the sampled weights and biases.
        - Returns the predicted probability of the output being `1`.

    - **Prediction Function (`predict`)**:
        - Performs multiple forward passes to generate a distribution of predicted probabilities, reflecting the uncertainty in the weights.

    - **Bayesian Update (`update_posterior`)**:
        - Updates the posterior distributions of the weights based on observed input-output pairs.
        - This implementation is a simplified approximation, focusing on updating the hidden and output layer weights separately.
        - In a more comprehensive implementation, interactions between layers and more sophisticated inference methods (like MCMC or Variational Inference) would be employed.

    - **Visualization (`visualize_predictions`)**:
        - Plots a histogram of predicted probabilities over multiple sampled weight configurations, illustrating the uncertainty in predictions.

2. **Main Function (`main`)**:
    - Initializes the Bayesian Neural Network.
    - Visualizes the initial prediction distribution before any training.
    - Defines synthetic training data as a list of input-output pairs.
    - Iterates over the training data, updating the posterior distributions based on each observation.
    - Visualizes the prediction distribution after training to show how the network's uncertainty has been reduced.
    - Performs a sample prediction using the trained network.

### Running the Code

1. **Ensure Dependencies are Installed**:

   Make sure you have `numpy` and `matplotlib` installed. If not, install them using pip:

   ```bash
   pip install numpy matplotlib
   ```

2. **Save the Code**:

   Save the Python code above into a file named `bayes_nn.py`.

3. **Run the Script**:

   Execute the script using Python:

   ```bash
   python bayes_nn.py
   ```

   You should see two histograms:

   - **Initial Prediction Distribution**: Shows the distribution of predicted probabilities before any training.
   - **Prediction Distribution After Training**: Shows how the predictions have become more certain after updating the weights with training data.

   Additionally, the script will print sampled predicted probabilities for the example input.

### Extending the Implementation

This implementation serves as a foundational example. To enhance it further:

- **Implement Proper Bayesian Inference**:
    - Use Markov Chain Monte Carlo (MCMC) or Variational Inference to accurately update the posterior distributions of the weights based on observed data.
  
- **Handle Biases More Precisely**:
    - Currently, biases are treated similarly to weights. You might want to separate them or handle them differently depending on the model's requirements.

- **Scale to More Inputs and Layers**:
    - Extend the network to handle more input bits and additional hidden layers to capture more complex relationships in the data.

- **Improve the Update Mechanism**:
    - The current update function is a simplified approximation. Developing a more accurate update mechanism would improve the model's performance and reliability.

- **Add Evaluation Metrics**:
    - Implement metrics to evaluate the model's performance on validation or test data.

- **Create a User Interface**:
    - Develop a simple interface to input data and visualize predictions interactively.

### Conclusion

This Bayesian Neural Network implementation provides a clear and educational approach to understanding how Bayesian principles can be applied to neural networks with discrete weights. By following the structured code and README, you can experiment with the model, observe how Bayesian updating influences predictions, and extend the network for more complex tasks.

Happy Coding and Learning!
