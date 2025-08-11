# This file is responsible for the core logic of the similarity-based chatbot.
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

class SimilarityModel:
    """
    Manages a sentence-transformer model for finding semantically similar sentences.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the model.

        Args:
            model_name (str): The name of the sentence-transformer model to load.
        """
        self.model_name = model_name
        self.model = None
        self.question_embeddings = None
        self.questions = []
        self.answers = []

    def load_model(self):
        """
        Loads the sentence-transformer model from the internet or cache.
        This might take a moment on the first run.
        """
        # Check if a GPU is available and use it, otherwise use CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading sentence transformer model '{self.model_name}' onto device: {device}")
        self.model = SentenceTransformer(self.model_name, device=device)
        print("Model loaded successfully.")

    def build_index(self, qa_pairs: list[tuple[str, str]]):
        """
        Encodes all questions into embeddings and stores them for fast look-up.

        Args:
            qa_pairs (list): A list of (question, answer) tuples.
        """
        if not self.model:
            self.load_model()
            
        if not qa_pairs:
            print("Warning: No Q&A pairs provided to build the index.")
            self.questions = []
            self.answers = []
            self.question_embeddings = None
            return

        self.questions = [q for q, a in qa_pairs]
        self.answers = [a for q, a in qa_pairs]
        
        print(f"Encoding {len(self.questions)} questions into embeddings...")
        # The model's encode function handles batching for efficiency
        self.question_embeddings = self.model.encode(self.questions, convert_to_tensor=True, show_progress_bar=True)
        print("Encoding complete. The index is ready.")

    def find_best_match(self, user_prompt: str) -> tuple[str | None, float]:
        """
        Finds the most similar question in the index to the user's prompt.

        Args:
            user_prompt (str): The user's input question.

        Returns:
            A tuple containing the best matching answer and the similarity score.
            Returns (None, 0.0) if the index is not built.
        """
        if self.question_embeddings is None or len(self.answers) == 0:
            return None, 0.0

        # 1. Encode the user's prompt
        prompt_embedding = self.model.encode(user_prompt, convert_to_tensor=True)

        # 2. Use sentence-transformers utility to compute cosine similarity
        # This is highly optimized and much faster than manual calculation.
        hits = util.semantic_search(prompt_embedding, self.question_embeddings, top_k=1)
        
        # hits is a list of lists, one for each prompt. Since we have one prompt, we take the first list.
        if not hits or not hits[0]:
            return None, 0.0

        best_hit = hits[0][0]
        score = best_hit['score']
        best_match_index = best_hit['corpus_id']
        
        # 3. Return the corresponding answer
        return self.answers[best_match_index], score
    

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers=[32, 32], output_size=1, learning_rate=0.001, dropout_rate=0.5):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.hidden_layers = hidden_layers
        
        self.weights = []
        self.biases = []
        self.adam_m = []
        self.adam_v = []

        layer_input_size = input_size
        for layer_size in hidden_layers:
            self.weights.append(np.random.randn(layer_input_size, layer_size) * np.sqrt(2. / layer_input_size))
            self.biases.append(np.zeros((1, layer_size)))
            layer_input_size = layer_size
        self.weights.append(np.random.randn(layer_input_size, output_size) * np.sqrt(2. / layer_input_size))
        self.biases.append(np.zeros((1, output_size)))

        for w, b in zip(self.weights, self.biases):
            self.adam_m.append([np.zeros_like(w), np.zeros_like(b)])
            self.adam_v.append([np.zeros_like(w), np.zeros_like(b)])

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # Subtract max for numerical stability
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def forward(self, X, training=True):
        self.activations = [X]
        self.z_values = []
        self.dropout_masks = []

        activation = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activation, w) + b
            self.z_values.append(z)
            if i < len(self.weights) - 1: # ReLU for hidden layers
                activation = self.relu(z)
                if training and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1 - self.dropout_rate, size=activation.shape) / (1 - self.dropout_rate)
                    activation *= mask
                    self.dropout_masks.append(mask)
            else: # Softmax for output layer
                activation = self.softmax(z)
            self.activations.append(activation)
        return self.activations[-1]

    def backward(self, y, output):
        m = y.shape[0]
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # Output layer
        dz = (output - y) * self.sigmoid_derivative(self.z_values[-1])
        grads_w[-1] = np.dot(self.activations[-2].T, dz) / m
        grads_b[-1] = np.sum(dz, axis=0, keepdims=True) / m

        # Hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            dz = np.dot(dz, self.weights[i+1].T) * self.relu_derivative(self.z_values[i])
            if self.dropout_rate > 0:
                dz *= self.dropout_masks[i]
            grads_w[i] = np.dot(self.activations[i].T, dz) / m
            grads_b[i] = np.sum(dz, axis=0, keepdims=True) / m
        
        return grads_w, grads_b

    def update_with_adam(self, t, grads_w, grads_b):
        for i in range(len(self.weights)):
            # Update weights
            self.adam_m[i][0] = self.beta1 * self.adam_m[i][0] + (1 - self.beta1) * grads_w[i]
            self.adam_v[i][0] = self.beta2 * self.adam_v[i][0] + (1 - self.beta2) * (grads_w[i]**2)
            m_hat = self.adam_m[i][0] / (1 - self.beta1**t)
            v_hat = self.adam_v[i][0] / (1 - self.beta2**t)
            self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            # Update biases
            self.adam_m[i][1] = self.beta1 * self.adam_m[i][1] + (1 - self.beta1) * grads_b[i]
            self.adam_v[i][1] = self.beta2 * self.adam_v[i][1] + (1 - self.beta2) * (grads_b[i]**2)
            m_hat = self.adam_m[i][1] / (1 - self.beta1**t)
            v_hat = self.adam_v[i][1] / (1 - self.beta2**t)
            self.biases[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def train(self, X, y, training_iterations=2000, validation_split=0.1, patience=10, progress_callback=None):
        val_size = int(len(X) * validation_split) if validation_split > 0 else 0
        X_train, X_val = (X[:-val_size], X[-val_size:]) if val_size > 0 else (X, [])
        y_train, y_val = (y[:-val_size], y[-val_size:]) if val_size > 0 else (y, [])

        y_train = y_train.reshape(-1, self.weights[-1].shape[1])
        if len(y_val) > 0:
            y_val = y_val.reshape(-1, self.weights[-1].shape[1])

        best_val_error = float('inf')
        patience_counter = 0
        train_errors, val_errors, accuracies = [], [], []

        for i in range(training_iterations):
            output = self.forward(X_train, training=True)
            grads_w, grads_b = self.backward(y_train, output)
            self.update_with_adam(i + 1, grads_w, grads_b)
            
            if i % 100 == 0:
                train_error = np.mean((output - y_train) ** 2)
                train_errors.append(train_error)
                
                if len(X_val) > 0:
                    val_output = self.forward(X_val, training=False)
                    val_error = np.mean((val_output - y_val) ** 2)
                    val_errors.append(val_error)
                    
                    similarity = np.mean([np.dot(val_output[j], y_val[j]) / (np.linalg.norm(val_output[j]) * np.linalg.norm(y_val[j])) for j in range(len(y_val)) if np.linalg.norm(val_output[j]) > 0 and np.linalg.norm(y_val[j]) > 0])
                    accuracies.append(similarity)

                    if progress_callback:
                        progress_callback(i, train_error, val_error, similarity)
                    
                    if val_error < best_val_error:
                        best_val_error = val_error
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at iteration {i}.")
                        break
                else:
                    if progress_callback:
                        progress_callback(i, train_error, None, None)

        return train_errors, val_errors, accuracies

    def predict(self, X, temperature=1.0):
        activation = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activation, w) + b
            if i < len(self.weights) - 1:
                activation = self.relu(z)               
            else:
                if temperature <= 0:
                    z = np.clip(z, -500, 500)
                    activation = self.relu_derivative(z)
                else:
                    z = np.clip(z, -500 / temperature, 500 / temperature)
                    activation = self.relu(z)
                if temperature > 0:
                    z /= temperature
                activation = self.softmax(z)
                
        return activation
