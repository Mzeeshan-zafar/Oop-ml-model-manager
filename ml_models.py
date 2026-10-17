"""
OOP ML Model Manager
Author: [Your Name]
Date: [Today's Date]
Description: Demonstrates OOP concepts applied to ML models.
"""

# ============================================================
# PARENT CLASS: MachineLearningModel
# ============================================================

class MachineLearningModel:
    """
    Parent class for all ML models.
    Demonstrates: Class variables, instance variables, encapsulation.
    """
    
    # CLASS VARIABLE - shared by all models
    framework = "Scikit-learn"
    
    def __init__(self, name, accuracy, training_time):
        """
        Constructor: Initializes a new ML model.
        
        Args:
            name (str): Model name
            accuracy (float): Model accuracy (0-1)
            training_time (float): Training time in seconds
        """
        # INSTANCE VARIABLES - unique to each model
        self.name = name
        self.accuracy = accuracy
        self.training_time = training_time
        
        # PRIVATE VARIABLE - cannot be accessed directly
        self.__model_version = "1.0.0"
    
    # ---------- PUBLIC METHODS ----------
    
    def train(self):
        """Train the model. Overridden by child classes."""
        print(f"[TRAIN] Training {self.name}...")
        print(f"  Framework: {self.framework}")
        print(f"  Status: Training complete")
    
    def evaluate(self):
        """Evaluate the model and return accuracy."""
        print(f"[EVAL] {self.name} Accuracy: {self.accuracy * 100:.2f}%")
        return self.accuracy
    
    # ---------- GETTER & SETTER (Encapsulation) ----------
    
    def get_version(self):
        """Getter for private variable __model_version."""
        return self.__model_version
    
    def set_version(self, new_version):
        """Setter for private variable with validation."""
        if isinstance(new_version, str) and len(new_version) > 0:
            self.__model_version = new_version
            print(f"[VERSION] Updated to {new_version}")
        else:
            print("[ERROR] Invalid version format")
    
    # ---------- MAGIC METHOD ----------
    
    def __str__(self):
        """String representation of the model."""
        return (f"{self.__class__.__name__}(name={self.name}, "
                f"accuracy={self.accuracy:.2f}, "
                f"training_time={self.training_time}s)")


# ============================================================
# CHILD CLASS 1: NeuralNetwork
# ============================================================

class NeuralNetwork(MachineLearningModel):
    """
    Neural Network model.
    Demonstrates: Inheritance, method overriding, super().
    """
    
    framework = "TensorFlow"  # Overriding class variable
    
    def __init__(self, name, accuracy, training_time, layers, activation="ReLU"):
        # Call parent constructor
        super().__init__(name, accuracy, training_time)
        
        # New attributes specific to NeuralNetwork
        self.layers = layers
        self.activation = activation
    
    def train(self):
        """Override parent's train method."""
        print(f"[TRAIN] Neural Network: {self.name}")
        print(f"  Framework: {self.framework}")
        print(f"  Layers: {self.layers}")
        print(f"  Activation: {self.activation}")
        print(f"  Forward pass... Backward pass... Complete!")
    
    def __str__(self):
        return (f"NeuralNetwork(name={self.name}, "
                f"accuracy={self.accuracy:.2f}, "
                f"layers={self.layers}, "
                f"activation={self.activation})")


# ============================================================
# CHILD CLASS 2: RandomForest
# ============================================================

class RandomForest(MachineLearningModel):
    """
    Random Forest model.
    Demonstrates: Inheritance, method overriding.
    """
    
    framework = "Scikit-learn"
    
    def __init__(self, name, accuracy, training_time, n_estimators=100, max_depth=10):
        super().__init__(name, accuracy, training_time)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def train(self):
        print(f"[TRAIN] Random Forest: {self.name}")
        print(f"  Framework: {self.framework}")
        print(f"  Trees: {self.n_estimators}")
        print(f"  Max Depth: {self.max_depth}")
        print(f"  Building decision trees... Ensemble complete!")
    
    def __str__(self):
        return (f"RandomForest(name={self.name}, "
                f"accuracy={self.accuracy:.2f}, "
                f"trees={self.n_estimators})")


# ============================================================
# CHILD CLASS 3: XGBoost
# ============================================================

class XGBoost(MachineLearningModel):
    """
    XGBoost model.
    Demonstrates: Inheritance with default parameters.
    """
    
    framework = "XGBoost"
    
    def __init__(self, name, accuracy, training_time, estimators=100, learning_rate=0.3):
        super().__init__(name, accuracy, training_time)
        self.estimators = estimators
        self.learning_rate = learning_rate
    
    def train(self):
        print(f"[TRAIN] XGBoost: {self.name}")
        print(f"  Framework: {self.framework}")
        print(f"  Estimators: {self.estimators}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Gradient boosting... Complete!")
    
    def __str__(self):
        return (f"XGBoost(name={self.name}, "
                f"accuracy={self.accuracy:.2f}, "
                f"estimators={self.estimators})")