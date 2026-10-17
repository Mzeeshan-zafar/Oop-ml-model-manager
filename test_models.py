"""
Test script for OOP ML Model Manager
Demonstrates: Polymorphism, Encapsulation, Inheritance
"""

from ml_models import MachineLearningModel, NeuralNetwork, RandomForest, XGBoost


def print_separator(title):
    """Helper to print a formatted separator."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ============================================================
# TEST 1: Create Individual Models
# ============================================================

print_separator("TEST 1: CREATE MODELS")

# Base model
base_model = MachineLearningModel("Logistic Regression", 0.85, 2.3)
base_model.train()
base_model.evaluate()
print(base_model)

# Neural Network
nn = NeuralNetwork("CNN Image Classifier", 0.92, 5.1, layers=4)
nn.train()
nn.evaluate()
print(nn)

# Random Forest
rf = RandomForest("Random Forest Classifier", 0.89, 3.8, n_estimators=150)
rf.train()
rf.evaluate()
print(rf)

# XGBoost
xgb = XGBoost("XGBoost Classifier", 0.94, 4.2, estimators=200)
xgb.train()
xgb.evaluate()
print(xgb)


# ============================================================
# TEST 2: Encapsulation (Getters & Setters)
# ============================================================

print_separator("TEST 2: ENCAPSULATION")

print(f"Current version: {base_model.get_version()}")
base_model.set_version("2.0.0")
print(f"Updated version: {base_model.get_version()}")

# Try invalid version
base_model.set_version("")  # Should fail


# ============================================================
# TEST 3: Polymorphism
# ============================================================

print_separator("TEST 3: POLYMORPHISM")

models = [base_model, nn, rf, xgb]

for model in models:
    print(f"\n--- {model.__class__.__name__} ---")
    model.train()
    model.evaluate()


# ============================================================
# TEST 4: Class Variable vs Instance Variable
# ============================================================

print_separator("TEST 4: CLASS vs INSTANCE VARIABLES")

print(f"MachineLearningModel.framework = {MachineLearningModel.framework}")
print(f"NeuralNetwork.framework = {NeuralNetwork.framework}")
print(f"RandomForest.framework = {RandomForest.framework}")
print(f"XGBoost.framework = {XGBoost.framework}")

print(f"\nInstance variable (unique):")
print(f"  base_model.name = {base_model.name}")
print(f"  nn.name = {nn.name}")


# ============================================================
# TEST 5: Model Comparison (Sorted by Accuracy)
# ============================================================

print_separator("TEST 5: MODEL COMPARISON")

sorted_models = sorted(models, key=lambda m: m.accuracy, reverse=True)
print("Models sorted by accuracy:")
for i, model in enumerate(sorted_models, 1):
    print(f"  {i}. {model.name}: {model.accuracy * 100:.2f}%")

best_model = sorted_models[0]
print(f"\n🏆 Best Model: {best_model.name} ({best_model.accuracy * 100:.2f}%)")