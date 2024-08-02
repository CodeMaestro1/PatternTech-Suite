import matplotlib.pyplot as plt

# Open the file and read the data
with open('Statistical_Calculations/Final_Stats.txt', 'r') as f:
    lines = f.readlines()[1:]  # Skip the first line

# Initialize lists
epochs = []
training_accuracy = []
validation_accuracy = []

# Parse the data
for line in lines:
    epoch, train_acc, loss, val_acc, val_loss = map(float, line.split())
    epochs.append(epoch)
    training_accuracy.append(train_acc)
    validation_accuracy.append(val_acc)

# Plotting the accuracies
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, validation_accuracy, label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
