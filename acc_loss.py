import os
import matplotlib.pyplot as plt
import numpy as np

# Create empty lists to store the values
accuracy_list, loss_list = [], []
val_accuracy_list, val_loss_list = [], []


path = f"{os.getcwd()}/logs/patchless/fodb/"

# Define a list of marker styles for the accuracy plots
markers = [    
    ".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4",
    "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_",
    "P", "X", "d", "o", "v", "^", "<", ">", "s", "p", "h", "H", "+", "x", "D", "d", "|", "_",    
    "o", "v", "^", "<", ">", "1", "2", "3", "4", "s", "p", "*", "h", "H", "+", "x", "d", "|", "_"
]


# Loop through all the log files
for filename in os.listdir(path):

    # Extract the accuracy and loss values from the log file
    with open(path + filename, "r") as f:
        lines = f.readlines()
        accuracy = [float(line.split(',')[1]) for line in lines[1:]]
        loss = [float(line.split(',')[2]) for line in lines[1:]]
        val_accuracy = [float(line.split(',')[3]) for line in lines[1:]]
        val_loss = [float(line.split(',')[4]) for line in lines[1:]]


    # Append the values to the corresponding list with the file name as label
    accuracy_list.append(accuracy)
    val_accuracy_list.append(val_accuracy)


    loss_list.append(loss)
    val_loss_list.append(val_loss)


j = 0

# Plot the accuracy graph
fig, ax = plt.subplots()
for i, accuracy in enumerate(accuracy_list):
    # Use the corresponding marker style for the current plot
    ax.plot(np.arange(len(accuracy)), accuracy, label=f'{os.listdir(path)[i]} train', marker=markers[j], linestyle='-', linewidth=2)
    j +=1

for i, accuracy in enumerate(val_accuracy_list):
    # Use the corresponding marker style for the current plot
    ax.plot(np.arange(len(accuracy)), accuracy, label=f'{os.listdir(path)[i]} val', marker=markers[j], linestyle='-', linewidth=2)    
    j += 1

ax.set_title("Accuracy vs Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()

fig.set_size_inches(10, 10)

plt.savefig(f'{os.getcwd()}/1d_fodb_acc.png', dpi=100)
plt.show()



# Define a list of marker styles for the loss plots


# Plot the loss graph
fig, ax = plt.subplots()

j = 0
for i, loss in enumerate(loss_list):
    # Use the corresponding marker style for the current plot
    ax.plot(np.arange(len(loss)), loss, label=f'{os.listdir(path)[i]} train', marker=markers[j], linestyle='-', linewidth=2)
    j += 1

for i, loss in enumerate(val_loss_list):
    # Use the corresponding marker style for the current plot
    ax.plot(np.arange(len(loss)), loss, label=f'{os.listdir(path)[i]} val', marker=markers[j], linestyle='-', linewidth=2)
    j +=1


ax.set_title("Loss vs Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.set_ylim([0, 1])

fig.set_size_inches(10, 10)

plt.savefig(f'{os.getcwd()}/1d_fodb_loss.png', dpi=100)
plt.show()
