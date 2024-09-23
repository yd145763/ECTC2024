# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 05:16:05 2024

@author: limyu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


N = np.arange(0,161,1)

n = 40

df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
df_full = df_full.reset_index(drop = True)
df_full.columns = range(df_full.shape[1])
df = df_full.iloc[25:65, 129:169]

   
df = df.reset_index(drop = True)
df.columns = range(df.shape[1])


data = np.array(df)

# Create a figure and axes
fig, ax = plt.subplots()

# Plot the data using imshow
im = ax.imshow(data, cmap='YlGnBu', interpolation='none')
# Adjust spacing between subplots
plt.tight_layout()
plt.axis('off')
# Show the plot
plt.show()

# Example 40x40 data array
#data = np.random.rand(40, 40)

# Define patch size
patch_size = 10
num_patches = data.shape[0] // patch_size

# Calculate the global min and max for normalization
data_min = np.min(data)
data_max = np.max(data)

# Create a figure with subplots
fig, axs = plt.subplots(num_patches, num_patches, figsize=(10, 10))

# Plot each patch in its own subplot with standardized color scales
for i in range(num_patches):
    for j in range(num_patches):
        print(i, j)
        patch = data[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
        ax = axs[i, j]
        im = ax.imshow(patch, cmap='YlGnBu', interpolation='none', vmin=data_min, vmax=data_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        

# Adjust spacing between subplots
plt.tight_layout()


# Show the plot
plt.show()


import pandas as pd
import ast
df = pd.read_csv("https://raw.githubusercontent.com/yd145763/ECTC2024/refs/heads/main/df_results.csv")

cnn_train_loss_list = []
for i in range(len(df)):
    string_list = df['cnn_train_loss_list'][i]
    parsed_list = ast.literal_eval(string_list)
    cnn_train_loss_list.append(parsed_list)
df['cnn_train_loss_list'] = cnn_train_loss_list

cnn_val_loss_list = []
for i in range(len(df)):
    string_list = df['cnn_val_loss_list'][i]
    parsed_list = ast.literal_eval(string_list)
    cnn_val_loss_list.append(parsed_list)
df['cnn_val_loss_list'] = cnn_val_loss_list

vit_train_loss_list = []
for i in range(len(df)):
    string_list = df['vit_train_loss_list'][i]
    parsed_list = ast.literal_eval(string_list)
    vit_train_loss_list.append(parsed_list)
df['vit_train_loss_list'] = vit_train_loss_list

vit_val_loss_list = []
for i in range(len(df)):
    string_list = df['vit_val_loss_list'][i]
    parsed_list = ast.literal_eval(string_list)
    vit_val_loss_list.append(parsed_list)
df['vit_val_loss_list'] = vit_val_loss_list

cnn_val_acc_list = []
for i in range(len(df)):
    string_list = df['cnn_val_acc_list'][i]
    parsed_list = ast.literal_eval(string_list)
    cnn_val_acc_list.append(parsed_list)
df['cnn_val_acc_list'] = cnn_val_acc_list

vit_val_acc_list = []
for i in range(len(df)):
    string_list = df['vit_val_acc_list'][i]
    parsed_list = ast.literal_eval(string_list)
    vit_val_acc_list.append(parsed_list)
df['vit_val_acc_list'] = vit_val_acc_list



cnn_max_index_list = [pd.Series(i).idxmax() for i in df['cnn_val_acc_list']]
df['cnn_max_index'] = cnn_max_index_list
vit_max_index_list = [pd.Series(i).idxmax() for i in df['vit_val_acc_list']]
df['vit_max_index'] = vit_max_index_list

cnn_difference = []
for i in range(len(df)):
    cnn_best_index = df['cnn_max_index'][i]
    difference = (df['cnn_train_loss_list'][i])[cnn_best_index] - (df['cnn_val_loss_list'][i])[cnn_best_index]
    cnn_difference.append(difference)
df['cnn_difference'] = cnn_difference

vit_difference = []
for i in range(len(df)):
    vit_best_index = df['vit_max_index'][i]
    difference = (df['vit_train_loss_list'][i])[vit_best_index] - (df['vit_val_loss_list'][i])[vit_best_index]
    vit_difference.append(difference)
df['vit_difference'] = vit_difference
    
    
vit_acc = df['vit_acc']
vit_acc.median()
vit_median_index = (vit_acc - vit_acc.median()).abs().idxmin()
cnn_acc = df['cnn_acc']
cnn_acc.median()
cnn_median_index = (cnn_acc - cnn_acc.median()).abs().idxmin()

from matplotlib.ticker import StrMethodFormatter
import matplotlib.pyplot as plt 
# Combine data into a list
data =[vit_acc, cnn_acc]

fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
# Create box plots
box = ax.boxplot(data, patch_artist=True, widths=0.5)

# Set the fill color of each box to 'none' and adjust line thickness
for patch in box['boxes']:
    patch.set(facecolor='none')
    patch.set_linewidth(1)  # Adjust the thickness of the box lines

# Adjust whisker and cap thickness
for whisker in box['whiskers']:
    whisker.set_linewidth(1)  # Adjust the thickness of the whisker lines
for cap in box['caps']:
    cap.set_linewidth(1)  # Adjust the thickness of the cap lines

# Adjust median line thickness
for median in box['medians']:
    median.set_linewidth(1.5)  # Adjust the thickness of the median line
    median.set(color='red')

# Adjust flier (outlier) markers
for flier in box['fliers']:
    flier.set_markeredgewidth(1)  # Adjust the thickness of the marker edges

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
labels = ['ViT', 'CNN']
ax.set_xticklabels(labels, weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_linewidth(1)
ax.spines["top"].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

plt.ylabel("Accuracies")

# Show the plot
plt.show()

vit_time = df['vit_time']
vit_time.median()
cnn_time = df['cnn_time']
cnn_time.median()

# Combine data into a list
data =[vit_time, cnn_time]

fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
# Create box plots
box = ax.boxplot(data, patch_artist=True, widths=0.5)

# Set the fill color of each box to 'none' and adjust line thickness
for patch in box['boxes']:
    patch.set(facecolor='none')
    patch.set_linewidth(1)  # Adjust the thickness of the box lines

# Adjust whisker and cap thickness
for whisker in box['whiskers']:
    whisker.set_linewidth(1)  # Adjust the thickness of the whisker lines
for cap in box['caps']:
    cap.set_linewidth(1)  # Adjust the thickness of the cap lines

# Adjust median line thickness
for median in box['medians']:
    median.set_linewidth(1.5)  # Adjust the thickness of the median line
    median.set(color='red')

# Adjust flier (outlier) markers
for flier in box['fliers']:
    flier.set_markeredgewidth(1)  # Adjust the thickness of the marker edges

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
labels = ['ViT', 'CNN']
ax.set_xticklabels(labels, weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_linewidth(1)
ax.spines["top"].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

plt.ylabel("Training Duration (s)")

# Show the plot
plt.show()

vit_epoch = df['vit_max_index']
vit_epoch.median()
cnn_epoch = df['cnn_max_index']
cnn_epoch.median()

# Combine data into a list
data =[vit_epoch, cnn_epoch]

fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
# Create box plots
box = ax.boxplot(data, patch_artist=True, widths=0.5)

# Set the fill color of each box to 'none' and adjust line thickness
for patch in box['boxes']:
    patch.set(facecolor='none')
    patch.set_linewidth(1)  # Adjust the thickness of the box lines

# Adjust whisker and cap thickness
for whisker in box['whiskers']:
    whisker.set_linewidth(1)  # Adjust the thickness of the whisker lines
for cap in box['caps']:
    cap.set_linewidth(1)  # Adjust the thickness of the cap lines

# Adjust median line thickness
for median in box['medians']:
    median.set_linewidth(1.5)  # Adjust the thickness of the median line
    median.set(color='red')

# Adjust flier (outlier) markers
for flier in box['fliers']:
    flier.set_markeredgewidth(1)  # Adjust the thickness of the marker edges

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
labels = ['ViT', 'CNN']
ax.set_xticklabels(labels, weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_linewidth(1)
ax.spines["top"].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

plt.ylabel("Best Epoch")

# Show the plot
plt.show()
plt.close()

vit_difference = df['vit_difference']
vit_difference.median()
cnn_difference = df['cnn_difference']
cnn_difference.median()

# Combine data into a list
data =[vit_difference, cnn_difference]

fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
# Create box plots
box = ax.boxplot(data, patch_artist=True, widths=0.5)

# Set the fill color of each box to 'none' and adjust line thickness
for patch in box['boxes']:
    patch.set(facecolor='none')
    patch.set_linewidth(1)  # Adjust the thickness of the box lines

# Adjust whisker and cap thickness
for whisker in box['whiskers']:
    whisker.set_linewidth(1)  # Adjust the thickness of the whisker lines
for cap in box['caps']:
    cap.set_linewidth(1)  # Adjust the thickness of the cap lines

# Adjust median line thickness
for median in box['medians']:
    median.set_linewidth(1.5)  # Adjust the thickness of the median line
    median.set(color='red')

# Adjust flier (outlier) markers
for flier in box['fliers']:
    flier.set_markeredgewidth(1)  # Adjust the thickness of the marker edges

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
labels = ['ViT', 'CNN']
ax.set_xticklabels(labels, weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_linewidth(1)
ax.spines["top"].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

plt.ylabel("Training - Validation")

# Show the plot
plt.show()
plt.close()



import numpy as np
df_cnn_conf = df['cnn_conf']
conf_string = df_cnn_conf.iloc[cnn_median_index]
matrix_list = [int(x) for x in conf_string.replace('[', '').replace(']', '').split()]
confusion_matrix = np.array(matrix_list).reshape(4, 4)

import seaborn as sns
# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 4))
ax = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='YlGnBu', 
            xticklabels=['Region A', 'Region B', 'Region C', 'Region D'], 
            yticklabels=['Region A', 'Region B', 'Region C', 'Region D'])


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

font = {'color': 'black', 'weight': 'bold', 'size': 12}
ax.set_ylabel("Actual", fontdict=font)
ax.set_xlabel("Predicted", fontdict=font)

# Setting tick labels bold
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(12)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')

plt.show()
plt.close()

import numpy as np
df_vit_conf = df['vit_conf']
conf_string = df_vit_conf.iloc[vit_median_index]
matrix_list = [int(x) for x in conf_string.replace('[', '').replace(']', '').split()]
confusion_matrix = np.array(matrix_list).reshape(4, 4)

import seaborn as sns
# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 4))
ax = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='YlGnBu', 
            xticklabels=['Region A', 'Region B', 'Region C', 'Region D'], 
            yticklabels=['Region A', 'Region B', 'Region C', 'Region D'])


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

font = {'color': 'black', 'weight': 'bold', 'size': 12}
ax.set_ylabel("Actual", fontdict=font)
ax.set_xlabel("Predicted", fontdict=font)

# Setting tick labels bold
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(12)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')

plt.show()
plt.close()