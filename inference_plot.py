import matplotlib.pyplot as plt
import os

dataset_name="myeloid"

if dataset_name=="ms":
    # Inference times (duration in seconds) and corresponding accuracy (in percentage)
    methods = ["Type 1","Type II", "Type III","Type IV",'scGPT', 'scBERT', 'CellPLM', 'TOSICA']
    inference_times = [ 0.032,0.028,0.037, 25.22, 16.43, 416.09, 9.29, 5.05 ]  # Example values for one dataset
    accuracies = [81.03, 87.55,88.17,86.70, 84.96,75.09,87.88,69.37]  # Hypothetical accuracy values

if dataset_name=="pancreas":

    methods = ["Type 1","Type II", "Type III","Type IV",'scGPT', 'scBERT', 'CellPLM', 'TOSICA']
    inference_times = [ 0.026,0.022,0.028, 19.90, 6.25, 341.93, 9.36, 2.19]  # Example values for one dataset
    accuracies = [89.16,97.66,97.76,97.27,96.41,96.45,96.30,97.33]  # Hypothetical accuracy values


if dataset_name=="myeloid":

    methods = ["Type 1","Type II", "Type III","Type IV",'scGPT', 'scBERT', 'CellPLM', 'TOSICA']
    inference_times = [ 0.025,0.023,0.026, 14.10, 4.35, 176.16, 10.09, 1.96]  # Example values for one dataset
    accuracies = [65.39,67.97,68.08,64.82,63.47,10.93,62.84,47.35]  # Hypothetical accuracy values


# Create the figure and axis
fig, ax = plt.subplots(figsize=(6,4))


# Define different marker styles for each method
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']  # Circle, square, triangle up, diamond, triangle down, pentagon, star, hexagon
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']  #



# Plot data with different markers
for i, method in enumerate(methods):
    ax.scatter(inference_times[i], accuracies[i], color=colors[i], s=100, marker=markers[i])

# Annotate each point with the method name
for i, method in enumerate(methods):
    ax.annotate(method, (inference_times[i], accuracies[i]), fontsize=10, fontweight='bold', ha='left')

# Set labels with bold font
ax.set_xlabel('Inference Time (s)', fontweight='bold', fontsize=15)
ax.set_ylabel('Test Accuracy (%)', fontweight='bold',fontsize=15)


# Set the x-axis range from 10^-3 to 10^3
ax.set_xlim(10**-3, 10**3)
# Set x-axis to log scale
ax.set_xscale('log')

plt.xticks(fontweight="bold",fontsize=12)
plt.yticks(fontweight="bold",fontsize=12)


legend = ax.legend(title=dataset_name.upper(), bbox_to_anchor=(1,1),fontsize="large", facecolor='blue', loc="upper right",edgecolor='black')
plt.setp(legend.get_title(), fontweight='bold')


# Bold the frame
for spine in ax.spines.values():
    spine.set_linewidth(2)

# Bold the ticks
ax.tick_params(axis='both', which='major', labelsize=8, width=2)

plt.grid(True)

# Adjust layout
plt.tight_layout()

# Ensure the directory exists
output_dir = "inference_plots"
os.makedirs(output_dir, exist_ok=True)

# Save the plot
output_path = os.path.join(output_dir, f"{dataset_name}_inference.png")
plt.savefig(output_path,dpi=1000)



