import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def plot_predictions(pickle_file):
    # Load the pickle file
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    # Extract predictions and true values
    y_pred = data["y_pred"]
    y_true = data["y_true"]

    # Reshape or select a specific time step
    # y_pred = y_pred[:, 0, 0]  # Select the first time step for plotting
    # y_true = y_true[:, 0, 0]  # Select the first time step for plotting

    # Reshape to (25909, 11) for plotting
    y_pred = y_pred.squeeze(-1)
    y_true = y_true.squeeze(-1)

    # Plot multiple sequences
    plt.figure(figsize=(12, 6))

    plt.plot(y_true[0], label=f"True", linestyle="--", alpha=0.5, color="blue")
    plt.plot(y_pred[0], label=f"Pred", alpha=0.5, color="red")

    # for i in range(10):  # Plot only 10 samples for readability
    #     plt.plot(y_true[i], label=f"True {i+1}", linestyle="--", alpha=0.5, color="blue")
    #     plt.plot(y_pred[i], label=f"Pred {i+1}", alpha=0.5, color="red")

    plt.xlabel("Time Steps")
    plt.ylabel("Glucose")
    plt.title("True vs Predicted Time Series")
    plt.legend()
    plt.show()

    # Animated time series
    # Create a directory to save frames
    save_dir = "/Users/sneha/Downloads/CausalTransformer-main/animation_frames"
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], 'b-', label="True")
    line2, = ax.plot([], [], 'r-', label="Predicted")

    # def update(frame):
    #     ax.clear()
    #     ax.plot(y_true[frame], 'b-', label="True")
    #     ax.plot(y_pred[frame], 'r-', label="Predicted")
    #     ax.set_title(f"Time Series Sample {frame}")
    #     ax.legend()

    #     # Save the frame
    #     plt.savefig(os.path.join(save_dir, f"frame_{frame:03d}.png"))

    # ani = animation.FuncAnimation(fig, update, frames=10, interval=500)
    # plt.show()

    # Create a figure with subplots (e.g., 5 rows, 10 columns for 50 frames)
    # num_frames = 10
    cols = 2  # Number of columns
    rows = 5  # Calculate required rows
    num_frames = cols * rows
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10), sharex=True, sharey=True)

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Plot each frame in a separate subplot
    for frame in range(num_frames):
        if frame < len(axes):  # Ensure we don't exceed available subplots
            axes[frame].plot(y_true[frame], 'b-', label="True", alpha=0.7)
            axes[frame].plot(y_pred[frame], 'r-', label="Predicted", alpha=0.7)
            axes[frame].set_title(f"Frame {frame}")
            axes[frame].set_xticks([])
            axes[frame].set_yticks([])

    # Remove unused subplots (if any)
    for i in range(frame + 1, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print('+++ ENTRY IN NEW FILE')
    if len(sys.argv) != 2:
        print("Usage: python plot_predictions.py <pickle_file>")
    else:
        plot_predictions(sys.argv[1])
        # with open(sys.argv[1], 'rb') as f:
        #     data = pickle.load(f)

        # y_pred = data["y_pred"].flatten()
        # y_true = data["y_true"].flatten()

        # plt.figure(figsize=(10, 5))
        # plt.plot(y_true[:100], label="True", linestyle='dashed')
        # plt.plot(y_pred[:100], label="Predicted")
        # plt.legend()
        # plt.title("First 100 Predictions vs. Ground Truth")
        # plt.show()
