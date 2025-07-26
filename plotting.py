import matplotlib.pyplot as plt
import matplotlib
import torch
matplotlib.use('Qt5Agg')  # Use TkAgg or Qt5Agg for interactive plotting

class plot:
    def __init__(self):
        # For tracking rewards
        self.episode_rewards = []  # List to store rewards per episode

        # Setup the plot
        plt.ion()  # Turn on interactive mode for real-time plotting
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-', label="Net Reward")
        self.mean_line, = self.ax.plot([], [], 'b-', label="100 episode Mean")  # Line for the mean        self.ax.set_ylim(0, 500)  # Set limits for y-axis (net reward range)
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Net Reward')
        self.ax.set_title('Agent Training Performance')
        self.ax.legend()

    def append_reward(self, reward):
        """Append the net reward of the current episode."""
        self.episode_rewards.append(reward)
    

    def update_plot(self):
        rewards_tensor = torch.tensor(self.episode_rewards)
        # Update the plot with real-time data
        self.line.set_xdata(range(len(self.episode_rewards)))
        self.line.set_ydata(rewards_tensor.numpy())
        self.ax.relim()  # Recalculate axis limits
        self.ax.autoscale_view()  # Autoscale the view

        # Update the mean line
        means = torch.zeros(1)  # Initialize means
        if len(self.episode_rewards) >= 100:
            means = rewards_tensor.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            self.mean_line.set_xdata(range(len(means)))  # Update x data
            self.mean_line.set_ydata(means.numpy())  # Update y data

        plt.draw()  # Redraw the plot with updated data
        plt.pause(0.001)

        return (means[-1])

    def finish_plot(self):
        """Finalize the plot after training."""
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Ensure the plot stays visible after training

    def close(self):
        plt.close()
