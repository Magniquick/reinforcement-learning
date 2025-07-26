import ale_py
import gymnasium as gym
import pygame
import numpy as np

# Initialize pygame
pygame.init()

# Create the Tetris environment with RGB rendering
env = gym.make("ALE/Tetris-v5", obs_type="ram", render_mode="rgb_array")

# Initialize the environment
state, info = env.reset()

rendered_image = env.render()
# Get the environment's original dimensions
env_width, env_height = len(rendered_image), len(rendered_image[0])

# Set up the Pygame window (scale the image)
scale_factor = 5  # Adjust this for larger or smaller window
window_width = env_width * scale_factor
window_height = env_height * scale_factor
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Tetris with Pygame')

# Define the keyboard keys for control
controls = {
    pygame.K_UP: 1,    # Rotate
    pygame.K_DOWN: 4,  # Move down
    pygame.K_LEFT: 3,  # Move left
    pygame.K_RIGHT: 2  # Move right
}

# Game loop
running = True
while running:
    # Process pygame events (keyboard inputs)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture keyboard inputs
    keys = pygame.key.get_pressed()
    action = 0  # Default no action

    for key, action_code in controls.items():
        if keys[key]:
            action = action_code

    # Take a step in the environment
    next_state, reward, terminated, truncated, info = env.step(action)

    # Render the environment to get the RGB image
    rendered_image = env.render()

    # Scale the image to fit the Pygame window
    scaled_image = pygame.surfarray.make_surface(np.transpose(rendered_image, (1, 0, 2)))
    scaled_image = pygame.transform.scale(scaled_image, (window_width, window_height))

    # Display the scaled image on the Pygame window
    screen.blit(scaled_image, (0, 0))
    pygame.display.flip()

    # Check if the game is over
    if terminated or truncated:
        print("Game Over!")
        break

    pygame.time.delay(100)  # Limit the speed of the game

# Close the environment and pygame window
env.close()
pygame.quit()
