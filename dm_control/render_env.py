from dm_control import suite
import matplotlib.pyplot as plt
import numpy as np
import cv2

max_frame = 90

width = 600
height = 480
video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)

# Load one task:
env = suite.load(domain_name="rope_two_hand", task_name="easy")

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()
# while not time_step.last():
for i in range(50):
    action = np.random.uniform(action_spec.minimum,
                                action_spec.maximum,
                                size=action_spec.shape)
    # action = np.array([.5,.5,.5,-.5,-.5,-.5])
    # print(action)
    time_step = env.step(action)
    video = np.hstack([env.physics.render(height, width, camera_id=0),
                            env.physics.render(height, width, camera_id=1)])
    #print(time_step.reward, time_step.discount, time_step.observation)
    cv2.imshow('Env', video)
    cv2.waitKey(200)
    # img = plt.imshow(video[i])
    # plt.pause(0.01)  # Need min display time > 0.0.
    # plt.draw()
cv2.waitKey(0)