from dm_control import suite
from dm_control.suite.wrappers import pixels
import matplotlib.pyplot as plt
import numpy as np
import cv2

max_frame = 90

width = 600
height = 480
video = np.zeros((max_frame, height, 2 * width, 3), dtype=np.uint8)

# Load one task:
env = suite.load(domain_name="cartpole", task_name="balance")
frame_skip = 4

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('swingup_test.avi', fourcc, 20.0, (640,480), True)

for i in range(1):
    if i % frame_skip == 0:
        action = np.random.uniform(action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        print(action)
        image = np.hstack([env.physics.render(height, width, camera_id=0),
                    env.physics.render(height, width, camera_id=1)])
        print(image.dtype)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('/home/ashwin/cfm-private/swingup_goal.png', image[:, :600])
        # out.write(image)
        cv2.imshow('Env', image)
        cv2.waitKey(1)
    time_step = env.step(action)
    #print(time_step.reward, time_step.discount, time_step.observation)
#   cv2.imwrite('swingup_goal.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # img = plt.imshow(video[i])
    # plt.pause(0.01)  # Need min display time > 0.0.
    # plt.draw()
out.release()
cv2.waitKey(0)
