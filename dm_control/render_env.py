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

# out = cv2.VideoWriter('two_hand.avi', -1, 20.0, (640,480))

# while not time_step.last():
for i in range(100):
    action = np.random.uniform(action_spec.minimum,
                                action_spec.maximum,
                                size=action_spec.shape)
    # action = np.array([.5,.5,.5,-.5,-.5,-.5])
    action = np.array([0.4,0.4,.8,.8])
    # print(action)
    time_step = env.step(action)
    image = np.hstack([env.physics.render(height, width, camera_id=0),
                            env.physics.render(height, width, camera_id=1)])
    #print(time_step.reward, time_step.discount, time_step.observation)
    cv2.imwrite('cloth_two_hand.png', image)
    # out.write(image)
    cv2.imshow('Env', image)
    cv2.waitKey(50)
    # img = plt.imshow(video[i])
    # plt.pause(0.01)  # Need min display time > 0.0.
    # plt.draw()
# out.release()
cv2.waitKey(0)
