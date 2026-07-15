import robosuite as suite
import imageio

"""
This is just a script to help me visualize each of the camera views for
the Lift Envrironment.
"""

cameras = [
    "frontview",
    "birdview",
    "agentview",
    "sideview",
    "robot0_robotview",
    "robot0_eye_in_hand",
]

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=cameras,
    camera_heights=256,
    camera_widths=256,
)

obs = env.reset()

for cam in cameras:
    img = obs[f"{cam}_image"]
    if img.dtype != "uint8":
        img = (img * 255).astype("uint8") if img.max() <= 1.0 else img.astype("uint8")
    imageio.imwrite(f"{cam}.png", img)
    print(f"wrote {cam}.png")

env.close()