from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from gym.envs.robotics import task_definitions


if __name__ == "__main__":

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # Choose environment and add it to options
    options["env_name"] = "Cloth"
    options["robots"] = "Panda"
    controller_name = "OSC_POSE"  # choose_controller()
    controller_name = "IK_POSE"

    options["controller_configs"] = load_controller_config(
        default_controller=controller_name)
    options["controller_configs"]["interpolation"] = "linear"
    print("KÖNFIG", options["controller_configs"])

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=10,
        constraints=task_definitions.constraints['diagonal_franka_1']
    )
    env.reset()
    # env.viewer.set_camera(camera_id=0)

    # Get action limits
    low, high = env.action_spec

    print(f"LOW: {low}")
    print(f"HIGH: {high}")

    site_pos = env.sim.data.get_site_xpos("gripper0_grip_site").copy()
    increases = np.array([
        [-0.8169702, 0.988936, 0.823965],
        [-0.94578123, 0.9432483, 0.7019442],
        [-0.96698755, 0.81606424, 0.78746295],
        [-0.97489953, 0.09362749, 0.86515844],
        [-0.97881544, -0.5612759, 0.98431313],
        [-0.9972523, -0.99255085, 0.9692926],
        [-0.997425, -0.98192877, 0.104845054],
        [-0.99912775, -0.6859133, -0.9893226],
        [-0.9040002, 0.7857158, -0.9968162],
        [-0.545036, -0.34209844, -0.9746304],
    ])*0.03
    #increase = np.array([0,0,0.001])
    # do visualization
    split = 20
    modded_increases = []
    for incr in increases:
        print("incr", incr)
        for s in range(1, split+1):
            modded_increases.append([x / s for x in incr])

    #print("modded", modded_increases)
    increases = np.array(modded_increases, dtype=object)
    # while True:
    # env.render()
    while True:
        env.reset()
        for i in range(200):
            print("Step", i)
            action = np.zeros(7)
            #action[2] = np.sin(i*0.025)*0.1
            action[2] = 0.01
            obs, reward, done, _ = env.step(action)
            site_pos_new = env.sim.data.get_site_xpos(
                "gripper0_grip_site").copy()
            #print("error vector", (site_pos + increase) - site_pos_new)
            #print("error norm", np.linalg.norm((site_pos + increase) - site_pos_new))
            #site_pos = site_pos_new
            env.render()
