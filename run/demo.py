#!/usr/bin/env python3
from rocobrick.env.Env import Env
from rocobrick.utils import *
from rocobrick.policy.Policy import Policy

async def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Create BrickSim environment
    env = Env(root_dir=SCRIPT_DIR,
              user_config_path="../config/user_config.json",
              system_config_path="../config/system_config.json")
    await env.reset()

    # Start simulation loop
    await env.play()
    await env.get_robot_ready()

    # User Policy
    my_policy = Policy(env)
    
    while(1):
        if my_policy.is_done():
            print("Demo complete.")
            break
        obs = env.get_observations()
        q = my_policy.get_action(obs)
        env.robot_apply_action(q)
        await env.step()