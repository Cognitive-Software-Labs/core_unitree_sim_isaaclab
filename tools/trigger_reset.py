#!/usr/bin/env python3
# trigger_reset.py
# Helper script to publish a reset command over DDS to the simulation environment.

import sys
import time
import argparse
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

def main():
    parser = argparse.ArgumentParser(description="Trigger Simulation Environment Reset / Randomization")
    parser.add_argument(
        "--type", 
        type=str, 
        choices=["object", "all"], 
        default="all", 
        help="Reset type: 'object' (re-randomize layout, keep current simulation state) or 'all' (reset scene defaults and re-randomize)."
    )
    parser.add_argument("--domain", type=int, default=1, help="DDS Domain ID (must match ChannelFactoryInitialize(X) in sim_main.py, default is 1)")
    args = parser.parse_args()

    # Define code categories matching sim_main.py:
    # "1" = reset object / re-randomize layout
    # "2" = reset all / default restore and re-randomize
    category_code = "1" if args.type == "object" else "2"

    print(f"Initializing DDS on domain {args.domain}...")
    ChannelFactoryInitialize(args.domain)
    
    print("Creating subscriber channel for 'rt/reset_pose/cmd'...")
    pub = ChannelPublisher("rt/reset_pose/cmd", String_)
    pub.Init()

    # Create standard string message
    msg = String_()
    msg.data = category_code

    print(f"Sending reset command: category_code={category_code} ({args.type.upper()} reset)...")
    
    # Send a few times to ensure reception over lossy transport
    for i in range(5):
        pub.Write(msg)
        time.sleep(0.1)

    print("Reset command sent successfully!")

if __name__ == "__main__":
    main()
