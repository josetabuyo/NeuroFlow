#!/usr/bin/env python3
"""Test WebSocket connection for NeuroFlow."""

import asyncio
import websockets
import json
import sys

async def test_websocket():
    uri = "ws://localhost:8510/ws/experiment"

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")

            # Send start message for Dynamic SOM
            start_message = {
                "action": "start",
                "config": {
                    "description": "Extends Sharp Pow Daemon (avg_vs_avg + x_pow_2 tension) with HALF_TOP/HALF_BOT input. No noise. The goal: find out how many ticks the network needs to settle on a consistent internal state for each pattern, and whether that state persists. Vary frames_per_char.",
                    "grid": {
                        "width": 50,
                        "height": 50
                    },
                    "wiring": {
                        "mask": "deamon_e3_g2_i12_de1_di1",
                        "dendrite_exc_weight": 0.9,
                        "dendrite_inh_weight": -1,
                        "process_mode": "avg_vs_avg",
                        "tension_function": {
                            "x": 3,
                            "x_pow_3": 9,
                            "x_pow_2": 2
                        }
                    },
                    "input": {
                        "text": "HALF_TOP,HALF_BOT",
                        "resolution": 20,
                        "frames_per_char": 10,
                        "dendrite_input_weight": 0.2
                    },
                    "learning": {
                        "rate": 0.01
                    }
                }
            }

            print(f"📤 Sending start message: {json.dumps(start_message, indent=2)}")
            await websocket.send(json.dumps(start_message))

            # Wait for response
            print("⏳ Waiting for response...")
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"📥 Received: {response}")

            # Send step message
            step_message = {"action": "step", "count": 1}
            print(f"📤 Sending step message: {json.dumps(step_message)}")
            await websocket.send(json.dumps(step_message))

            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"📥 Received: {response}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    # Check if websockets is installed
    try:
        import websockets
    except ImportError:
        print("Installing websockets package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
        import websockets

    asyncio.run(test_websocket())