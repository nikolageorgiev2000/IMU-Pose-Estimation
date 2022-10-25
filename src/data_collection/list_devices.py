import asyncio
from bleak import BleakScanner

async def run():
    devices = await BleakScanner.discover()
    print(len(devices))
    for d in devices:
        if len(d.name) > 0:
            print(d.name)

loop = asyncio.get_event_loop()
loop.run_until_complete(run())