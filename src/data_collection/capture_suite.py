# Needed to have a class type hint within the class's definition
# More: https://www.python.org/dev/peps/pep-0563/#backwards-compatibility
from __future__ import annotations

from src import *
from .posers import IMU_Record

from bleak import BleakClient, BleakScanner, BleakError
from bleak.backends.device import BLEDevice
import uuid
import asyncio
import json
from enum import IntEnum

class Motion_Type(IntEnum):
    MISC = 0
    Xpos_STILL = 1
    Ypos_STILL = 2
    Zpos_STILL = 3
    Xneg_STILL = 4
    Yneg_STILL = 5
    Zneg_STILL = 6
    X_CLK = 7
    Y_CLK = 8
    Z_CLK = 9
    CALIB = 10


class Tracker(BleakClient):
    def __init__(self, device: BLEDevice, **kwargs):
        super().__init__(device, **kwargs)
        # prefix string with namespace to prevent name conflicts.
        self.id = uuid.uuid3(uuid.NAMESPACE_DNS,
                             device.name + device.address).hex
        self.device = device
        self.ble_ids = {
            "service": "eb03300a-160b-4c02-91d7-75cef930279f",
            "IMU": "9cfd98a5-7bb9-4ae9-b0de-8a81ee6800c9",
            "poll_rate": "553f87f5-351a-49ed-83a4-2a640c0c7084",
        }

    async def set_poll_rate(self, sample_rate: int):
        await self.write_gatt_char(self.ble_ids["poll_rate"], int.to_bytes(sample_rate, 1, 'big', signed=False), response=True)

    async def capture_IMU(self, capture_length: int, sample_rate: int, recorder_callback):
        await self.set_poll_rate(sample_rate)
        await self.start_notify(self.ble_ids["IMU"], recorder_callback)
        await asyncio.sleep(capture_length)
        try:
            await self.stop_notify(self.ble_ids["IMU"])
            await self.set_poll_rate(0)
        except BleakError:
            print(f"Failed to stop tracker {self.device.name}.")


class Trial(object):
    def __init__(self, name: str = "", motion_type=Motion_Type.MISC) -> None:
        super().__init__()
        self.name = name
        self.id = uuid.uuid4().hex
        self.motion_type = motion_type
        self.records: List[IMU_Record] = []

    async def capture_data(self, capture_length: int, sample_rate: int, trackers: List[Tracker]):
        self.records = [IMU_Record(
            tracker.device.name, tracker.id, sample_rate) for tracker in trackers]
        record_tracker_pairs = zip(self.records, trackers)

        '''
        Helpful: https://stackoverflow.com/a/34021333
        The expression [(lambda x: x * i) for i in range(4)]
        is roughly equivalent to:
        [(lambda x: x * 3), (lambda x: x * 3), ... (lambda x: x * 3)] and NOT [(lambda x: x * 0), (lambda x: x * 1), ... (lambda x: x * 3)]
        The lambdas in the list comprehension are a closure over the scope of this comprehension;
        i.e. a lexical closure, so they refer to the `i` via reference, and not its value when they were evaluated!
        Hence, I use the weird double lambda below, otherwise the last value of the referenced pair records everything.
        '''
        await asyncio.gather(*((lambda r, t: t.capture_IMU(capture_length, sample_rate, lambda _, data: r.record_sample(data)))(record, tracker) for record, tracker in record_tracker_pairs))

    def save_trial(self, session_dir: str):
        trial_dir = os.path.join(session_dir, self.name)
        if not os.path.exists(trial_dir):
            os.mkdir(trial_dir)

        config = {}
        config['name'] = self.name
        config['id'] = self.id
        config['motion_type'] = Motion_Type(self.motion_type)
        with open(os.path.join(trial_dir, f'{self.name}.ini'), 'w') as configfile:
            json.dump(config, configfile)

        for record in self.records:
            record.save_to_file(trial_dir, record.source_name)

    def load_trial(self, trial_dir: str):
        if not os.path.exists(trial_dir):
            raise FileNotFoundError(trial_dir)

        # trial directory and config file (.ini) should have same name
        config_path = os.path.join(
            trial_dir, f'{os.path.basename(trial_dir)}.ini')
        if not os.path.exists(config_path):
            raise FileNotFoundError(config_path)

        with open(config_path, "r") as config_file:
            config = json.load(config_file)
            self.name = config['name']
            self.id = config['id']
            self.motion_type = config['motion_type']

        record_files = [os.path.join(trial_dir, record_npz) for record_npz in os.listdir(
            trial_dir) if os.path.splitext(record_npz)[1] in [".npz", ".pt"]]
        for rec in record_files:
            new_rec = IMU_Record(self.name, self.id)
            new_rec.load(rec)
            self.records.append(new_rec)


class CaptureSession(object):
    def __init__(self, name: str = "") -> None:
        super().__init__()
        self.name = name
        self.id = uuid.uuid4().hex
        self.tracker_ids: List[List[str]] = []
        self.trials: List[Trial] = []

    async def capture_trial(self, name: str, capture_length: int, sample_rate: int, trackers: List[Tracker], motion_type: bool = Motion_Type.MISC):
        self.tracker_ids = list(set(self.tracker_ids).union(
            set((tracker.device.name, tracker.id) for tracker in trackers)))
        trial = Trial(name, motion_type)
        await trial.capture_data(capture_length, sample_rate, trackers)
        self.trials.append(trial)

    def save_session(self, project_dir: str):
        if not os.path.exists(project_dir):
            os.mkdir(project_dir)
            print(f"Created project directory at: {project_dir}")
        session_dir = os.path.join(project_dir, self.name)
        if not os.path.exists(session_dir):
            os.mkdir(session_dir)

        config = {}
        config['name'] = self.name
        config['id'] = self.id
        config['tracker_ids'] = list(self.tracker_ids)
        with open(os.path.join(session_dir, f'{self.name}.ini'), 'w') as configfile:
            json.dump(config, configfile)

        for trial in self.trials:
            trial.save_trial(session_dir)

    def load_session(self, session_dir: str):
        if not os.path.exists(session_dir):
            raise FileNotFoundError(session_dir)

        # session directory and config file (.ini) should have same name
        config_path = os.path.join(
            session_dir, f'{os.path.basename(session_dir)}.ini')
        if not os.path.exists(config_path):
            raise FileNotFoundError(config_path)

        with open(config_path, "r") as config_file:
            config = json.load(config_file)
            self.name = config['name']
            self.id = config['id']
            self.tracker_ids = config['tracker_ids']

        # subdirectories under session_dir are assumed to be those of trials
        subdirs = [os.path.join(session_dir, trial_dir) for trial_dir in os.listdir(
            session_dir) if os.path.isdir(os.path.join(session_dir, trial_dir))]
        for trial_path in subdirs:
            new_trial = Trial()
            new_trial.load_trial(trial_path)
            self.trials.append(new_trial)


class BaseStation(object):

    def __init__(self) -> None:
        super().__init__()
        self.discovered: List[BLEDevice] = []
        self.trackers: List[Tracker] = []

    async def find_devices(self):
        self.discovered = await BleakScanner.discover()

    async def connect_to(self, devices: List[Union[BLEDevice, str]]):
        async def connectionHandler(device: BLEDevice):
            tracker = Tracker(device)
            if await tracker.connect():
                print(f"SUCCESS: {tracker.device.name}")
                self.trackers.append(tracker)
            else:
                # handle connection failure
                print(f"FAILED : {tracker.device.name}")

        await asyncio.gather(*(connectionHandler(device) for device in devices))

    async def disconnect_all(self):
        await asyncio.gather(*(tracker.disconnect() for tracker in self.trackers))

    async def benchmark(self, sample_rate: int = 30):
        # test connection for 10 seconds, checking samples received
        capture_length = 10

        session = CaptureSession("ConnectionTest")
        await session.capture_trial("TestTrial", capture_length, sample_rate, self.trackers)

        print(f"Expected number of samples: {sample_rate * capture_length}")

        for record in session.trials[0].records:
            sample_count = len(record.timestamps)
            print(
                f"Samples received from {record.source_name}: {sample_count}")

    async def __aenter__(self):
        await self.find_devices()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.disconnect_all()
