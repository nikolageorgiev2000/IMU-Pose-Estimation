from __future__ import annotations
import time
import uuid

from src import *
from src.math3d.rotations import exp_rotmat, normalize
from src.math3d.kinematics import *

# for Cambridge, UK
default_grav_vec = FT([0, 0, -1]) * G_LOCAL / G_STANDARD
default_north_vec = exp_rotmat(
    FT([0, torch.deg2rad(FT([67.071])), 0])) @ FT([1, 0, 0])


class RefFrame(object):
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.xyz: FT = torch.stack((self.x, self.y, self.z))


class NavContext(RefFrame):
    def __init__(self, grav_vec: FT = default_grav_vec, north_vec: FT = default_north_vec):
        self.grav: FT = grav_vec
        self.north: FT = normalize(north_vec)
        z: FT = -normalize(grav_vec)
        y: FT = normalize(torch.cross(z, self.north))
        x: FT = normalize(torch.cross(y, z))
        super().__init__(x, y, z)
        self.dip_angle = torch.acos(torch.dot(self.x, self.north))
        self.dip_rotmat = exp_rotmat(self.dip_angle * self.y)

    def set_dip_angle(self, angle):
        self.dip_angle = angle
        self.north = self.get_dip_rot() @ self.x

    def get_dip_rot(self):
        return exp_rotmat(self.dip_angle * self.y)

    def set_g(self, g_coeff):
        self.grav = g_coeff * normalize(self.grav)


default_nav_context = NavContext()


class Poser(object):
    def __init__(self, nav: NavContext = default_nav_context):
        self.nav: NavContext = nav
        self.pos: FT = torch.empty((0, 3))
        self.vel: FT = torch.empty((0, 3))
        self.rot: FT = torch.empty((0, 3, 3))
        self.timestamps: torch.IntTensor = torch.empty((0, 1))

    def differentiate(self) -> PoseDerivs:
        pds = PoseDerivs()
        pds.intervals = torch.diff(self.timestamps, dim=0) / 1e6
        pds.a = diff_linear_velocity(self.vel, pds.intervals)
        pds.w = get_angular_velocity(self.rot, pds.intervals)
        return pds

    def generate_imu_record(self):
        pds = self.differentiate()
        sample_rate = 1/torch.mean(pds.intervals)
        rec = IMU_Record(Hz=sample_rate)
        rec.realStartTime = self.timestamps[0]
        rec.timestamps = self.timestamps
        rec.spf = get_specific_force(pds.a, self.rot[:-1], self.nav.grav)
        rec.gyro = pds.w
        rec.magno = ein('tij, j->ti', self.rot, self.nav.north)
        # extend with arbitrary value to match length of timestamps (intervals +1)
        rec.spf = torch.cat((rec.spf, torch.zeros((1, 3))))
        rec.gyro = torch.cat((rec.gyro, torch.zeros((1, 3))))
        # rec.magno = torch.cat((rec.magno, torch.zeros((1, 3))))
        return rec


class PoseDerivs(object):
    def __init__(self):
        self.intervals: FT = torch.empty((0, 1))
        self.a: FT = torch.empty((0, 3))
        self.w: FT = torch.empty((0, 3))

    def integrate(self, nav: NavContext, init_pos: FT, init_vel: FT, init_rot: FT, init_time: torch.IntTensor) -> Poser:
        poser = Poser(nav)

        poser.timestamps = torch.cat(
            (init_time[None, :], init_time + torch.cumsum(1e6 * self.intervals, dim=0)))

        poser.rot = integrate_angular_velocity(
            self.w, self.intervals, init_rot)

        poser.vel, poser.pos = integrate_linear_acceleration(
            self.a, self.intervals, init_vel, init_pos)

        return poser


class IMU_Record(object):

    def __init__(self, source_name: str = "", source_id: str = "", Hz: int = 30):
        self.source_name = source_name
        self.id = uuid.uuid4().hex
        self.source_id = source_id
        # keep time in microseconds
        self.realStartTime = FT([time.time_ns()/1000])
        self.timestamps = torch.empty((0, 1))
        self.spf = torch.empty((0, 3))
        self.gyro = torch.empty((0, 3))
        self.magno = torch.empty((0, 3))
        self.sample_rate = Hz

    def record_sample(self, byte_data: bytearray):
        # 32-bit unsigned long to signed int32
        timestamp = (torch.frombuffer(
            byte_data[-4:], dtype=torch.int)).type(FT)
        # 32-bit float to set dtype
        new_data = (torch.frombuffer(
            byte_data[:-4], dtype=torch.float)).type(FT)

        self.timestamps = torch.cat((self.timestamps, timestamp[None, :]))

        def swap_xy(vec):
            # yxz -> xyz
            vec[[0, 1]] = vec[[1, 0]]

        def invert_idx(vec, idx):
            # flip axis
            vec[idx] *= -1

        a = new_data[:3]
        swap_xy(a)
        # switch to radians (thus avoiding future conversions)
        g = new_data[3:6]
        g = torch.deg2rad(g)
        swap_xy(g)

        m = new_data[6:]
        swap_xy(m)
        invert_idx(m, 1)

        # convert to right-handed form
        self.spf = torch.cat((self.spf, a[None, :]))
        self.gyro = torch.cat((self.gyro, g[None, :]))
        self.magno = torch.cat((self.magno, m[None, :]))

    def save_to_file(self, save_path: str, base_name: str):
        # np.savez_compressed(os.path.join(save_path, filename), source_name=self.source_name, source_id=self.source_id, id=self.id,
        #                     realStartTime=self.realStartTime, timestamps=self.timestamps, accel=self.spf, gyro=self.gyro, magno=self.magno, sample_rate=self.sample_rate)

        data_dict = {
            'source_name': self.source_name, 'source_id': self.source_id, 'id': self.id,
            'realStartTime': self.realStartTime, 'timestamps': self.timestamps,
            'accel': self.spf, 'gyro': self.gyro, 'magno': self.magno,
            'sample_rate': self.sample_rate
        }
        full_name = os.path.join(save_path, base_name)+'.pt'
        torch.save(data_dict, full_name)

    def load(self, filename: str):
        _, ext = os.path.splitext(filename)
        if ext == '.pt':
            record_file = torch.load(filename)
            self.source_name = str(record_file['source_name'])
            self.source_id = str(record_file['source_id'])
            self.id = str(record_file['id'])
            self.realStartTime = record_file['realStartTime']
            self.timestamps = record_file['timestamps']
            self.spf = record_file['accel']
            self.gyro = record_file['gyro']
            self.magno = record_file['magno']
            self.sample_rate = int(record_file['sample_rate'])
        elif ext == '.npz':
            with np.load(filename) as record_file:
                self.source_name = str(record_file['source_name'])
                self.source_id = str(record_file['source_id'])
                self.id = str(record_file['id'])
                self.realStartTime = record_file['realStartTime']
                self.timestamps = torch.from_numpy(
                    record_file['timestamps'].astype(np.int32)).type(FT)[:, None]
                self.spf = torch.from_numpy(record_file['accel']).type(FT)
                self.gyro = torch.from_numpy(record_file['gyro']).type(FT)
                self.magno = torch.from_numpy(record_file['magno']).type(FT)
                self.sample_rate = int(record_file['sample_rate'])

    def generate_pose(self, nav: NavContext, init_pos: FT, init_vel: FT, init_rot: FT) -> Poser:
        pds = PoseDerivs()
        pds.intervals = torch.diff(self.timestamps, dim=0) / 1e6
        pds.w = self.gyro[:-1]
        rots = integrate_angular_velocity(
            self.gyro[:-1], pds.intervals, init_rot)
        pds.a = get_linear_acceleration(self.spf[:-1], rots[:-1], nav.grav)
        poser = pds.integrate(nav, init_pos, init_vel,
                              init_rot, FT([self.realStartTime]))
        return poser
