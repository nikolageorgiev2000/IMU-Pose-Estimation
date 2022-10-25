from __future__ import annotations

from src import *
from src.data_collection.posers import *
from src.data_collection.capture_suite import *

from src.data_collection.capture_suite import Trial, Motion_Type
from src.data_collection.posers import IMU_Record, NavContext
from src.math3d import linalg
from src.calibration.accel_dip_calibrator import DipCalibrator
from src.calibration.basic_gyro_calibrator import Gyro_Calibrator
from src.calibration.magno_calibrator import fit_ellipsoid

class IMU_Model(object):

    def __init__(self, source_name: str = "", source_id: str = "") -> None:
        self.source_name = source_name
        self.source_id = source_id

    def autofit(self, input_rec: IMU_Record, nav: NavContext) -> None:

        calib_rec = copy.deepcopy(input_rec)

        T = len(calib_rec.timestamps)

        # Magnetometer calib

        self.m_inve, self.m_offset = fit_ellipsoid(calib_rec.magno)

        calib_rec.magno = self.calibrate_data(calib_rec.magno, self.m_inve, self.m_offset)
        calib_rec.magno = normalize(calib_rec.magno)

        # Identify Gravity Vectors

        inds = torch.zeros(T)

        # expect a tenth of samples gravity vectors
        window_range = (FT([1/3, 1/2]) * calib_rec.sample_rate).int()
        for w in range(*window_range, 2):
            half_window_size = w // 2  # dividend assumed odd for clear midpoint
            new_inds = torch.logical_and(linalg.moving_covariance_threshold(
                calib_rec.spf, half_window_size, 3.0), linalg.moving_covariance_threshold(
                calib_rec.magno, half_window_size, 3.0))
            if new_inds.sum() > inds.sum():
                inds = new_inds
                print(w, sum(inds))

        self.grav_inds = inds

        # Accelerometer calib

        uncal_still_accels = calib_rec.spf[self.grav_inds]
        cal_still_magnos = calib_rec.magno[self.grav_inds]

        Sa, ba = fit_ellipsoid(normalize(uncal_still_accels))
        print(Sa, ba)
        init_state = [Sa, ba, FT([-1])]
        grav = G_LOCAL / G_STANDARD
        af = DipCalibrator(init_state, [uncal_still_accels, grav, cal_still_magnos], hyperparams=[
                              1, 11, 1/9, 1e10, 1e-10, 10000], verbose=True)
        af.fit()
        self.a_inve, self.a_offset, dip_cos = af.state

        self.dip_angle = torch.acos(dip_cos) - torch.pi/2

        calib_rec.spf = self.calibrate_data(calib_rec.spf, self.a_inve, self.a_offset)

        # Gyroscope calib
        
        dt = torch.diff(calib_rec.timestamps, dim=0) / 1e6
        sample_count = len(dt)
        rot_vecs = calib_rec.gyro[:sample_count]
        magnos = normalize(calib_rec.magno)

        self.g_simple_offset = torch.mean(
            calib_rec.gyro[self.grav_inds], dim=0)
        init_state = torch.eye(3)
        self.g_offset = self.g_simple_offset
        gf = Gyro_Calibrator(init_state, [dt, rot_vecs, magnos, self.g_offset], hyperparams=[
                             1, 11, 1/9, 1e10, 1e-10, 10001], verbose=True)
        gf.fit()
        self.g_inve = gf.state

        calib_rec.gyro = self.calibrate_data(calib_rec.gyro, self.g_inve, self.g_offset)

        return calib_rec


    def calibrate_record(self, record: IMU_Record) -> IMU_Record:
        new_rec = copy.deepcopy(record)
        
        # calibrate the new record
        new_rec.spf = self.calibrate_data(new_rec.spf, self.a_inve, self.a_offset)
        
        new_rec.gyro = self.calibrate_data(new_rec.gyro, self.g_inve, self.g_offset)

        new_rec.magno = self.calibrate_data(new_rec.magno, self.m_inve, self.m_offset)
        new_rec.magno = normalize(new_rec.magno)
        return new_rec

    def calibrate_data(self, data, Sinv, b):
        return ein('ij,tj->ti', Sinv, (data - b))

