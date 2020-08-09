from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):

    def __init__(self, vehicle_mass, decel_limit, accel_limit, wheel_radius, wheel_base,
        steer_ratio, max_lat_accel,max_steer_angle):
        # TODO: Implement
        #Yew Controller object min speed = 0.1
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        #Udacity Provided experimented PID values
        kp = 0.3
        ki = 0.1
        kd = 0.
        mn = 0   # Minimum throttle value
        mx = 0.2 # Maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = 0.5 # Used to calculate cutoff frequency  = 1/2*PI*tau
        ts = 0.02 # Sample_time
        self.vel_lpf = LowPassFilter(tau, ts) #Remove high frequency noise from velocity
        self.last_time = rospy.get_time()

        self.vehicle_mass = vehicle_mass
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

    def control(self, dbw_enabled, current_vel, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        if dbw_enabled == 0:
            self.throttle_controller.reset()
            return 0., 0., 0. #When driver is taking control we dont want error to accumulate

        current_vel = self.vel_lpf.filt(current_vel)    #Filter with LPF

        #Get Steering Value
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        velocity_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        #Get throttle Value
        throttle = self.throttle_controller.step(velocity_error, sample_time)

        #Set Brake to not applied
        brake = 0

        if linear_vel == 0. and current_vel < 0.1: #Code to stop vehicle completely
            throttle = 0
            brake = 700   # 700 NM torque reuired to stop Carla from moving
        elif throttle < 0.1 and velocity_error < 0:
            throttle = 0
            decel = max(velocity_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius  #Brake torque Nm Deceleration * Mass * Wheel Radius

        return throttle, brake, steering
