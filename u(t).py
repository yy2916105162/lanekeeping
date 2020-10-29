from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


class Car:
    def __init__(self, length=2.3, velocity=5, x=0, y=0, pose=0):
        self.__length = length
        self.__velocity = velocity
        self.__x = x
        self.__y = y
        self.__pose = pose

    def move(self, steering_angle, dt):
        # simulate the motion(trajectory) of the car
        def bicycle_model(t, z):
            theta = z[2]
            return [self.__velocity * np.cos(theta),
                    self.__velocity * np.sin(theta),
                    self.__velocity * np.tan((steering_angle + 1*np.pi/180) / self.__length)]

        z_initial = [self.__x, self.__y, self.__pose]
        solution = solve_ivp(bicycle_model,
                             [0, dt],
                             z_initial)
        self.__x = solution.y[0, -1]
        self.__y = solution.y[1, -1]
        self.__pose = solution.y[2, -1]

    # return the solution of each variable
    def x(self):
        return self.__x

    def y(self):
        return self.__y

    def pose(self):
        return self._pose


class PIDControler:
    """define a structure for PidController"""

    def __init__(self, kp, ki, kd, ts):
        """parameters of PidController
        :param kp: proportional gain
        :param ki: integral gain
        :param kd: derivative gain
        :param ts: discrete time
        """
        self.__kp = kp       # discrete_time Kp
        self.__kd = kd / ts  # discrete-time Kd
        self.__ki = ki * ts  # discrete-time Ki
        self.__previous_error = None  # error in previous discrete time
        self.__error_sum = 0.  # sum of errors

    def control(self, y, y_set_point=0.):
        """Constructor for PIDControler

        :param y: position of car
        :param set_point: targeted position
        :return: steering angle for control
        """
        # P controller
        error = y_set_point - y  # compute the control error
        control_action = self.__kp * error  # compute the proportional component
        self.__error_sum += error  # accumulation for errors

        # D component
        if self.__previous_error is not None:
            error_diff = error - self.__previous_error   # compute the error difference of two neighboring slots
            control_action += self.__kd * error_diff   # add the differential component
        self.__previous_error = error

        # I component
        control_action += self.__ki * self.__error_sum  # add the integration component
        return control_action  # return the value of sum of all three components


t_sampling = 0.025  # value of each discrete time
num_points = 2000


pid = PIDControler(kp=0.4, ki=0.5, kd=1., ts=t_sampling)
car = Car(x=0, y=0.3, pose=5 * np.pi / 180)
u_cache = np.arange(0,0) # values for u(t)


for k in range(num_points):
    control_action = pid.control(car.y())
    u_cache = np.append(u_cache,control_action) # insert u(t) in to array

time_span = t_sampling * np.arange(num_points) # time span in the axis
plt.plot(time_span,u_cache)
plt.xlabel('t (s)')
plt.ylabel('u (t)')
plt.title('Plot of u(t)')
plt.grid()
plt.show()
