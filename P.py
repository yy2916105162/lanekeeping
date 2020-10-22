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
        """Constructor for PIDController

        :param y: position of car
        :param set_point: targeted position
        :return: steering angle for control
        """
        # P controller
        error = y_set_point - y  # compute the control error
        control_action = self.__kp * error  # add the proportional component
        return control_action  # return proportional component as steering angle value


t_sampling = 0.025  # value of each discrete time
num_points = 2000

# Cars under same conditions and
# P Controllers with different proportional gains.
pid_1 = PIDControler(kp=0.4, ki=0.0, kd=0.0, ts=t_sampling)
pid_2 = PIDControler(kp=0.8, ki=0.0, kd=0.0, ts=t_sampling)
pid_3 = PIDControler(kp=1.2, ki=0.0, kd=0.0, ts=t_sampling)
car_1 = Car(x=0, y=0.3, pose=5 * np.pi / 180)
car_2 = Car(x=0, y=0.3, pose=5 * np.pi / 180)
car_3 = Car(x=0, y=0.3, pose=5 * np.pi / 180)

# insert thr current (first) value of x and y
# into the caches.
y_cache_1 = np.array([car_1.y()])
x_cache_1 = np.array([car_1.x()])
y_cache_2 = np.array([car_2.y()])
x_cache_2 = np.array([car_2.x()])
y_cache_3 = np.array([car_3.y()])
x_cache_3 = np.array([car_3.x()])

# record values of x and y in each discrete time then plot them
# repeat this process by implementing different P Controllers
for k in range(num_points):
    control_action = pid_1.control(car_1.y())
    car_1.move(control_action, t_sampling)
    y_cache_1 = np.append(y_cache_1, car_1.y())
    x_cache_1 = np.append(x_cache_1, car_1.x())

plt.plot(x_cache_1, y_cache_1, 'r')

for k in range(num_points):
    control_action = pid_2.control(car_2.y())
    car_2.move(control_action, t_sampling)
    y_cache_2 = np.append(y_cache_2, car_2.y())
    x_cache_2 = np.append(x_cache_2, car_2.x())

plt.plot(x_cache_2, y_cache_2, 'y')

for k in range(num_points):
    control_action = pid_3.control(car_3.y())
    car_3.move(control_action, t_sampling)
    y_cache_3 = np.append(y_cache_3, car_3.y())
    x_cache_3 = np.append(x_cache_3, car_3.x())

plt.plot(x_cache_3, y_cache_3, 'k')

# add appropriate labels for the plot
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Trajectories of (x, y) for P Controllers with different Kp')
plt.grid()
plt.show()