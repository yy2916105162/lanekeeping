from scipy . integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# values provided
v=5
L=2.3
u=2.*np.pi/180


#Define the system dynamics as function of the form f(t,z)
def dynamics(_t,z):
    theta = z[2]
    return [v*np.cos(theta), v*np.sin(theta), v*np.tan(u)/L]
t_final = 2
z_init = [0, 0.3, 5*np.pi/180]
num_points = 100  #resolution

#Simulate dynamical system
sol = solve_ivp(dynamics,
                [0, t_final],
                z_init,
                t_eval=np.linspace(0, t_final, num_points))

#solve_ivp returns the solution
#(t, x(t)) is stored in 'sol.t' and 'sol.y[0]'. Both are numpy arrays
times = sol.t
x_trajectory = sol.y[0]

#plot the figure of x(t) again time
plt . plot(times, x_trajectory.T)
plt.xlabel('Time (s)')
plt.ylabel('x (m)')
plt.title('Trajectory of x(t)')
plt.grid()
plt . show()