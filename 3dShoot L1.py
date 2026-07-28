import numpy as np
import matplotlib.pyplot as plt

mu = 0.01215
tol = 1e-12
dt = 1e-4
def f(S):

    x, y, z, vx, vy, vz = S[0:6]
    stm = S[6:].reshape((6, 6))

    r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)

    vxdot = 2 * vy + x - (1-mu) * (x + mu) / r1 ** 3 - mu * (x - 1 + mu) / r2 ** 3
    vydot = -2 * vx + y - (1-mu) * y / r1 ** 3 - mu * y / r2 ** 3
    vzdot = - (1-mu) * z / r1 ** 3 - mu * z / r2 ** 3

    omegaxx = 1 - (1-mu) / r1 ** 3 - mu / r2 ** 3 + 3 * (1-mu) * (x + mu) ** 2 / r1 ** 5 + 3 * mu * (x - 1 + mu) ** 2 / r2 ** 5
    omegayy = 1 - (1-mu) / r1 ** 3 - mu / r2 ** 3 + 3 * (1-mu) * y ** 2 / r1 ** 5 + 3 * mu * y ** 2 / r2 ** 5
    omegazz = -(1-mu) / r1 ** 3 - mu / r2 ** 3 + 3 * (1-mu) * z ** 2 / r1 ** 5 + 3 * mu * z ** 2 / r2 ** 5
    omegaxy = 3 * (1-mu) * (x + mu) * y / r1 ** 5 + 3 * mu * (x - 1 + mu) * y / r2 ** 5
    omegaxz = 3 * (1-mu) * (x + mu) * z / r1 ** 5 + 3 * mu * (x - 1 + mu) * z / r2 ** 5
    omegayz = 3 * (1-mu) * y * z / r1 ** 5 + 3 * mu * y * z / r2 ** 5

    stmdot = np.zeros((6, 6))
    stmdot[0:3] = 0
    stmdot[0, 3] = 1
    stmdot[1, 4] = 1
    stmdot[2, 5] = 1
    stmdot[3, 0] = omegaxx
    stmdot[3, 1] = omegaxy
    stmdot[3, 2] = omegaxz
    stmdot[4, 0] = omegaxy
    stmdot[4, 1] = omegayy
    stmdot[4, 2] = omegayz
    stmdot[5, 0] = omegaxz
    stmdot[5, 1] = omegayz
    stmdot[5, 2] = omegazz
    stmdot[3, 4] = 2
    stmdot[4, 3] = -2

    Sdot = np.zeros(42)
    Sdot[0:6] = [vx, vy, vz, vxdot, vydot, vzdot]
    Sdot[6:] = (stmdot @ stm).flatten()

    return Sdot

def rkf45(S, dt):

    f1 = f(S)
    f2 = f(S + (1/4) * dt * f1)
    f3 = f(S + (3/32) * dt * f1 + (9/32) * dt * f2)
    f4 = f(S + (1932/2197) * dt * f1 - (7200/2197) * dt * f2 + (7296/2197) * dt * f3)
    f5 = f(S + (439/216) * dt * f1 - 8 * dt * f2 + (3680/513) * dt * f3 - (845/4104) * dt * f4)
    f6 = f(S - (8/27) * dt * f1 + 2 * dt * f2 - (3544/2565) * dt * f3 + (1859/4104) * dt * f4 - (11/40) * dt * f5)

    S4 = S + dt * (25/216 * f1 + 1408/2565 * f3 + 2197/4104 * f4 - 1/5 * f5)
    S5 = S + dt * (16/135 * f1 + 6656/12825 * f3 + 28561/56430 * f4 - 9/50 * f5 + 2/55 * f6)
    
    error = max(np.abs(S5 - S4))

    if error > 1e-12:
        dt = dt * (1e-12 / error) ** 0.25
        return rkf45(S, dt)

    return S5

def sim(S_0):
    S = S_0.copy()
    signchange = False
    while signchange == False:
        S = rkf45(S, dt)
        if S[1] < -1e-20:
            signchange = True
    while abs(S[1]) > 1e-12:
        S = rkf45(S, -S[1] / S[4])
    
    return S

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def plot_orbit(S_0, dt_step=1e-4):
    """
    Propagates the converged initial state for one full period 
    and plots the resulting 3D orbit, ensuring both bodies are visible.
    """
    S = S_0.copy()
    
    # Lists to store the trajectory history
    x_vals = [S[0]]
    y_vals = [S[1]]
    z_vals = [S[2]]
    
    # Take one step to get off the y=0 plane before checking for crossings
    S = rkf45(S, dt_step)
    x_vals.append(S[0])
    y_vals.append(S[1])
    z_vals.append(S[2])
    
    crossings = 0
    
    # A full periodic orbit crosses the y=0 plane twice
    while crossings < 2:
        y_old = S[1]
        S = rkf45(S, dt_step)
        
        x_vals.append(S[0])
        y_vals.append(S[1])
        z_vals.append(S[2])
        
        # Detect if we crossed the y=0 plane (y changes sign)
        if y_old * S[1] < 0:
            crossings += 1


    
    # Plot the trajectory
    ax.plot(x_vals, y_vals, z_vals, color='blue', linewidth=1)

    
    # Plot the PRIMARY body (e.g., Earth) at (-mu, 0, 0)
    ax.scatter([-mu], [0], [0], color='orange', s=200, zorder=4)
    
    # Plot the SECONDARY body (e.g., the Moon) at (1-mu, 0, 0)
    ax.scatter([1 - mu], [0], [0], color='gray', s=100, zorder=4)
    
    # Calculate limits to ensure everything is visible
    max_x = max(max(x_vals), 1 - mu + 0.2)
    min_x = min(min(x_vals), -mu - 0.2)
    max_range = max_x - min_x
    
    # Apply the limits
    ax.set_xlim3d(min_x, max_x)
    ax.set_ylim3d(-max_range/2, max_range/2)
    ax.set_zlim3d(-max_range/2, max_range/2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Periodic Orbit in the CR3BP')
    ax.grid(True)

vyguess = .1
zguess = 0.0

xinit = .8
initial = np.array([xinit, 0, zguess, 0, vyguess, 0,
                         1, 0, 0, 0, 0, 0,
                         0, 1, 0, 0, 0, 0,
                         0, 0, 1, 0, 0, 0,
                         0, 0, 0, 1, 0, 0,
                         0, 0, 0, 0, 1, 0,
                         0, 0, 0, 0, 0, 1])


for i in range(100):
        initial = np.array([xinit, 0, zguess, 0, vyguess, 0,
                         1, 0, 0, 0, 0, 0,
                         0, 1, 0, 0, 0, 0,
                         0, 0, 1, 0, 0, 0,
                         0, 0, 0, 1, 0, 0,
                         0, 0, 0, 0, 1, 0,
                         0, 0, 0, 0, 0, 1])

        Sf = sim(initial)

        E = np.array([Sf[3], Sf[5]])
        error = np.linalg.norm(E)
        if error < tol:
            print(f"Converged at {i} iterations")
            print (initial[0:7])
            plot_orbit(initial, dt)
        
            break



        xddot = f(Sf)[3]
        zddot = f(Sf)[5]

        J = np.zeros((2, 2))
        J[0, 0] = Sf[26] - xddot / Sf[4] * Sf[14]
        J[0, 1] = Sf[28] - xddot / Sf[4] * Sf[16]
        J[1, 0] = Sf[38] - zddot / Sf[4] * Sf[14]
        J[1, 1] = Sf[40] - zddot / Sf[4] * Sf[16]

        Jinv = np.linalg.inv(J)
        U = - Jinv @ E

        vyguess += U[1]
        zguess += U[0]

    
plt.show()