#Author: Alex Kult
#Description: Create the geometry of a rocket nozzle using the method of characteristics
#Date: 5-14-2025

import MOC_lib as moc
import numpy as np
import matplotlib.pyplot as plt

## Inputs
gamma = 1.16 #ratio of specific heats
mach_e = 2.6541 #exit Mach number
n = 10 #number of characteristic lines
theta_min = 0.000001 #kickoff angle
throat_mach = 1.00000001 #kickoff mach number
throat_rad = 1 #radius of throat [in]
mesh_type = "average" #"leading" is calculated from angles of prior points, "average" is calculated from average of prior and current points
interpolation = True #Make a function from nozzle wall points

## Initialziing Arrays
x_wall_lst = [0]
y_wall_lst = [throat_rad]

char_points = -np.ones((n,n+1,2))
nu_map = -np.ones((n,n+1))
mach_map = -np.ones((n,n+1))
mu_map = -np.ones((n,n+1))
theta_map = -np.ones((n,n+1))

#Max expansion angle
theta_max = 0.5*moc.prandtl_meyer(gamma, mach_e) #radians

initial_theta_lst = np.linspace(theta_min, theta_max, n) #Interpolated thetas
initial_slope_lst = initial_theta_lst - moc.mach_angle(throat_mach) #Slopes of characteristics from throat

#First negative characteristic line
k_minus_lst = initial_theta_lst + moc.prandtl_meyer(gamma, throat_mach)
k_plus_lst = -k_minus_lst
nu_1 = k_minus_lst[0]
nu_map[0,0] = nu_1
mach_1 = moc.invert_prandtl_meyer_angle(gamma, nu_1)
mach_map[0,0] = mach_1
mu_1 = moc.mach_angle(mach_1)
mu_map[0,0] = mu_1

#Finding xy coordinates of first point on centerline
m_1 = np.tan(initial_slope_lst[0])
x_1 = y_wall_lst[0]/(-m_1)
char_points[0,0,0] = x_1
char_points[0,0,1] = 0

#First positive characteristic Line
theta_map[:,0] = 0

for i in range(n - 1):
    #Finding Invariants
    k_minus = k_minus_lst[i+1]
    k_plus = k_plus_lst[0]
    
    #Calculating point parameters
    nu = 0.5*(k_minus - k_plus)
    theta = 0.5*(k_minus + k_plus)
    mach = moc.invert_prandtl_meyer_angle(gamma, nu)
    mu = moc.mach_angle(mach)
    
    #Updating arrays
    nu_map[0,i+1] = nu
    theta_map[0,i+1] = theta
    mach_map[0,i+1] = mach
    mu_map[0,i+1] = mu
    
    #Calculate slope from reference points
    if mesh_type == "leading":
        m_1 = np.tan(initial_slope_lst[i+1])
        m_2 = np.tan(theta_map[0,i] + mu_map[0,i])
    elif mesh_type == "average":
        m_1 = np.tan(0.5*(initial_slope_lst[i+1] + theta - mu))
        m_2 = np.tan(0.5*(theta_map[0,i] + mu_map[0,i] + theta + mu))
    else:
        raise ValueError("Please select a valid mesh type")
    
    #Calculate new point location
    x = (y_wall_lst[0] - char_points[0,i,1] + m_2*char_points[0,i,0] - m_1*x_wall_lst[0]) / (m_2 - m_1)
    y = y_wall_lst[0] + m_1*(x - x_wall_lst[0])
    
    #Update coordinate array
    char_points[0,i+1,0] = x
    char_points[0,i+1,1] = y

#Calculating parameters of first wall point
wall_ang_1 = initial_theta_lst[-1]
theta = theta_map[0,n-1]
nu = nu_map[0,n-1]
mach = moc.invert_prandtl_meyer_angle(gamma, nu)
mu = moc.mach_angle(mach)

#Updating arrays
nu_map[0,n] = nu
theta_map[0,n] = theta
mach_map[0,n] = mach
mu_map[0,n] = mu

#Calculating slope from reference points
m_1 = np.tan(wall_ang_1) #ensures initial expansion angle is correct
m_2 = np.tan(theta + mu)

#Calculating wall point location
x = (y_wall_lst[0] - char_points[0,-2,1] + m_2*char_points[0,-2,0] - m_1*x_wall_lst[0]) / (m_2 - m_1)
y = y_wall_lst[0] + m_1*(x - x_wall_lst[0])

#Updating coordinate array
x_wall_lst.append(x)
y_wall_lst.append(y)
char_points[0,-1,0] = x
char_points[0,-1,1] = y

## Calculating characteristics 2 through n
for i in range(n-1):
    #Finding Invariants
    k_minus = k_minus_lst[i+1]
    k_plus = k_plus_lst[i+1]
    
    #Calculating centerline point parameters
    nu_1 = k_minus
    nu_map[i+1,0] = nu_1
    mach_1 = moc.invert_prandtl_meyer_angle(gamma, nu_1)
    mach_map[i+1,0] = mach_1
    mu_1 = moc.mach_angle(mach_1)
    mu_map[i+1,0] = mu_1

    #Finding xy coordinates of centerline point
    theta = theta_map[i,1]
    mu = mu_map[i,1]
    m_1 = np.tan(theta - mu)

    x_0 = char_points[i,1,0]
    y_0 = char_points[i,1,1]
    x_1 = y_0/(-m_1) + x_0
    
    #Updating coordinate array
    char_points[i+1,0,0] = x_1
    char_points[i+1,0,1] = 0
    
    #Calculating interior points on characteristic line
    for j in range(n - i - 2):
        #Updating left running invariant
        k_minus = k_minus_lst[i+j+2]
        
        #Calculating point parameters
        nu = 0.5*(k_minus - k_plus)
        theta = 0.5*(k_minus + k_plus)
        mach = moc.invert_prandtl_meyer_angle(gamma, nu)
        mu = moc.mach_angle(mach)
        
        #Updating arrays
        nu_map[i+1,j+1] = nu
        theta_map[i+1,j+1] = theta
        mach_map[i+1,j+1] = mach
        mu_map[i+1,j+1] = mu
        
        #Calculating slope from reference points
        if mesh_type == "leading":
            m_1 = np.tan(theta_map[i,j+2] - mu_map[i,j+2])
            m_2 = np.tan(theta_map[i+1,j] + mu_map[i+1,j])
        elif mesh_type == "average":
            m_1 = np.tan(0.5*(theta_map[i,j+2] - mu_map[i,j+2] + theta - mu))
            m_2 = np.tan(0.5*(theta_map[i+1,j] + mu_map[i+1,j] + theta + mu))
        else:
            raise ValueError("Please select a valid mesh type")
        
        #Calculating new point location
        x = (char_points[i,j+2,1] - char_points[i+1,j,1] + m_2*char_points[i+1,j,0] - m_1*char_points[i,j+2,0]) / (m_2 - m_1)
        y = char_points[i,j+2,1] + m_1*(x - char_points[i,j+2,0])
        
        #Updating coordinate array
        char_points[i+1,j+1,0] = x
        char_points[i+1,j+1,1] = y

    #Finding wall point of characteristic
    wall_ang = initial_theta_lst[-i-2]
    theta = theta_map[i+1,n-i-2]
    nu = nu_map[i+1,n-i-2]
    mach = moc.invert_prandtl_meyer_angle(gamma, nu)
    mu = moc.mach_angle(mach)
    
    #Updating arrays
    nu_map[i+1,n-i-1] = nu
    theta_map[i+1,n-i-1] = theta
    mach_map[i+1,n-i-1] = mach
    mu_map[i+1,n-i-1] = mu

    #Calculating slopes from reference points
    if mesh_type == "leading":
        m_1 = np.tan(wall_ang)
    elif mesh_type == "average":
        m_1 = np.tan(0.5*(wall_ang + theta))
    else:
        raise ValueError("Please select a valid mesh type")
    m_2 = np.tan(theta + mu)

    #Calculating new point location
    x = (char_points[i,-i-1,1] - char_points[i+1,-i-3,1] + m_2*char_points[i+1,-i-3,0] - m_1*char_points[i,-i-1,0]) / (m_2 - m_1)
    y = char_points[i,-i-1,1] + m_1*(x - char_points[i,-i-1,0])

    #Updating coordinate array
    x_wall_lst.append(x)
    y_wall_lst.append(y)
    char_points[i+1,-i-2,0] = x
    char_points[i+1,-i-2,1] = y

#Plotting wall points
plt.plot(x_wall_lst, y_wall_lst)

#Plotting left running characteristics
for k in range(n):
    x_points_l = [0]
    y_points_l = [throat_rad]
    for i in range(n):
        for j in range(n+1):
            if i + j == k:
                x = char_points[i,j,0]
                y = char_points[i,j,1]
                
                x_points_l.append(x)
                y_points_l.append(y)
    plt.plot(x_points_l, y_points_l)

#Plotting right running characteristics               
for i in range(n):
    x_points_r = []
    y_points_r = []
    
    for j in range(n+1):
        if char_points[i,j,0] != -1:
            x = char_points[i,j,0]
            y = char_points[i,j,1]
            # plt.scatter(x,y)
            x_points_r.append(x)
            y_points_r.append(y)
    plt.plot(x_points_r,y_points_r)

##Output
area_ratio = char_points[n-1,1,1]**2/throat_rad**2

print(f"Maximum expansion angle: {np.degrees(theta_max)} [deg]")
print(f"Area ratio: {area_ratio}")
print(f"Nozzle length: {char_points[n-1,1,0]} [in]")

#Nozzle Plot
plt.xlabel("Nozzle Length [in]")
plt.ylabel("Nozzle Height [in]")
plt.title(f"MOC Nozzle for Gamma: {gamma} and Exit Mach: {mach_e}")
plt.show()

##Interpolation
if interpolation:
    from scipy.interpolate import CubicSpline
    f_interp = CubicSpline(x_wall_lst, y_wall_lst)
    x_int = np.linspace(min(x_wall_lst), max(x_wall_lst), n+1)
    y_int = f_interp(x_int)
    plt.plot(x_wall_lst, y_wall_lst)
    plt.plot(x_int, y_int)
    plt.xlabel("Nozzle Length [in]")
    plt.ylabel("Nozzle Height [in]")
    plt.title("Cubic Interpolation")
    plt.show()

#Creating Points in CSV for Fusion Import (Use ImportSpline)
if interpolation:
    x_pts = 2.54*x_int.reshape(-1,1)
    y_pts = 2.54*y_int.reshape(-1,1)
else:
    x_pts = 2.54*np.array(x_wall_lst).reshape(-1,1)
    y_pts = 2.54*np.array(y_wall_lst).reshape(-1,1)

z_pts = np.zeros((n+1,1))
wall_pts = np.hstack((x_pts, y_pts, z_pts))
np.savetxt("moc_nozzle_pts.csv", wall_pts, delimiter=",", fmt="%.3f", comments="")