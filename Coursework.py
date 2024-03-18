import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.sparse as sps
import scipy.sparse.linalg as LA

class Grid:
    '''Class defining a 2D computational grid.  The grid object
    contains is a regular cartesian grid with a single variable, u.
    It stores information about the number of grid points in the i
    and j directions, the coordinates of these points and the bottom
    left corner of the grid (the origin) and the top right corner
    (the extent).
    
    Written by Prof David Ingram, School of Engineering
    (c) 2022 The University of Edinburgh
    Licensed under CC-BY-NC.'''
    
    DIRICHLET_BC = 0
    NEUMANN_BC = 1

    BC_NAME = ['left', 'right', 'top', 'bottom']
    
    def __init__(self,ni,nj):
        # Set up information about the grid
        self.origin = (0.0, 0.0)  # bottom left
        self.extent = (1.0, 1.0)  # top right
        self.Ni = ni # grid points in i direction
        self.Nj = nj # grid points in j direction
        
        # Initialse x,y and u arrays
        self.u = np.zeros((nj, ni))
        self.x = np.zeros((nj, ni))
        self.y = np.zeros((nj, ni))
        
        # Boundary conditions (left right top and bottom)
        self.BC = [self.DIRICHLET_BC, self.DIRICHLET_BC, 
                   self.DIRICHLET_BC, self.DIRICHLET_BC]
        
        # Set the time
        self.time = 0.0
        
        # Initialize thermal diffusivity
        self.kappa = 1.0

    def set_origin(self,x0,y0):
        
        # Set origin of domain
        self.origin = (x0, y0)
    
    def set_extent(self,width,height):
        
        # Set extent of domain
        self.extent = (width, height)
        
    def generate(self,Quiet=True):
        '''generate a uniformly spaced grid covering the domain from the
        origin to the extent.  We are going to do this using linspace from
        numpy to create lists of x and y ordinates and then the meshgrid
        function to turn these into 2D arrays of grid point ordinates.'''
        x_ord = np.linspace(self.origin[0], self.origin[0]+self.extent[0], self.Ni)
        y_ord = np.linspace(self.origin[1], self.origin[1]+self.extent[1], self.Nj)
        self.x, self.y = np.meshgrid(x_ord,y_ord)
        if not Quiet:
            print(self)

    def set_diffusivity(self,k):
        self.kappa = k
        
    def Delta_x(self):
        # Calculate delta x
        dx = self.x[0,1]-self.x[0,0]
        
        return dx
    
    def Delta_y(self):
        # Calculate delta y
        dy = self.y[1,0]-self.y[0,0]
        
        return dy
    
    def find(self,point):
        '''find the i and j ordinates of the grid cell which contains 
        the point (x,y).  To do this we calculate the distance from
        the point to the origin in the x and y directions and then
        divide this by delta x and delta y.  The resulting real ordinates
        are converted to indices using the int() function.'''
        grid_x = (point[0] - self.origin[0])/self.Delta_x()
        grid_y = (point[1] - self.origin[1])/self.Delta_y()
        return int(grid_x), int(grid_y)
    
    def set_Neumann_bc(self,side):
        try:
             self.BC[self.BC_NAME.index(side)] = self.NEUMANN_BC
        except:
             print('error {} must be one of {}'.format(side,self.BC_NAME))
     
    def set_Dirichlet_bc(self,side):
        try:
            self.BC[self.BC_NAME.index(side)] = self.DIRICHLET_BC
        except:
            print('error {} must be one of {}'.format(side,self.BC_NAME))
    
    def plot(self,title):
        '''produce a contour plot of the solution at the current time'''
        
        # Create a figure title
        caption = f'{title}, t={self.time}s ({self.Ni} x {self.Nj}) grid.'
        
        # Create a figure and add a plot to it
        fig, ax1 = plt.subplots()
        
        # Draw the contour plot
        cmap = plt.get_cmap('jet')
        cf = ax1.contourf(self.x,self.y,self.u,cmap=cmap, levels = 21)
        
        # Add colorbar
        fig.colorbar(cf, ax=ax1)
        
        # Set titlea
        ax1.set_title(caption)
        return plt

     
    def report_BC(self):
        '''compile a string listing the boundary conditions on each side.
        We build up a string of four {side name}: {BC type} pairs and
        return it'''
        
        # Initialise the string
        string = ''
        
        # loop over the sides
        for side in range(4):
            # Add the side name
            string = string + self.BC_NAME[side]
            # And the boundary condition type
            if self.BC[side] == self.DIRICHLET_BC:
                string = string + ': Dirichlet, '
            elif self.BC[side] == self.NEUMANN_BC:
                string = string + ': Neumann, '
        return string[:-2] +'.' # Lose the last comma and space.
    
    def __str__(self):
        # Description of the object when asked to print it
        describe = 'Uniform {}x{} grid from {} to {}.'.format(self.Ni, self.Nj, self.origin, self.origin+self.extent)
        boundaries = self.report_BC()
        return describe + '\n The boundary conditions are - ' + boundaries +f'\ntime ={self.time:.4g}'
    
def set_grid(ni,nj):
    ''' This function sets up an ni x nj grid to solve the problem specified in the Coursework'''
    
    # Initialize the grid class with ni and nj grid points
    mesh = Grid(ni,nj)
    
    # Set origin and extent of grid
    mesh.set_origin(0, -np.pi)
    mesh.set_extent(np.pi * 0.5, 2*np.pi)
    
    # Set diffusivity
    mesh.set_diffusivity(0.2)
    
    # This generates the grid
    mesh.generate()
     
    # Now set the inital conditions
    mesh.u = np.sin(mesh.x)
    
    return mesh

def set_boundary_conditions(mesh,time):
    '''Function that sets the boundary conditions. This functions takes 
       as input a mesh (mesh) and a time step (time)'''
    
    # Set x = 0 boundary
    mesh.u[:,0] = time
    
    # Set x = 0.5pi boundary
    mesh.u[:,-1] = np.exp(-mesh.kappa * time)
    
    # Set y = -pi boundary
    mesh.u[0,:] = 0

    # Set y = pi boundary
    mesh.u[-1,:] = 2 * np.cos(mesh.x[-1,:]) * np.exp(-mesh.kappa * time)

# This creates a 121 by 121 grid
grid = set_grid(41, 41)

# Call the function to set up boundary conditions for the first time step (t=0)
set_boundary_conditions(grid, 0.0)

# Plot the initial solution matrix - Check that initial and boundary conditions are implemented correctly!
#grid.plot('Coursework')

def Crank_Nicholson(mesh,t_stop,Courant_no=5.0):
    '''Advance the solution from the current time to t=t_stop
    using the Crank-Nicholson method.
    
    mesh is an object of the grid class, 
    t_stop is the time at which the solution is required and 
    Courant_no is the Courant number to be used for the calculation.  
    While the Crank-Nicholson method is unconditionally stable, a low
    Courant number is recommended on grounds of accuracy.'''
    
    # Calculate delta_t based on the Courant Number 
    delta_t = Courant_no*min(mesh.Delta_x()**2, mesh.Delta_y()**2)/(2*mesh.kappa)
    
    # Calculate number of iterations based on stopping time t_stop and time step delta_t
    maxit = int((t_stop-mesh.time)/delta_t)
    
    # We set a variable of every how many iterations (time steps) to report
    out_it = maxit // 50
    
    print('Crank Nicholson with ∆t={:.4g} ({:d} times steps)'.format(delta_t,maxit))
    
    # Initialize the A matrix using the lil format (sparse matrix) and the b vector as a numpy vector.
    N = (grid.Nj-2)*(grid.Ni-2) # How many interior points our solution grid has
    A_mat = sps.lil_matrix((N, N), dtype=np.float64)
    b_vec = np.zeros(N, dtype=np.float64)
    
    # Initialize the solution vector
    x_vec = np.zeros(N, dtype=np.float64)
    

    # Calculate rx and ry coefficients
    rx = mesh.kappa * delta_t / mesh.Delta_x()**2
    ry = mesh.kappa * delta_t / mesh.Delta_y()**2

    # Now build the A and B matricies
    for i in range(1,mesh.Ni-1):
        for j in range(1,mesh.Nj-1):
            # We introduce index k
            k = (i-1) + (mesh.Ni-2)*(j-1)
                
            # Set the leading diagonal coefficient
            A_mat[k,k] = (1+rx+ry)
               
            # Set coefficients of neighbouring points in the i direction
            for m in range(i-1,i+2,2):
                if not(m<1 or m>mesh.Ni-2):
                    l = (m-1) + (mesh.Ni-2)*(j-1)
                    A_mat[k,l] = -rx/2

            # Set coefficients of neighbouring points in the j direction
            for m in range(j-1,j+2,2):
                if not(m<1 or m>mesh.Nj-2):
                    l = (i-1) + (mesh.Ni-2)*(m-1)
                    A_mat[k,l] = -ry/2
    
    # Assemble the preconditioner
    ilu = LA.spilu(A_mat.tocsc(), drop_tol=1e-6, fill_factor=100)
    M_mat = LA.LinearOperator(A_mat.shape, ilu.solve)

    # Set an iteration counter
    it = 0
    
    # Set a variable for the convergence status. This is set initially to 1, which means convergence not achieved. 
    # The LA.bicgstab will return status = 0 when we have successful convergence.
    status = 1
    
    # Start the time loop - Every iteration is a time step
    while mesh.time < t_stop:
        
        # Progress counter
        if it % out_it ==0:
            print('#',end='')
            
        # Calculate time step dt - Ensure we don't overshoot the stop time
        dt = min(delta_t, t_stop-mesh.time)
        
        # Set the boundary conditions
        set_boundary_conditions(mesh, mesh.time)
        
        # Extract the x_vector - We take the interior points of the 2D solution array (mesh.u[1:-1,1:-1]) and we reshape
        # that into a vector (1D array) with N elements
        x_vec = np.reshape(mesh.u[1:-1,1:-1],(N))

        # Calculate the b vector (RHS) using the values from the current time step
        for i in range(1,mesh.Ni-1):
            for j in range(1,mesh.Nj-1):
                k = (i-1) + (mesh.Ni-2)*(j-1)
                
                # Calculate b(k) from the stencil for the current time step
                b_vec[k] = (rx/2) * mesh.u[j, i+1] + (ry/2) * mesh.u[j+1, i] + (1 - rx - ry) * mesh.u[j, i] + (rx/2) * mesh.u[j, i-1] + (ry/2) * mesh.u[j-1, i]
 
        # Now apply the boundary conditions for the next time level
        set_boundary_conditions(mesh, mesh.time+dt)
        
        # Update the b vector to include boundary conditions
        for i in range(1,mesh.Ni-1):
            for j in range(1,mesh.Nj-1):
                k = (i-1) + (mesh.Ni-2)*(j-1)
                
                # i direction
                for m in range(i-1,i+2,2):
                    if m<1 or m>mesh.Ni-2:
                        b_vec[k] += rx/2 * mesh.u[j,m]

                # j direction
                for m in range(j-1,j+2,2):
                    if m<1 or m>mesh.Nj-2:
                        b_vec[k] += ry/2 * mesh.u[m,i]
                        
        # Solve the matrix system Ax=b using the preconditioner M
        x_vec, status = LA.bicgstab(A_mat, b_vec, x0=x_vec, M=M_mat)
        
        # If status=0 is returned then we have convergence. If not, convergence not achieved and we break the loop
        if status==0:
            # Out solution is in the format of a vector (x_vec - 1D array). We need to unpack this into a 2D grid (2D array)
            mesh.u[1:-1,1:-1] = np.reshape(x_vec, (mesh.Ni-2,mesh.Nj-2))
        if status != 0:
            break
        
        # Update time step and iteration number for next step
        mesh.time += dt
        it += 1
    
    print('.')
    if status == 0:
        return it
    else:
        return -status

it = Crank_Nicholson(grid, 5)

# Prints how many iterations were performed
print(it,'iterations completed.')

# Plots solution
fig = grid.plot('Coursework')
fig.show()