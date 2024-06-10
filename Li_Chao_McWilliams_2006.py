import numpy as np 
from scipy.optimize import minimize
from numba import njit, prange, uint32
from utils.workerside.njit_options import opts, parallel

"""
To do:
    - Make continuous or closed N/S/W/E-boundary-condition an optional argument.
    - Implement Arakawa C-grid instead of B-grid.
Other than that this code follows the paper.

For each computation of 'Ax':
    dPsi/dx, dPsi/dy, dChi/dx & dChi/dy are linearly interpolated onto 'y'-grid.

 A*x        =  [ -d/dy, d/dx ][ Psi ]
               [  d/dx, d/dy ][ Chi ]
            =  [ -dPsi/dy + dChi/dx ]
               [  dPsi/xy + dChi/dy ]
            =  [ u_Psi + u_Chi ]
               [ v_Psi + v_Chi ]
            =  [ u ]
               [ v ]

For each computation of 'A^T*(y-Ax)':
    d(y_u-Ax_u)/dx, d(y_u-Ax_u)/dy, d(y_v-Ax_v)/dx & d(y_v-Ax_v)/dy are linearly interpolated onto 'x'-grid.
    And set to constant zero on North/South boundary.

 A^T*(y-Ax) =  [ -d/dy, d/dx ][ u_err ]
               [  d/dx, d/dy ][ v_err ]
            =  [ -duerr/dy + dverr/dx ]
               [  duerr/dx + dverr/dy ]
"""

def optimize_psi_chi( 
    U, V, lat, lon, init_psi, init_chi, alpha, 
    gtol=1e-15, disp=False, wraparound=True, run=True
):
    """
    Arguments:
        U             2darray   [m^1 s^-1]  - time-mean of zonal velocity-timeseries
        V             2darray   [m^1 s^-1]  - time-mean of meridional velocity-timeseries
        lat   m-sized 1darray   [deg North] - gridpoints degrees latitude
        lon   n-sized 1darray   [deg East]  - gridpoints degrees longitude
        init_psi      2darray   [m^2 s^-1]  - initial value for Psi pre-optimization
        init_chi      2darray   [m^2 s^-1]  - initial value for Chi pre-optimization
        alpha         float     [unitless]  - Tikhonov's regularization parameter
        gtol          float                 - (See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
        disp          bool                  - (See https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
        wraparound    bool                  - if True: Velocities on -180.0 degree longitude are concatenated onto "east-end" of array s.t. n = n + 1
    Return:
        psi:          2darray   [m^2 s^-1]  - Computed streamfunction
        chi:          2darray   [m^2 s^-1]  - Computed velocity-potential
    """
    
    m, n = U.shape
    
    if wraparound:
        U = np.concatenate( (U, U[:,0].reshape(m,1)), axis=1 )
        V = np.concatenate( (V, V[:,0].reshape(m,1)), axis=1 )
        init_psi = np.concatenate( 
            (init_psi, init_psi[:,0].reshape(m+1,1)), axis=1
        )
        init_chi = np.concatenate(
            (init_chi, init_chi[:,0].reshape(m+1,1)), axis=1
        )
        _, n = U.shape
    elif not wraparound:
        raise ValueError("Code only works on global arrays atm.")

    # Build initial x
    x = np.empty((2,m+1,n+1))
    x[0,:,:] = init_psi
    x[1,:,:] = init_chi
    x = x.reshape(2*(m+1)*(n+1)).copy()
    
    # Build y
    y3D = np.empty((2,m,n))
    y3D[0,:,:] = U
    y3D[1,:,:] = V
    inn3D = ~np.isnan(y3D)       # is-not-NaN
    inn3Dsize = np.int32(inn3D.sum())
    y3D[np.isnan(y3D)] = 0.0

    # Define dx & dy
    Rearth = 6.371e6
    C = np.pi * Rearth / 180.0   # ~ meters per degree latitude
    dx = np.ones_like(lat) * C * np.cos(np.radians(lat))
    dy = np.diff(lat)[0] * C
    
    print("innsize = ", inn3Dsize)
    print("x.shape = ", x.shape)
    
    def fun(x):
        return J_alpha(
            x,            # size = 2*(m+1)*(n+1)
            y3D,          # size = 2*m*n
            dx,           # size = m
            dy,           # size = 1
            alpha,
            inn3D,
            inn3Dsize,
            m,
            n
        )
    
    print("fun(x0) = ",fun(x))
    
    if not run:
        raise ValueError()

    
    def jacobian_fun(x):
        return nabla_J_alpha(
            x,            # size = 2*(m+1)*(n+1)
            y3D,          # size = 2*m*n
            dx,           # size = m
            dy,           # size = 1
            alpha,
            inn3D,
            inn3Dsize,
            m,
            n
        )
    
    
    result = minimize(
        fun,
        x,
        method = 'L-BFGS-B', ## what minimization-method?
        jac = jacobian_fun,
        options = {
            'gtol':gtol, 
            'disp':disp
        }
    )
    
    return (
        result.x[:(m+1)*(n+1)].reshape(m+1,n+1)[:,:-2 if wraparound else -1],  # Optimized Psi
        result.x[(m+1)*(n+1):].reshape(m+1,n+1)[:,:-2 if wraparound else -1],  # Optimized Chi
        result # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
    )

@njit(**parallel(), locals=dict(innsize=uint32, m=uint32,n=uint32))
def J_alpha( x, y, dx, dy, alpha, inn, innsize, m, n ):
    """
    Equation 10: J_alpha = 1/2*(y-Ax)^T*(y-Ax) + 1/2*alpha*(x^T*x)
    """
    
    x_reshaped = x.reshape(2,m+1,n+1).copy()
    Ax = get_3DAx( x_reshaped, dx, dy, inn, m, n )
    
    eTe = 0.0
    for i in prange(m):
        for j in range(n):
            euij = y[0,i,j] - Ax[0,i,j]
            evij = y[1,i,j] - Ax[1,i,j]
            eTe = eTe + euij*euij + evij*evij
        
    xTx = 0.0
    for i in prange(m+1):
        for j in range(n+1):
            Psi_ij = x_reshaped[0,i,j]
            Chi_ij = x_reshaped[1,i,j]
            xTx = xTx + Psi_ij*Psi_ij + Chi_ij*Chi_ij
            
    return 0.5*(eTe + alpha*xTx)    
    
    
@njit( **parallel(), locals=dict(m=uint32,n=uint32) ) # opts(), since "if-condition" not parallelizable.
def get_3DAx( x, dx, dy, inn, m, n ):
    
    """
    Build 'Ax' [ 2 , m , n ] using 'x' [ 2 , (m+1) , (n+2) ]
    """
    
    Ax = np.empty((2,m,n))
    
    for i in prange(m):
        
        dxi = dx[i]
        
        for j in range(n):
            
            PsiUR = x[0, i+1, j+1]
            PsiUL = x[0, i+1, j  ]
            PsiLR = x[0, i,   j+1]
            PsiLL = x[0, i,   j  ]
            ChiUR = x[1, i+1, j+1]
            ChiUL = x[1, i+1, j  ]
            ChiLR = x[1, i,   j+1]
            ChiLL = x[1, i,   j  ]

            # u = -dPsi/dy + dChi/dx         
            Ax[0,i,j] = -0.5 * ( PsiUR + PsiUL - PsiLR - PsiLL ) / dy + 0.5 * ( ChiUR + ChiLR - ChiUL - ChiLL ) / dxi 

            # v =  dPsi/dx + dChi/dy
            Ax[1,i,j] =  0.5 * ( PsiUR + PsiLR - PsiUL - PsiLL ) / dxi + 0.5 * ( ChiUR + ChiUL - ChiLR - ChiLL ) / dy   

    return Ax
    

@njit( **parallel(), locals = dict(m=uint32, n=uint32, innsize=uint32) )
def nabla_J_alpha( x, y, dx, dy, alpha, inn, innsize, m, n ):
    """
    Equation 11: nabla J_alpha = - A^T(y-Ax) + alpha*x
    """
    x_reshaped = x.reshape(2,m+1,n+1).copy()
    Ax = get_3DAx( x_reshaped, dx, dy, inn, m, n )
    
    adj = adjoint_term_3D( Ax, y, dx, dy, inn, innsize, m, n )
    
    gradient_of_J_alpha = np.empty_like(adj)
    
    for i in prange(m+1):
        for j in range(n+1):
            gradient_of_J_alpha[0,i,j] = - adj[0,i,j] + alpha * x_reshaped[0,i,j]
            gradient_of_J_alpha[1,i,j] = - adj[0,i,j] + alpha * x_reshaped[0,i,j]
            
    return gradient_of_J_alpha.reshape(2*(m+1)*(n+1))



@njit( **parallel(), locals = dict(m=uint32, n=uint32, innsize=uint32) )
def adjoint_term_3D( Ax, y, dx, dy, inn, innsize, m, n ):
    """
    adjoint_term = A^T * (y-Ax)
    """
    
    error = np.empty_like(y)
    for i in prange(m):
        for j in range(n):
            error[0,i,j] = y[0,i,j] - Ax[0,i,j]
        
    adjoint_term = np.empty((2,m+1,n+1))
    
    # Fill North/South-edge with zeros
    for j in prange(n+1):
        adjoint_term[0,0,j] = 0.0
        adjoint_term[1,0,j] = 0.0
        adjoint_term[0,m,j] = 0.0
        adjoint_term[1,m,j] = 0.0

    
    # Do East/West wraparound
    for i in prange(1, m):
        dxi = 0.5 * (dx[i] + dx[i-1])
        euUR = error[0,i,0]
        euUL = error[0,i,n-1]
        euLR = error[0,i-1,0]
        euLL = error[0,i-1,n-1]
        evUR = error[1,i,0]
        evUL = error[1,i,n-1]
        evLR = error[1,i-1,0]
        evLL = error[1,i-1,n-1]
        adjoint_term[0,i,0] = - 0.5 * ( euUR + euUL - euLR - euLL ) / dy + 0.5 * ( evUR + evLR - evUL - evLL ) / dxi
        adjoint_term[1,i,0] =   0.5 * ( euUR + euLR - euUL - euLL ) / dxi + 0.5 * ( evUR + evUL - evLR - evLL ) / dy
        adjoint_term[0,i,n] = - 0.5 * ( euUR + euUL - euLR - euLL ) / dy + 0.5 * ( evUR + evLR - evUL - evLL ) / dxi
        adjoint_term[1,i,n] =   0.5 * ( euUR + euLR - euUL - euLL ) / dxi + 0.5 * ( evUR + evUL - evLR - evLL ) / dy
        
    

    # Calculate the rest of the adjoint-term-array
    for i in prange(1, m):
        dxi = 0.5 * (dx[i] + dx[i-1])
        for j in range(1, n):
            euUR = error[0,i,j]
            euUL = error[0,i,j-1]
            euLR = error[0,i-1,j]
            euLL = error[0,i-1,j-1]
            evUR = error[1,i,j]
            evUL = error[1,i,j-1]
            evLR = error[1,i-1,j]
            evLL = error[1,i-1,j-1]
            adjoint_term[0,i,j] = - 0.5 * ( euUR + euUL - euLR - euLL ) / dy + 0.5 * ( evUR + evLR - evUL - evLL ) / dxi
            adjoint_term[1,i,j] =   0.5 * ( euUR + euLR - euUL - euLL ) / dxi + 0.5 * ( evUR + evUL - evLR - evLL ) / dy
            
    
    return adjoint_term
