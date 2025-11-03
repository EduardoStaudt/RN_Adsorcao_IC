import numpy as np, math
import scipy.sparse as sp, scipy.sparse.linalg as spla
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Grid and params
nx, ny = 10000, 10000
Lx = Ly = 1.0
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
dx = x[1]-x[0]
dy = y[1]-y[0]
X, Y = np.meshgrid(x, y, indexing='ij')
N = nx*ny
nu = 0.1

def T_exact(x, y, t):
    return math.tanh(t) * (math.sin(math.pi * x) + math.cos(math.pi * y))
T_exact_vec = np.vectorize(T_exact)

def idx(i,j): return i*ny+j

# Upwind+diffusion operator as before
lx = 1.0/(dx*dx)
ly = 1.0/(dy*dy)
data=[]
rows=[]
cols=[]
for i in range(nx):
    for j in range(ny):
        k = idx(i,j)
        if i==0 or i==nx-1 or j==0 or j==ny-1:
            rows.append(k)
            cols.append(k)
            data.append(0.0)
        else:
            a_center = nu*(-2*lx - 2*ly) - (1.0/dx) - (1.0/dy)
            a_ip1 = nu*lx
            a_im1 = nu*lx + (1.0/dx)   # upwind x (flow to +x)
            a_jp1 = nu*ly
            a_jm1 = nu*ly + (1.0/dy)   # upwind y (flow to +y)
            rows += [k,k,k,k,k]
            cols += [k, idx(i+1,j), idx(i-1,j), idx(i,j+1), idx(i,j-1)]
            data += [a_center, a_ip1, a_im1, a_jp1, a_jm1]
A = sp.csr_matrix((data,(rows,cols)), shape=(N,N))
I = sp.eye(N, format='csr')

# Time parameters (refined dt)
t0, t1 = 0.0, 1.0
nt = 4000
dt = (t1-t0)/nt
M_L = (I - 0.5*dt*A).tocsr()
M_R = (I + 0.5*dt*A).tocsr()

# Masks/indices
boundary_mask = np.zeros((nx, ny), dtype=bool)
boundary_mask[0,:]=True
boundary_mask[-1,:]=True
boundary_mask[:,0]=True
boundary_mask[:,-1]=True
boundary_ids = np.where(boundary_mask.flatten())[0]
interior_ids = np.where(~boundary_mask.flatten())[0]
M_L_int = M_L[interior_ids[:,None], interior_ids]
M_R_int = M_R[interior_ids[:,None], interior_ids]
M_L_int_bdry = M_L[interior_ids[:,None], boundary_ids]
M_R_int_bdry = M_R[interior_ids[:,None], boundary_ids]

# Source from the report (consistent with manufactured solution derivation)
def S_report(x,y,t,nu):
    return (1.0/(math.cosh(t)**2))*(math.sin(math.pi*x)+math.cos(math.pi*y)) + \
        math.tanh(t)*(math.pi*math.cos(math.pi*x) - math.pi*math.sin(math.pi*y) + nu*(math.pi**2)*(math.sin(math.pi*x)+math.cos(math.pi*y)))
S_vec = np.vectorize(lambda xx,yy,tt: S_report(xx,yy,tt,nu))

# Factorize
solve_int = spla.factorized(M_L_int.tocsc())

# Integrate
U = np.zeros((nx,ny))
t=t0
for n in range(nt):
    t_n = t
    t_np1 = t+dt
    # Exact Dirichlet on boundary
    U_bc_n = np.zeros((nx,ny))
    U_bc_np1 = np.zeros((nx,ny))
    U_bc_n[0,:] = T_exact_vec(x[0],y,t_n)
    U_bc_np1[0,:] = T_exact_vec(x[0],y,t_np1)
    U_bc_n[-1,:]= T_exact_vec(x[-1],y,t_n)
    U_bc_np1[-1,:]= T_exact_vec(x[-1],y,t_np1)
    U_bc_n[:,0] = T_exact_vec(x,y[0],t_n)
    U_bc_np1[:,0] = T_exact_vec(x,y[0],t_np1)
    U_bc_n[:,-1]= T_exact_vec(x,y[-1],t_n)
    U_bc_np1[:,-1]= T_exact_vec(x,y[-1],t_np1)

    S_n=S_vec(X,Y,t_n)
    S_np1=S_vec(X,Y,t_np1)

    U_flat=U.flatten()
    b = M_R @ U_flat + 0.5*dt*(S_n.flatten()+S_np1.flatten())
    b_int = b[interior_ids] - M_L_int_bdry @ U_bc_np1.flatten()[boundary_ids] + M_R_int_bdry @ U_bc_n.flatten()[boundary_ids]
    U_int_next = solve_int(b_int)

    U_next = U_flat.copy()
    U_next[interior_ids] = U_int_next
    U_next[boundary_ids] = U_bc_np1.flatten()[boundary_ids]
    U = U_next.reshape((nx,ny))
    t = t_np1

# Metrics
T_ref = T_exact_vec(X,Y,t1)
abs_err = np.abs(U - T_ref)
with np.errstate(divide='ignore', invalid='ignore'):
    rel_err = np.where(np.abs(T_ref) > 1e-12, abs_err/np.abs(T_ref), 0.0)

mae = abs_err.mean()
mre = 100.0*rel_err.mean()
maxe = abs_err.max()
U_vec=U.flatten()
T_vec=T_ref.flatten()
r2 = 1.0 - np.sum((T_vec-U_vec)**2)/np.sum((T_vec - T_vec.mean())**2)

cmp = pd.DataFrame([
    {"Métrica":"MAE","CN (dt refinado)":mae,"RKF45 (relatório)":0.148},
    {"Métrica":"Erro relativo (%)","CN (dt refinado)":mre,"RKF45 (relatório)":4.428},
    {"Métrica":"Maior erro","CN (dt refinado)":maxe,"RKF45 (relatório)":0.783},
    {"Métrica":"R²","CN (dt refinado)":r2,"RKF45 (relatório)":0.897},
])
print(cmp)
plt.figure()
plt.imshow((U-T_ref).T, origin='lower', extent=(0,1,0,1), cmap='seismic')
plt.title("Erro CN (dt=1/4000)")
plt.colorbar()
plt.tight_layout()
plt.show()
plt.close('all')
