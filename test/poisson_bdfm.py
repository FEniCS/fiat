"""
FEniCS tutorial demo program:
Poisson equation with Dirichlet conditions.
Simplest example of computation and visualization.

-Laplace(u) = f on the unit square.
u = u0 on the boundary.
u0 = u = 1 + x^2 + 2y^2, f = -6.
"""

from dolfin import *

# Define boundary conditions
p0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]',degree=3)

# Exact solution for u.
u0 = Expression(('-2*x[0]','-4*x[1]'),degree=3)

def p0_boundary(x, on_boundary):
    return on_boundary

def solve_poisson(size):
    # Create mesh and define function space
    mesh = UnitSquare(size, size)
    P = FunctionSpace(mesh, 'CG', 2)
    V = VectorFunctionSpace(mesh, 'DG', 1)
    W=P*V

    bc = DirichletBC(P, p0, p0_boundary)

    # Define variational problem
    q,v = TestFunctions(W)
    p,u = TrialFunctions(W)
    f = Constant(-6.0)
    #a = inner(grad(u), grad(v))*dx
    a=-inner(grad(q),u)*dx+inner(v,u-grad(p))*dx
    L = f*q*dx

    # Compute solution
    u=Function(W)
    solve(a==L, u, bc)
    
    return u, mesh

def solve_poisson_bdfm(size):
    # Create mesh and define function space
    mesh = UnitSquare(size, size)
    P = FunctionSpace(mesh, 'DG', 1)
    V = FunctionSpace(mesh, 'BDFM', 1)
    W=P*V

# bc = DirichletBC(P, p0, p0_boundary)

    n=FacetNormal(mesh)

    # Define variational problem
    q,v = TestFunctions(W)
    p,u = TrialFunctions(W)
    f = Constant(-6.0)
    #a = inner(grad(u), grad(v))*dx
    a=q*div(u)*dx-div(v)*p*dx+inner(v,u)*dx
    L = f*q*dx-p0*dot(v,n)*ds

    ffc_opt = {"quadrature_degree": 4, "representation": "quadrature"}

    # Compute solution
    u=Function(W)
    solve(a==L, u, form_compiler_parameters=ffc_opt)
    
    return u

def convergence_bdfm():

    e_p=[]
    e_u=[]
    e_div=[]

    for s in [4,8,16,32,64]:
        print s
        w=solve_poisson_bdfm(s)
        p,u=split(w)

        mesh=w.function_space().mesh()
        cg=VectorFunctionSpace(mesh,'CG',3)
        u_0=interpolate(u0,cg)

        #error = errornorm(p, p0, "L2")
        #e_p.append(error)
        e_p.append((assemble((p-p0)**2*dx))**0.5)
        e_u.append((assemble(inner(u-u_0,u-u_0)*dx))**0.5)
        e_div.append((assemble(div(u-u_0)**2*dx))**0.5)
        #error = errornorm(u, u0, "L2")
        #e_u.append(error)


    return e_p,e_u,e_div
        
        
        
def run_poisson(size):
    
    u,mesh=solve_poisson(size)

    # Plot solution and mesh
    plot(u)
    plot(mesh)
    
    # Dump solution to file in VTK format
    file = File('poisson.pvd')
    file << u
    
    # Hold plot
    interactive()
