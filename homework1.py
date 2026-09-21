#!/usr/bin/env python3
"""Computational Physics Homework 1

Reproduces all numerical calculations and figures for Problems 1--3.
All arithmetic requested as single precision is explicitly performed with np.float32.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"; RES = ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
DATA = ROOT / "lcdm_z0.matter_pk"

EPS32 = np.finfo(np.float32).eps

# ---------------- Problem 1: numerical differentiation ----------------
def f32_cos(x): return np.cos(np.float32(x), dtype=np.float32)
def f32_exp(x): return np.exp(np.float32(x), dtype=np.float32)

def forward_difference(f, x, h):
    x, h = np.float32(x), np.float32(h)
    return np.float32((f(x + h) - f(x)) / h)

def central_difference(f, x, h):
    x, h = np.float32(x), np.float32(h)
    return np.float32((f(x + h) - f(x - h)) / (np.float32(2.0) * h))

def extrapolated_difference(f, x, h):
    # Same fourth-order Richardson formula, parameterized with h and 2h.
    h = np.float32(h)
    d_h = central_difference(f, x, h)
    d_2h = central_difference(f, x, np.float32(2.0) * h)
    return np.float32((np.float32(4.0) * d_h - d_2h) / np.float32(3.0))

def relerr(approx, exact): return abs(float(approx) - float(exact)) / abs(float(exact))

def fit_truncation_slope(hs, errs, order, optimum_index=None):
    # Fit the truncation-dominated points to the right of the representative
    # minimum.  Restricting to errors below 0.2 avoids the far-right region
    # where higher Taylor terms can spoil the leading-order power law.
    mask=np.isfinite(errs) & (errs>0) & (errs<0.2)
    if optimum_index is not None:
        mask &= np.arange(len(hs)) > optimum_index
    if np.count_nonzero(mask) < 4:
        return np.nan
    return float(np.polyfit(np.log10(hs[mask].astype(float)),np.log10(errs[mask]),1)[0])

def problem1():
    hs = np.logspace(-8, 0, 500).astype(np.float32)
    cases = [("cos", f32_cos, lambda x: -np.sin(x), 0.1),
             ("cos", f32_cos, lambda x: -np.sin(x), 10.0),
             ("exp", f32_exp, np.exp, 0.1),
             ("exp", f32_exp, np.exp, 10.0)]
    methods = [("Forward", forward_difference,1),("Central", central_difference,2),
               ("Extrapolated", extrapolated_difference,4)]
    rows=[]; all_errors={}
    for name,f,df,x in cases:
        exact=float(df(x)); all_errors[(name,x)]={}
        for label,method,order in methods:
            errs=np.array([relerr(method(f,x,h),exact) for h in hs],dtype=float)
            safe=np.where(np.isfinite(errs)&(errs>0),errs,np.nan)
            filled=np.where(np.isfinite(np.log10(safe)),np.log10(safe),10.0)
            med=median_filter(filled,size=21,mode='nearest')
            i=int(np.nanargmin(med)); representative=float(10**med[i])
            digits=float(-np.log10(representative))
            slope=fit_truncation_slope(hs,errs,order,i)
            rows.append((name,x,label,float(hs[i]),representative,digits,slope))
            all_errors[(name,x)][label]=(errs,i)

    fig, axes = plt.subplots(2,2, figsize=(10,8), constrained_layout=True)
    for ax,(name,f,df,x) in zip(axes.flat,cases):
        for label,method,order in methods:
            errs,_=all_errors[(name,x)][label]; ax.loglog(hs,errs,label=label,lw=1.0)
        fx = r"\cos x" if name == "cos" else r"e^x"
        ax.set_title(rf"$f(x)={fx},\quad x={x:g}$"+"\nConnected error curves")
        ax.set_xlabel(r"Step size $h$"); ax.set_ylabel("Relative error")
        ax.grid(True,which="both",alpha=.25); ax.legend(fontsize=8)
    fig.savefig(FIG/"problem1_connected_errors.png",dpi=220); plt.close(fig)

    fig, axes = plt.subplots(2,2, figsize=(10,8), constrained_layout=True)
    for ax,(name,f,df,x) in zip(axes.flat,cases):
        for label,method,order in methods:
            errs,_=all_errors[(name,x)][label]; ax.loglog(hs,errs,'.',ms=1.8,label=label)
        if name=='exp' and abs(x-0.1)<1e-12:
            errs,_=all_errors[(name,x)]['Extrapolated']; valid=np.isfinite(errs)&(errs>0)
            ids=np.where(valid)[0]; j=ids[np.argmin(errs[valid])]
            ax.scatter([hs[j]],[errs[j]],marker='*',s=90,edgecolors='k',zorder=5,label='Raw minimum')
        fx = r"\cos x" if name == "cos" else r"e^x"
        ax.set_title(rf"$f(x)={fx},\quad x={x:g}$"+"\nIndividual error values")
        ax.set_xlabel(r"Step size $h$"); ax.set_ylabel("Relative error")
        ax.grid(True,which="both",alpha=.25); ax.legend(fontsize=8)
    fig.savefig(FIG/"problem1_discrete_errors.png",dpi=220); plt.close(fig)

    with open(RES/"problem1_minimum_errors.csv","w") as out:
        out.write("function,x,method,h_representative,representative_relative_error,reliable_digits,fitted_truncation_slope\n")
        for r in rows: out.write(",".join(map(str,r))+"\n")
    return rows

# ---------------- Problem 2: quadrature ----------------
def sequential_sum_float32(a):
    # cumsum deliberately exposes ordinary sequential single-precision accumulation.
    return np.cumsum(np.asarray(a,dtype=np.float32),dtype=np.float32)[-1]

def midpoint_rule(N):
    N=int(N); h=np.float32(1.0)/np.float32(N)
    x=(np.arange(N,dtype=np.float32)+np.float32(0.5))*h
    return np.float32(h*sequential_sum_float32(np.exp(-x,dtype=np.float32)))

def trapezoid_rule(N):
    N=int(N); h=np.float32(1.0)/np.float32(N)
    if N==1: interior=np.float32(0)
    else:
        x=np.arange(1,N,dtype=np.float32)*h
        interior=sequential_sum_float32(np.exp(-x,dtype=np.float32))
    endpoints=np.float32(0.5)*(np.float32(1.0)+np.exp(np.float32(-1.0),dtype=np.float32))
    return np.float32(h*(endpoints+interior))

def simpson_rule(N):
    N=int(N)
    if N%2: N+=1
    h=np.float32(1.0)/np.float32(N)
    odd=np.arange(1,N,2,dtype=np.float32)*h
    even=np.arange(2,N,2,dtype=np.float32)*h
    so=sequential_sum_float32(np.exp(-odd,dtype=np.float32))
    se=np.float32(0.0) if len(even)==0 else sequential_sum_float32(np.exp(-even,dtype=np.float32))
    total=(np.float32(1.0)+np.exp(np.float32(-1.0),dtype=np.float32)
           +np.float32(4.0)*so+np.float32(2.0)*se)
    return np.float32(h*total/np.float32(3.0))

def problem2():
    exact=1.0-np.exp(-1.0)
    Ns=np.unique(np.logspace(0,np.log10(3_000_000),45).astype(int))
    Ns=np.array([n if n%2==0 else n+1 for n in Ns],dtype=int); Ns=np.unique(Ns)
    methods=[("Midpoint",midpoint_rule),("Trapezoid",trapezoid_rule),("Simpson",simpson_rule)]
    curves={}; rows=[]
    for label,method in methods:
        e=[]
        for N in Ns:
            q=method(N); er=relerr(q,exact); e.append(er)
        e=np.array(e); curves[label]=e
        i=np.argmin(e)
        rows.append((label,int(Ns[i]),float(e[i]),float(method(Ns[i]))))
    fig,ax=plt.subplots(figsize=(7.4,5.2),constrained_layout=True)
    for label,_ in methods: ax.loglog(Ns,curves[label],marker="o",ms=3,label=label)
    ax.set_xlabel("number of bins N"); ax.set_ylabel("relative error")
    ax.set_title(r"Single-precision quadrature for $\int_0^1 e^{-t}\,dt$")
    ax.grid(True,which="both",alpha=.25); ax.legend()
    fig.savefig(FIG/"problem2_quadrature_errors.png",dpi=220); plt.close(fig)
    with open(RES/"problem2_minimum_errors.csv","w") as out:
        out.write("method,N_at_min,minimum_relative_error,integral_estimate\n")
        for r in rows: out.write(",".join(map(str,r))+"\n")
    return rows, Ns, curves

# ---------------- Problem 3: correlation function / BAO ----------------
def load_power_spectrum():
    d=np.loadtxt(DATA); return d[:,0],d[:,1]

def build_logpk_spline(k,P):
    return CubicSpline(np.log(k),np.log(P),bc_type="natural",extrapolate=False)

def simpson_uniform(y, dx):
    # Composite Simpson rule for an even number of bins.
    n=len(y)-1
    if n%2: raise ValueError("Simpson rule requires an even number of bins")
    return dx/3.0*(y[0]+y[-1]+4.0*np.sum(y[1:-1:2])+2.0*np.sum(y[2:-1:2]))

def xi_curve_simpson(rs,k,P,kmax,N):
    if N%2: raise ValueError("N must be even")
    kmin=float(k[0]); kmax=min(float(kmax),float(k[-1]))
    spline=build_logpk_spline(k,P)
    u=np.linspace(np.log(kmin),np.log(kmax),N+1)
    kk=np.exp(u); pp=np.exp(spline(u)); du=u[1]-u[0]
    base=kk**3*pp
    xis=[]
    for r in rs:
        z=kk*r
        kernel=np.sinc(z/np.pi) # sin(z)/z
        xis.append(simpson_uniform(base*kernel,du)/(2*np.pi**2))
    return np.asarray(xis)

def local_peak(rs,y,rlo=90.0,rhi=120.0):
    mask=(rs>=rlo)&(rs<=rhi); ids=np.where(mask)[0]
    i=int(ids[np.argmax(y[mask])]); rg=float(rs[i]); yg=float(y[i])
    if 0<i<len(rs)-1:
        c=np.polyfit(rs[i-1:i+2],y[i-1:i+2],2)
        rp=float(-c[1]/(2*c[0])); yp=float(np.polyval(c,rp))
    else: rp,yp=rg,yg
    return i,rg,yg,rp,yp

def problem3():
    k,P=load_power_spectrum(); rs=np.arange(50.0,120.0+0.05,0.1)
    final_kmax=100.0; final_N=2**18
    cutoffs=[3.0,10.0,30.0,100.0]
    curves={km: rs**2*xi_curve_simpson(rs,k,P,km,final_N) for km in cutoffs}
    y=curves[final_kmax]
    _,grid_r,grid_y,rpeak,ypeak=local_peak(rs,y)

    fig,ax=plt.subplots(figsize=(7.4,5.2),constrained_layout=True)
    ax.plot(rs,y,label=fr"$r^2\xi(r)$, $k_{{\max}}={final_kmax:g}\,h/\mathrm{{Mpc}}$")
    ax.axvline(rpeak,ls="--",label=f"BAO peak = {rpeak:.2f} Mpc/h")
    ax.scatter([rpeak],[ypeak],zorder=3)
    ax.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$"); ax.set_ylabel(r"$r^2\xi(r)$")
    ax.set_title("Matter correlation function and BAO peak")
    ax.grid(True,alpha=.25); ax.legend()
    fig.savefig(FIG/"problem3_bao_peak.png",dpi=220); plt.close(fig)

    conv=[]
    for km in cutoffs:
        _,gr,gy,rp,yp=local_peak(rs,curves[km]); conv.append((km,final_N,gr,gy,rp,yp))
    for N in [2**14,2**16,2**18,2**19]:
        yy=rs**2*xi_curve_simpson(rs,k,P,final_kmax,N)
        _,gr,gy,rp,yp=local_peak(rs,yy); conv.append((final_kmax,N,gr,gy,rp,yp))
    with open(RES/"problem3_convergence.csv","w") as out:
        out.write("kmax,N,grid_peak_r,grid_peak_r2xi,parabolic_peak_r,parabolic_peak_r2xi\n")
        for row in conv: out.write(",".join(map(str,row))+"\n")
    return rpeak,ypeak,conv

def main():
    p1=problem1(); p2=problem2(); p3=problem3()
    print("float32 epsilon =",EPS32)
    print("Problem 1 minima:")
    for r in p1: print(r)
    print("Problem 2 minima:")
    for r in p2[0]: print(r)
    print("Problem 3 BAO peak: r = %.4f Mpc/h, r^2 xi = %.6g"%(p3[0],p3[1]))

if __name__ == "__main__": main()
