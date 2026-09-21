#!/usr/bin/env python3
"""Computational Physics Homework 1

Reproduces all numerical calculations and figures for Problems 1--3.
All arithmetic requested as single precision is explicitly performed with np.float32.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"; RES = ROOT / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
# Locate the supplied matter-power-spectrum file in the same folder as this script.
# This supports the original download name (lcdm_z0(1).matter_pk) as well as
# cleaner GitHub names such as lcdm_z0.matter_pk.
def find_data_file():
    preferred = [ROOT / "lcdm_z0.matter_pk", ROOT / "lcdm_z0(1).matter_pk"]
    for path in preferred:
        if path.exists():
            return path
    matches = sorted(ROOT.glob("lcdm_z0*.matter_pk"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Could not find the matter power spectrum. Put lcdm_z0(1).matter_pk "
        "or lcdm_z0.matter_pk in the same folder as this Python script."
    )

DATA = find_data_file()

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
    # Richardson extrapolation of the O(h^2) central difference.
    h = np.float32(h)
    d_h = central_difference(f, x, h)
    d_h2 = central_difference(f, x, h / np.float32(2.0))
    return np.float32((np.float32(4.0) * d_h2 - d_h) / np.float32(3.0))

def relerr(approx, exact): return abs(float(approx) - float(exact)) / abs(float(exact))

def problem1():
    hs = np.logspace(-7, 0, 141).astype(np.float32)
    cases = [("cos", f32_cos, lambda x: -np.sin(x), 0.1),
             ("cos", f32_cos, lambda x: -np.sin(x), 10.0),
             ("exp", f32_exp, np.exp, 0.1),
             ("exp", f32_exp, np.exp, 10.0)]
    methods = [("Forward", forward_difference), ("Central", central_difference),
               ("Extrapolated", extrapolated_difference)]
    rows=[]
    fig, axes = plt.subplots(2,2, figsize=(10,8), constrained_layout=True)
    for ax,(name,f,df,x) in zip(axes.flat,cases):
        exact=float(df(x))
        for label,method in methods:
            errs=np.array([relerr(method(f,x,h),exact) for h in hs])
            positive=np.where(np.isfinite(errs) & (errs>0))[0]
            i=positive[np.argmin(errs[positive])]
            rows.append((name,x,label,float(hs[i]),float(errs[i]),float(method(f,x,hs[i])),exact))
            ax.loglog(hs,errs,label=label)
            ax.scatter([hs[i]],[errs[i]],s=20)
        ax.set_title(f"{name}(x) at x = {x:g}")
        ax.set_xlabel("step size h")
        ax.set_ylabel("relative error")
        ax.grid(True,which="both",alpha=.25); ax.legend(fontsize=8)
    fig.suptitle("Single-precision numerical differentiation")
    fig.savefig(FIG/"problem1_differentiation_errors.png",dpi=220); plt.close(fig)
    with open(RES/"problem1_minimum_errors.csv","w") as out:
        out.write("function,x,method,h_at_min,minimum_relative_error,numerical_derivative,exact_derivative\n")
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

def build_pk_spline(k,P):
    # log-log cubic spline respects the smooth power-law behavior and positivity.
    spl=CubicSpline(np.log(k),np.log(P),extrapolate=True)
    return lambda x: np.exp(spl(np.log(x)))

def xi_of_r(r, pk, kmin, kmax):
    # xi = [1/(2 pi^2 r)] integral k P(k) sin(kr) dk.
    val,_=quad(lambda kk: kk*pk(kk), kmin, kmax, weight="sin", wvar=float(r),
               epsabs=1e-8,epsrel=1e-7,limit=500,limlst=500)
    return val/(2.0*np.pi**2*r)

def problem3():
    k,P=load_power_spectrum(); pk=build_pk_spline(k,P)
    rs=np.linspace(50.0,120.0,281)
    cutoffs=[10.0,30.0,100.0,300.0]
    curves={}
    for km in cutoffs:
        xis=np.array([xi_of_r(r,pk,k[0],km) for r in rs])
        curves[km]=rs**2*xis
    y=curves[300.0]
    # The BAO feature is the broad local bump near ~100 Mpc/h, not the
    # larger broadband value at the left edge of the requested interval.
    peak_mask=(rs>=80.0) & (rs<=120.0)
    peak_idx=np.where(peak_mask)[0]
    i=int(peak_idx[np.argmax(y[peak_mask])]); rpeak=float(rs[i]); ypeak=float(y[i])
    # Parabolic three-point refinement of peak location.
    if 0<i<len(rs)-1:
        coeff=np.polyfit(rs[i-1:i+2],y[i-1:i+2],2)
        rpeak=float(-coeff[1]/(2*coeff[0])); ypeak=float(np.polyval(coeff,rpeak))
    fig,ax=plt.subplots(figsize=(7.4,5.2),constrained_layout=True)
    ax.plot(rs,y,label=r"$r^2\xi(r)$, $k_{\max}=300\,h\,\mathrm{Mpc}^{-1}$")
    ax.axvline(rpeak,ls="--",label=f"BAO peak = {rpeak:.2f} Mpc/h")
    ax.scatter([rpeak],[ypeak],zorder=3)
    ax.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$"); ax.set_ylabel(r"$r^2\xi(r)$")
    ax.set_title("Matter correlation function and BAO peak")
    ax.grid(True,alpha=.25); ax.legend()
    fig.savefig(FIG/"problem3_bao_peak.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.4,5.2),constrained_layout=True)
    for km in cutoffs: ax.plot(rs,curves[km],label=fr"$k_{{\max}}={km:g}$")
    ax.set_xlabel(r"$r\ [\mathrm{Mpc}/h]$"); ax.set_ylabel(r"$r^2\xi(r)$")
    ax.set_title("Upper-limit convergence of the correlation function")
    ax.grid(True,alpha=.25); ax.legend()
    fig.savefig(FIG/"problem3_cutoff_convergence.png",dpi=220); plt.close(fig)
    # P(k) figure showing baryon-wiggle region.
    fig,ax=plt.subplots(figsize=(7.4,5.2),constrained_layout=True)
    ax.loglog(k,P); ax.axvspan(0.03,0.3,alpha=.12,label="BAO wiggle region")
    ax.set_xlabel(r"$k\ [h/\mathrm{Mpc}]$"); ax.set_ylabel(r"$P(k)$")
    ax.set_title("Input matter power spectrum"); ax.grid(True,which="both",alpha=.25); ax.legend()
    fig.savefig(FIG/"problem3_power_spectrum.png",dpi=220); plt.close(fig)
    # cutoff peak table
    with open(RES/"problem3_cutoff_convergence.csv","w") as out:
        out.write("kmax,grid_peak_r,peak_r2xi\n")
        for km in cutoffs:
            j=int(np.argmax(curves[km])); out.write(f"{km},{rs[j]},{curves[km][j]}\n")
    return rpeak,ypeak

def main():
    p1=problem1(); p2=problem2(); p3=problem3()
    print("float32 epsilon =",EPS32)
    print("Problem 1 minima:")
    for r in p1: print(r)
    print("Problem 2 minima:")
    for r in p2[0]: print(r)
    print("Problem 3 BAO peak: r = %.4f Mpc/h, r^2 xi = %.6g"%p3)

if __name__ == "__main__": main()
