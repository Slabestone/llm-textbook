#!/usr/bin/env python3
"""EP29 four independent mathematical teaching toys. No neural model trained."""
import json, math

def ddpm():
    x0=2.0; alphas=[.9,.8]; noises=[.4,-.6]; states=[x0]
    for a,e in zip(alphas,noises): states.append(math.sqrt(a)*states[-1]+math.sqrt(1-a)*e)
    abar=alphas[0]*alphas[1]
    eps_bar=(states[2]-math.sqrt(abar)*x0)/math.sqrt(1-abar)
    recovered=(states[2]-math.sqrt(1-abar)*eps_bar)/math.sqrt(abar)
    assert abs(recovered-x0)<1e-12 and all(abs(eps_bar-e)>1e-3 for e in noises)
    return {"x0":x0,"alpha":alphas,"independent_step_eps":noises,"states":states,"alpha_bar":abar,"derived_marginal_eps":eps_bar,"recovered_if_exact_eps":recovered,"note":"marginal eps is a derived combination of step noises, not a new independent Markov noise"}

def flow():
    z0,z1=-1.,3.; v=z1-z0
    trajectory={str(t):(1-t)*z0+t*z1 for t in (0,.25,.5,.75,1)}
    assert z0+v==z1
    pairs=[(-1.,1.),(1.,-1.)]; t=.5
    states=[(1-t)*a+t*b for a,b in pairs]
    vel=[b-a for a,b in pairs]
    assert states==[0.,0.] and vel==[2.,-2.]
    prediction=sum(vel)/len(vel)
    mse=sum((v-prediction)**2 for v in vel)/len(vel)
    assert prediction==0 and mse==4
    return {"single_pair":{"z0":z0,"z1":z1,"velocity":v,"trajectory":trajectory,"euler_one_step_exact":z0+v},
            "crossing_pairs":{"pairs":pairs,"time":t,"same_state":states,"opposite_velocities":vel,"conditional_mean_velocity":prediction,"irreducible_mse":mse},
            "note":"straight conditional target is not proof of one-step exactness for the learned marginal v(x,t)"}

def ode():
    exact=math.e
    out={}
    for n in (1,2,4,8,16,32):
        x=1.
        for _ in range(n): x=x+x/n
        out[str(n)]={"Euler_x1":x,"abs_error":abs(x-exact)}
    assert all(out[str(b)]["abs_error"]<out[str(a)]["abs_error"] for a,b in zip((1,2,4,8,16),(2,4,8,16,32)))
    return {"ode":"dx/dt=x, x(0)=1","exact_x1":exact,"Euler":out,"note":"nonconstant-velocity ODE Euler discretization, not generative-model quality"}

if __name__=="__main__":
    print(json.dumps({"schema":"EP29_mathematical_toys_v01","DDPM":ddpm(),"flow":flow(),"solver":ode(),"scope":"exact arithmetic only; no trained denoiser/flow, image or video generation"},ensure_ascii=False,indent=2))
