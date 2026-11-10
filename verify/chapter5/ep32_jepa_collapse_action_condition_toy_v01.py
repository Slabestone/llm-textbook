#!/usr/bin/env python3
"""EP32 JEPA learning-target failure modes: four deterministic scalar cases.

Educational 2-component observations (task state + nuisance).
No trained JEPA, learned encoder, robot, loss landscape or video model.
"""
import json
import statistics

ACTIONS=[-1.,1.,-1.,1.]
NUISANCES=[-10.,-10.,10.,10.]
FUTURE_STATE=ACTIONS[:]

def mse(pairs):
    pairs=list(pairs)
    return sum((pred-target)**2 for pred,target in pairs)/len(pairs)

def action_conditional():
    predicted_blind=[0.]*len(ACTIONS)
    predicted_aware=ACTIONS[:]
    blind_mse=mse(zip(predicted_blind,FUTURE_STATE))
    aware_mse=mse(zip(predicted_aware,FUTURE_STATE))
    assert blind_mse==1. and aware_mse==0.
    return {"observed_current_state":[0.]*len(ACTIONS),
            "actions":ACTIONS,"target_future_states":FUTURE_STATE,
            "action_blind_mean_prediction":predicted_blind,
            "action_blind_future_state_mse":blind_mse,
            "action_aware_prediction":predicted_aware,
            "action_aware_future_state_mse":aware_mse,
            "note":"actions disambiguate this contrived conditional future; actual controlled dynamics may be stochastic"}

def collapse():
    target_state=FUTURE_STATE
    # A learned encoder e(s)=0 everywhere and predictor p=0 produce zero
    # squared JEPA feature matching loss while preserving zero information.
    encoded_collapsed=[0.]*len(target_state)
    predicted_collapsed=[0.]*len(target_state)
    collapsed_loss=mse(zip(predicted_collapsed,encoded_collapsed))
    # Noncollapsed e(s)=s plus a-conditioned predictor p(s,a)=a also yields 0.
    encoded_noncollapsed=target_state[:]
    predicted_noncollapsed=ACTIONS[:]
    noncollapsed_loss=mse(zip(predicted_noncollapsed,encoded_noncollapsed))
    noncollapsed_var=statistics.pvariance(encoded_noncollapsed)
    collapsed_var=statistics.pvariance(encoded_collapsed)
    assert collapsed_loss==noncollapsed_loss==0
    assert collapsed_var==0 and noncollapsed_var==1
    return {"trivial_constant_encoder_latent_loss":collapsed_loss,
            "constant_encoder_population_variance":collapsed_var,
            "meaningful_action_conditional_latent_loss":noncollapsed_loss,
            "meaningful_embedding_population_variance":noncollapsed_var,
            "warning":"A simple jointly learned MSE objective permits constant collapse; actual JEPA systems use architectural/training choices that must be specified, not assumed."}

def nuisance():
    # Targets are (physical action-conditioned state, unrelated appearance)
    target=list(zip(FUTURE_STATE,NUISANCES))
    pred=[(a,0.) for a in ACTIONS]
    elementwise_mse=sum((ps-s)**2+(pb-b)**2 for (ps,pb),(s,b) in zip(pred,target))/(len(target)*2)
    task_state_mse=mse(zip([p[0] for p in pred],FUTURE_STATE))
    assert elementwise_mse==50. and task_state_mse==0.
    return {"observations":target,"predictions":pred,"appearance_values":NUISANCES,
            "two_component_pixel_like_MSE":elementwise_mse,
            "latent_task_state_MSE":task_state_mse,
            "warning":"Nuisance is artificial and equally weighted. A robust representation may discard it if task-irrelevant; no empirical JEPA performance established."}

if __name__=="__main__":
    result={"schema":"EP32_JEPA_toy_v01",
            "action_condition":action_conditional(),
            "latent_collapse":collapse(),
            "unpredictable_nuisance":nuisance(),
            "scope":"deterministic toy mathematics only; no JEPA pretraining, video understanding or robot manipulation"}
    print(json.dumps(result,ensure_ascii=False,indent=2))
