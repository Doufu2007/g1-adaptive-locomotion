"""A single causal reference oscillator, frozen before the stage-B experiment."""
import math
import torch

class ContactPhaseReference:
    def __init__(self,num_envs,device,targets,dt=.02):
        if num_envs<1 or dt!=.02 or len(targets)!=2 or not all(math.isfinite(x) for x in targets):
            raise ValueError('invalid reference configuration')
        self.num_envs=num_envs;self.device=device;self.dt=dt
        self.targets=torch.tensor(targets,device=device,dtype=torch.float64)
        self.theta=torch.zeros(num_envs,device=device,dtype=torch.float64)
        self.period=torch.full_like(self.theta,.6);self.pending=torch.zeros_like(self.theta)
        self.stable=torch.zeros(num_envs,2,device=device,dtype=torch.bool)
        self.candidate=self.stable.clone();self.count=torch.zeros(num_envs,2,device=device,dtype=torch.long)
        self.last_touch=torch.full_like(self.count,-1);self.touchdowns=torch.zeros_like(self.count)
        self.last_stamp=torch.full((num_envs,),-1,device=device,dtype=torch.long)
        self.last_episode=self.last_stamp.clone();self.initialized=torch.zeros(num_envs,device=device,dtype=torch.bool)
    def reset(self,ids):
        if ids is None:ids=torch.arange(self.num_envs,device=self.device)
        self.initialized[ids]=False
    def update(self,stamp,episode_steps,contact):
        if contact.shape!=(self.num_envs,2) or episode_steps.shape!=(self.num_envs,):
            raise ValueError('invalid input shape')
        if not bool(torch.isfinite(contact).all()) or not bool(((contact==0)|(contact==1)).all()):
            raise ValueError('contact must be finite binary')
        if not bool(torch.isfinite(episode_steps).all()) or bool((episode_steps<0).any()):raise ValueError('invalid episode step')
        if not bool((episode_steps==episode_steps.long()).all()) or not isinstance(stamp,int):raise ValueError('integer timestamps required')
        ep=episode_steps.long();raw=contact.bool()
        reset=(~self.initialized)|(ep<self.last_episode)
        # A runner may randomize episode lengths once before its first rollout.
        reset|=(self.last_stamp==stamp)&(ep!=self.last_episode)
        reset|=(self.last_stamp==0)&(stamp==1)&(self.last_episode==0)&(ep>1)
        fresh=(self.last_stamp!=stamp)&(~reset)
        if bool((fresh&((self.last_stamp+1!=stamp)|(ep!=self.last_episode+1))).any()):
            raise ValueError('nonreset reference step gap')
        if bool(reset.any()):
            self.theta[reset]=ep[reset].double()*self.dt*2*math.pi/.6;self.period[reset]=.6;self.pending[reset]=0
            self.stable[reset]=raw[reset];self.candidate[reset]=raw[reset];self.count[reset]=3
            self.last_touch[reset]=-1;self.touchdowns[reset]=0;self.initialized[reset]=True
        if bool(fresh.any()):
            base=2*math.pi*self.dt/self.period
            correction=(self.pending/5).clamp(-base*.25,base*.25)
            self.theta[fresh]+=base[fresh]+correction[fresh];self.pending[fresh]-=correction[fresh]
            same=raw==self.candidate
            counts=torch.where(same,(self.count+1).clamp(max=3),torch.ones_like(self.count))
            self.count[fresh]=counts[fresh];self.candidate[fresh]=raw[fresh]
            confirmed=(self.count>=3)&(self.candidate!=self.stable)&fresh[:,None]
            touchdown=confirmed&self.candidate
            self.stable[confirmed]=self.candidate[confirmed]
            single=touchdown.sum(dim=1)==1
            for foot in range(2):
                event=single&touchdown[:,foot]
                interval=(ep-self.last_touch[:,foot]).double()*self.dt
                valid=event&(self.last_touch[:,foot]>=0)&(interval>=.3)&(interval<=1.2)
                self.period[valid]=(.8*self.period[valid]+.2*interval[valid]).clamp(.4,1.)
                error=torch.remainder(self.targets[foot]-self.theta+math.pi,2*math.pi)-math.pi
                self.pending[event]=(.5*error[event]).clamp(-math.pi/6,math.pi/6)
                self.last_touch[event,foot]=ep[event];self.touchdowns[event,foot]+=1
        changed=reset|fresh;self.last_stamp[changed]=stamp;self.last_episode[changed]=ep[changed]
        return torch.stack((torch.sin(self.theta),torch.cos(self.theta)),dim=-1).float()


def calibrate_batches(packs):
    """Pool training-only confirmed touchdown phases; never optimize control scores."""
    samples=[[],[]];counts=[]
    for pack in packs:
        critic=pack['observations']['critic'].cpu();done=pack['buffers']['dones'].cpu()
        if critic.shape!=(24,64,113) or done.shape!=(24,64,1):raise ValueError('unexpected training layout')
        phase=critic[:,:,9:11].double();contact=critic[:,:,105:107]
        if not bool(torch.isfinite(critic).all()) or not bool(torch.isfinite(done).all()):raise ValueError('nonfinite training capture')
        if not bool(((done==0)|(done==1)).all()):raise ValueError('invalid terminal flag')
        if not torch.allclose(phase.norm(dim=-1),torch.ones(24,64,dtype=torch.float64),atol=2e-6,rtol=0):
            raise ValueError('training phase must be an unnormalized unit vector')
        state=ContactPhaseReference(64,'cpu',(0.,math.pi));ep=torch.zeros(64,dtype=torch.long);per=[0,0]
        for t in range(24):
            if t:
                ep+=1;ids=done[t-1,:,0].bool().nonzero().flatten();state.reset(ids);ep[ids]=0
            before=state.touchdowns.clone();state.update(t,ep,contact[t])
            for foot in range(2):
                event=state.touchdowns[:,foot]>before[:,foot]
                theta=torch.atan2(phase[t,event,0],phase[t,event,1])
                samples[foot].extend(theta.tolist());per[foot]+=int(event.sum())
        counts.append(per)
    targets=[];concentrations=[]
    for values in samples:
        if not values:targets.append(0.);concentrations.append(0.);continue
        x=torch.tensor(values,dtype=torch.float64)
        sine=float(x.sin().mean());cosine=float(x.cos().mean())
        targets.append(math.atan2(sine,cosine));concentrations.append(math.hypot(sine,cosine))
    event_counts=list(map(len,samples))
    return dict(targets=targets,concentrations=concentrations,event_counts=event_counts,per_capture_counts=counts,
        passed=min(event_counts)>=50 and min(concentrations)>=.8,minimum_events=50,minimum_concentration=.8,
        phase_index=[9,11],contact_index=[105,107],source_role='training collection only; no evaluation inputs',
        event_timestamp='causal 3-frame confirmation; no backdating')


def calibrate_trajectory(phase,contact,dones,start_step):
    """Fixed burn-in on a training-only source-policy trajectory, one attempt."""
    if phase.ndim!=3 or phase.shape[-1]!=2 or contact.shape!=phase.shape or dones.shape!=phase.shape[:2]:
        raise ValueError('invalid continuous calibration layout')
    if not 0<=start_step<phase.shape[0] or not bool(torch.isfinite(phase).all()):raise ValueError('invalid burnin/phase')
    if not torch.allclose(phase.double().norm(dim=-1),torch.ones_like(phase[:,:,0],dtype=torch.float64),atol=2e-6,rtol=0):
        raise ValueError('non-unit reference calibration phase')
    n=phase.shape[1];s=ContactPhaseReference(n,'cpu',(0.,math.pi));ep=torch.zeros(n,dtype=torch.long);samples=[[],[]]
    for t in range(len(phase)):
        if t:
            ep+=1;ids=dones[t-1].bool().nonzero().flatten();s.reset(ids);ep[ids]=0
        before=s.touchdowns.clone();s.update(t,ep,contact[t])
        if t<start_step:continue
        for foot in range(2):
            event=s.touchdowns[:,foot]>before[:,foot]
            samples[foot].extend(torch.atan2(phase[t,event,0],phase[t,event,1]).tolist())
    targets=[];concentrations=[]
    for values in samples:
        if not values:targets.append(0.);concentrations.append(0.);continue
        x=torch.tensor(values,dtype=torch.float64);si=float(x.sin().mean());co=float(x.cos().mean())
        targets.append(math.atan2(si,co));concentrations.append(math.hypot(si,co))
    counts=list(map(len,samples))
    return dict(targets=targets,concentrations=concentrations,event_counts=counts,passed=min(counts)>=50 and min(concentrations)>=.8,
        start_step=start_step,minimum_events=50,minimum_concentration=.8,source_role='new training-only calibration; no PPO updates')
