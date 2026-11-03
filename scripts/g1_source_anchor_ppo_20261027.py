"""Native feedforward single-GPU PPO plus one frozen-source KL term.
Update equations retain the rsl_rl native PPO implementation (BSD-3-Clause,
ETH Zurich/NVIDIA, 2021-2026). Unsupported extensions fail explicitly.
"""
import copy,math
import torch
from rsl_rl.algorithms import PPO

class SourceAnchoredPPO(PPO):
 def __init__(self,*args,anchor_coefficient=.1,**kwargs):
  super().__init__(*args,**kwargs)
  if not math.isfinite(anchor_coefficient) or anchor_coefficient<0:raise ValueError('invalid source anchor coefficient')
  if self.actor.is_recurrent or self.critic.is_recurrent or self.rnd or self.symmetry or self.is_multi_gpu:
   raise ValueError('source anchor supports feedforward single GPU without RND/symmetry')
  self.anchor_coefficient=float(anchor_coefficient)
  self.source_teacher=copy.deepcopy(self.actor).eval().requires_grad_(False)
  self.last_source_kl=[]

 def source_kl(self,observations,current_params):
  with torch.no_grad():
   teacher=self.source_teacher
   teacher.distribution.update(teacher.mlp(teacher.get_latent(observations)))
   params=tuple(x.detach() for x in teacher.output_distribution_params)
  return self.actor.distribution.kl_divergence(params,current_params)

 def save(self):
  payload=super().save()
  payload.update(source_teacher_state_dict=self.source_teacher.state_dict(),source_anchor_coefficient=self.anchor_coefficient)
  return payload

 def load(self,loaded_dict,load_cfg,strict):
  has_teacher='source_teacher_state_dict' in loaded_dict
  has_coefficient='source_anchor_coefficient' in loaded_dict
  if has_teacher!=has_coefficient:raise ValueError('incomplete source teacher checkpoint')
  if has_teacher:
   if loaded_dict['source_anchor_coefficient']!=self.anchor_coefficient:raise ValueError('source anchor coefficient changed on restore')
  elif load_cfg is None or {k for k,v in load_cfg.items() if v}!={'actor'}:
   raise ValueError('missing source teacher; only initial actor-only bootstrap is allowed')
  result=super().load(loaded_dict,load_cfg,strict)
  if has_teacher:
   self.source_teacher.load_state_dict(loaded_dict['source_teacher_state_dict'],strict=True)
  else:
   self.source_teacher.load_state_dict(self.actor.state_dict(),strict=True)
  self.source_teacher.eval().requires_grad_(False)
  return result

 def update(self):
  mean_value_loss=0;mean_surrogate_loss=0;mean_entropy=0;mean_source_kl=0;self.last_source_kl=[]
  generator=self.storage.mini_batch_generator(self.num_mini_batches,self.num_learning_epochs)
  for batch in generator:
   original_batch_size=batch.observations.batch_size[0]
   if self.normalize_advantage_per_mini_batch:
    with torch.no_grad():batch.advantages=(batch.advantages-batch.advantages.mean())/(batch.advantages.std()+1e-8)
   self.actor(batch.observations,masks=batch.masks,hidden_state=batch.hidden_states[0],stochastic_output=True)
   actions_log_prob=self.actor.get_output_log_prob(batch.actions)
   values=self.critic(batch.observations,masks=batch.masks,hidden_state=batch.hidden_states[1])
   distribution_params=tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
   entropy=self.actor.output_entropy[:original_batch_size]
   if self.desired_kl is not None and self.schedule=='adaptive':
    with torch.inference_mode():
     kl=self.actor.get_kl_divergence(batch.old_distribution_params,distribution_params);kl_mean=torch.mean(kl)
     if kl_mean>self.desired_kl*2.0:self.learning_rate=max(1e-5,self.learning_rate/1.5)
     elif kl_mean<self.desired_kl/2.0 and kl_mean>0.0:self.learning_rate=min(1e-2,self.learning_rate*1.5)
     for param_group in self.optimizer.param_groups:param_group['lr']=self.learning_rate
   ratio=torch.exp(actions_log_prob-torch.squeeze(batch.old_actions_log_prob))
   surrogate=-torch.squeeze(batch.advantages)*ratio
   surrogate_clipped=-torch.squeeze(batch.advantages)*torch.clamp(ratio,1.0-self.clip_param,1.0+self.clip_param)
   surrogate_loss=torch.max(surrogate,surrogate_clipped).mean()
   if self.use_clipped_value_loss:
    value_clipped=batch.values+(values-batch.values).clamp(-self.clip_param,self.clip_param)
    value_losses=(values-batch.returns).pow(2);value_losses_clipped=(value_clipped-batch.returns).pow(2)
    value_loss=torch.max(value_losses,value_losses_clipped).mean()
   else:value_loss=(batch.returns-values).pow(2).mean()
   loss=surrogate_loss+self.value_loss_coef*value_loss-self.entropy_coef*entropy.mean()
   if self.anchor_coefficient:
    anchor=self.source_kl(batch.observations,distribution_params)
    source_kl=anchor.mean();loss=loss+self.anchor_coefficient*source_kl
    self.last_source_kl.append(dict(mean=float(source_kl.detach()),maximum=float(anchor.detach().max()),beta=self.anchor_coefficient))
    mean_source_kl+=source_kl.item()
   self.optimizer.zero_grad();loss.backward()
   torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
   torch.nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm)
   self.optimizer.step()
   mean_value_loss+=value_loss.item();mean_surrogate_loss+=surrogate_loss.item();mean_entropy+=entropy.mean().item()
  n=self.num_learning_epochs*self.num_mini_batches;self.storage.clear()
  return dict(value=mean_value_loss/n,surrogate=mean_surrogate_loss/n,entropy=mean_entropy/n,source_kl=mean_source_kl/n,
              source_kl_max=max((r['maximum'] for r in self.last_source_kl),default=0.),anchor_coefficient=self.anchor_coefficient)
