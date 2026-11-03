"""Run the locked training queue; observe phase consistency during its smoke only."""
import math
import traceback

from scripts import run_g1_cause_audit_windows_20261026 as recovery


def main():
    training = recovery.install()
    original_train = training.train
    def observed_train(arm, seed, mode, *, smoke=False):
        if not smoke:
            return original_train(arm, seed, mode, smoke=False)
        recovery.bootstrap()
        import torch
        from mjlab.rl import RslRlVecEnvWrapper
        original_step = RslRlVecEnvWrapper.step
        record = dict(status='RUNNING', at=training.audit.now(), steps=0,
            compared_environment_frames=0, reset_transitions=0,
            max_abs_error={'actor':0.,'critic':0.},
            verification_script_sha256=training.audit.sha(__file__),
            scope='Read returned actor/critic observations and cached E output after actual PPO steps; '
                  'no additional observe(), action, RNG draw or physics step.')
        def checked_step(env, actions):
            result = original_step(env, actions)
            raw = env.unwrapped
            runtime = raw._new_e_online_phase
            manager = raw.observation_manager
            assert bool((runtime.last_stamp == raw.common_step_counter).all())
            for group in ('actor','critic'):
                names = manager.active_terms[group]
                dims = manager.group_obs_term_dim[group]
                offset = sum(math.prod(dim) for name,dim in zip(names,dims) if
                             names.index(name) < names.index('phase'))
                width = math.prod(dims[names.index('phase')])
                assert width == 2
                value = result[0][group][:,offset:offset+width]
                error = float((value-runtime.output).abs().max())
                record['max_abs_error'][group] = max(record['max_abs_error'][group], error)
                torch.testing.assert_close(value,runtime.output,rtol=0.,atol=1e-6)
            record['steps'] += 1
            record['compared_environment_frames'] += raw.num_envs
            record['reset_transitions'] += int(result[2].sum())
            return result
        RslRlVecEnvWrapper.step = checked_step
        try:
            result = original_train(arm,seed,mode,smoke=True)
            assert record['steps'] == 2*24
            assert record['compared_environment_frames'] == 2*24*64
            record.update(status='PASS',ended_at=training.audit.now())
            return result
        except BaseException:
            record.update(status='FAILED',traceback=traceback.format_exc())
            raise
        finally:
            RslRlVecEnvWrapper.step = original_step
            training.write(training.OUT/'smoke_sampling_phase_consistency.json',record)
    training.train = observed_train
    training.queue()


if __name__ == '__main__':
    main()
