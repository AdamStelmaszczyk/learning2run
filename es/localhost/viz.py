import click


@click.command()
@click.argument('env_id')
@click.argument('policy_type')
@click.argument('policy_file')
@click.option('--record', is_flag=True, default=False)
@click.option('--stochastic', is_flag=True, default=False)
@click.option('--extra_kwargs')
@click.option('--env-extra-kwargs')
def main(env_id, policy_type, policy_file, record, stochastic, extra_kwargs, env_extra_kwargs):
    import gym
    from gym import wrappers
    import tensorflow as tf
    import numpy as np
    from es.es_distributed import policies

    if extra_kwargs:
        import json
        extra_kwargs = json.loads(extra_kwargs)

    if env_extra_kwargs:
        import json
        env_extra_kwargs = json.loads(env_extra_kwargs)

    if env_id == 'osim.env.run:RunEnv':
        from osim.env.run import RunEnv
        if env_extra_kwargs:
            env = RunEnv(True, **env_extra_kwargs)
        else:
            env = RunEnv(True)

    else:
        env = gym.make(env_id)
    if record:
        import uuid
        env = wrappers.Monitor(env, '/tmp/' + str(uuid.uuid4()), force=True)


    with tf.Session():
        policy = getattr(policies, policy_type)
        pi = policy.Load(policy_file, extra_kwargs=extra_kwargs)
        while True:
            rews, t = pi.rollout(env, render=True, random_stream=np.random if stochastic else None)
            print('return={:.4f} len={}'.format(rews.sum(), t))

            if record:
                env.close()
                return


if __name__ == '__main__':
    main()
