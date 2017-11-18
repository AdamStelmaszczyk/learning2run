from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque, defaultdict
from math import sqrt


PREPROCESS = True
OBSERVATION_DIM = 82


def obstacle_x_importance(x):
    return -(1.0 / 3) * x + (5.0 / 6) if -0.5 < x < 2.5 else 0


def set_obstacles(result, dx, step):
    assert len(result) == 42
    assert step >= 1

    x = result[36]
    y = result[37]
    r = result[38]

    if step == 1:
        # first observation has incorrect obstacle info, skip it
        # initial estimate, no information
        set_obstacles.prev = [-2.0, 0.0, 0.1]
        set_obstacles.next = [2.0, 0.0, 0.1]
    elif step == 2:
        set_obstacles.next = [x, y, r]
    else:
        if y != set_obstacles.next[1] or r != set_obstacles.next[2]:  # new found obstacle
            if x == 100:  # in fact it's a marker saying that there are no more obstacles ahead
                x = -2.0
            set_obstacles.prev = list(set_obstacles.next)
            set_obstacles.next = [x, y, r]
        else:
            set_obstacles.next[0] -= dx
        set_obstacles.prev[0] -= dx

    result[36:39] = set_obstacles.prev
    result[39:42] = set_obstacles.next

    # normalization
    for i in [36, 39]:
        result[i] = obstacle_x_importance(result[i])
    for i in [37, 40]:
        result[i] /= 0.14433756729740643
        result[i] *= result[i - 1]
    for i in [38, 41]:
        result[i] = (result[i] - 0.1) / 0.05
        result[i] *= result[i - 2]


set_obstacles.prev = None
set_obstacles.next = None


def add_velocities(result, old, indices):
    def pelvis_x_or_y(which, i):
        return which[i % 2 + 1]

    def pelvis_v_x_or_y(i):
        return result[i % 2 + 4]

    for i in indices:
        new_abs_pos = result[i] + pelvis_x_or_y(result, i)
        old_abs_pos = old[i] + pelvis_x_or_y(old, i)
        v = (new_abs_pos - old_abs_pos) / 0.01
        v_relative = v - pelvis_v_x_or_y(i)
        result.append(v_relative)


def add_accelerations(result, old):
    def acc(i):
        return result[i] - old[i]

    pelvis_a_r = acc(3)
    pelvis_a_x = acc(4)
    pelvis_a_y = acc(5)

    result.append(pelvis_a_r)
    result.append(pelvis_a_x)
    result.append(pelvis_a_y)

    for i in range(12, 18):
        result.append(acc(i) - pelvis_a_r)

    for i in [20, 21, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]:
        pelvis_a_x_or_y = pelvis_a_x if i % 2 == 0 else pelvis_a_y
        result.append(acc(i) - pelvis_a_x_or_y)

    for i in range(54, 57):  # pelvis accelerations
        result[i] = double_sqrt(result[i])

    for i in range(57, 63):  # hip, knee, ankle a.r
        result[i] = double_sqrt(result[i] / 4.0)

    for i in range(63, 65):  # mass accelerations
        result[i] = double_sqrt(result[i])

    for i in [65, 67, 69, 71, 73, 75]:  # head, torso, toes, talus a.x
        result[i] = double_sqrt(result[i])

    for i in [66, 68, 70, 72, 74, 76]:  # head, torso, toes, talus a.y
        result[i] = double_sqrt(result[i]) / 0.6


def add_ground_touch(result, indices):
    for i in indices:
        abs_y = result[i] + result[2]
        touch = max(1.0 - 20.0 * abs_y, -1.0)
        result.append(touch)


def softsign(x):
    return x / (1.0 + abs(x))


def add_clear_ahead(result, pelvis_x):
    clear_ahead = softsign(pelvis_x - 28.0)
    return result.append(clear_ahead)


# based on https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
def running_mean_and_std(i, x, step):
    delta = x - preprocess.mean[i]
    preprocess.mean[i] += delta / step
    delta2 = x - preprocess.mean[i]
    preprocess.m2[i] += delta * delta2
    if step < 2:
        std = 1
    else:
        std = preprocess.m2[i] / (step - 1)
    return preprocess.mean[i], std


def auto_normalize(result, indices, step):
    for i in indices:
        preprocess.min[i] = min(preprocess.min[i], result[i])
        preprocess.max[i] = max(preprocess.max[i], result[i])
        mean, std = running_mean_and_std(i, result[i], step)
        # result[i] = (result[i] - mean) / std


def double_sqrt(x):
    return sqrt(x) if x >= 0 else -sqrt(-x)


def preprocess(x, step, verbose=False):
    if not PREPROCESS:
        return x

    pelvis_r = x[0]
    pelvis_x = x[1]
    pelvis_y = x[2]
    pelvis_v_r = x[3]
    pelvis_v_x = x[4]
    pelvis_v_y = x[5]
    result = [
        pelvis_r,           # pelvis.r
        pelvis_x,           # pelvis.x
        pelvis_y,           # pelvis.y

        pelvis_v_r,         # pelvis.v.r
        pelvis_v_x,         # pelvis.v.x
        pelvis_v_y,         # pelvis.v.y

        x[6] - pelvis_r,    # hip.right.r
        x[7] - pelvis_r,    # knee.right.r
        x[8] - pelvis_r,    # ankle.right.r
        x[9] - pelvis_r,    # hip.left.r
        x[10] - pelvis_r,   # knee.left.r
        x[11] - pelvis_r,   # ankle.left.r

        x[12] - pelvis_v_r,  # hip.right.v.r
        x[13] - pelvis_v_r,  # knee.right.v.r
        x[14] - pelvis_v_r,  # ankle.right.v.r
        x[15] - pelvis_v_r,  # hip.left.v.r
        x[16] - pelvis_v_r,  # knee.left.v.r
        x[17] - pelvis_v_r,  # ankle.left.v.r

        x[18] - pelvis_x,     # mass.x
        x[19] - pelvis_y,     # mass.y
        x[20] - pelvis_v_x,   # mass.v.x
        x[21] - pelvis_v_y,   # mass.v.y

        x[22] - pelvis_x,     # head.x
        x[23] - pelvis_y,     # head.y

        # x[24],              # duplicated pelvis.x
        # x[25],              # duplicated pelvis.y

        x[26] - pelvis_x,  # torso.x
        x[27] - pelvis_y,  # torso.y

        x[28] - pelvis_x,   # toes.left.x
        x[29] - pelvis_y,   # toes.left.y
        x[30] - pelvis_x,   # toes.right.x
        x[31] - pelvis_y,   # toes.right.y

        x[32] - pelvis_x,   # talus.left.x
        x[33] - pelvis_y,   # talus.left.y
        x[34] - pelvis_x,   # talus.right.x
        x[35] - pelvis_y,   # talus.right.y

        x[36],              # psoas.left.strength
        x[37],              # psoas.right.strength

        x[38],              # next obstacle x distance from pelvis
        x[39],              # next obstacle y position of the center
        x[40],              # next obstacle radius
    ]
    result.extend([0] * 3)

    if preprocess.old is None:
        preprocess.old = list(result)
        zeros = OBSERVATION_DIM - len(preprocess.old)
        preprocess.old.extend([0] * zeros)

    dx = pelvis_x - preprocess.old[1]
    set_obstacles(result, dx, step)
    add_velocities(result, preprocess.old, [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])
    add_accelerations(result, preprocess.old)
    add_ground_touch(result, [27, 29, 31, 33])  # toes and talus y
    add_clear_ahead(result, pelvis_x)

    preprocess.old = list(result)

    # normalize these in the end, because it was used before for relative calculations
    result[1] /= 100.0
    result[2] = (result[2] - 0.9) / 0.1

    result[3] = double_sqrt(result[3] - 0.3)  # pelvis.v.r
    result[4] = double_sqrt(result[4] - 1.0)  # pelvis.v.x
    result[5] = double_sqrt(result[5] - 0.3)  # pelvis.v.y

    for i in range(6, 12):  # hip, knee, ankle r
        result[i] = double_sqrt(result[i])

    for i in range(12, 18):  # hip, knee, ankle v.r
        result[i] = double_sqrt(result[i] / 5.0)

    result[18] = (result[18] + 0.14) / 0.05  # mass.x
    result[19] = (result[19] - 0.07) / 0.03  # mass.y

    for i in range(20, 22):  # mass velocities
        result[i] = double_sqrt(result[i])

    result[22] = (result[22] + 0.15) / 0.25   # head.x
    result[23] = (result[23] - 0.61) / 0.02   # head.y

    result[24] = (result[24] + 0.10) / 0.02   # torso.x
    result[25] = (result[25] - 0.08) / 0.02   # torso.y

    for i in [26, 28, 30, 32]:  # toes, talus x
        result[i] = (result[i] + 0.1)

    for i in [27, 29, 31, 33]:  # toes, talus y
        result[i] = (result[i] + 0.9) / 0.3

    for i in range(34, 36):  # psoas
        result[i] = (result[i] - 1.0) / 0.2

    for i in range(42, 46):  # head, torso v
        result[i] = double_sqrt(result[i])

    for i in [46, 48, 50, 52]:  # toes, talus v.x
        result[i] = double_sqrt(result[i]) / 2.5

    for i in [47, 49, 51, 53]:  # toes, talus v.y
        result[i] = double_sqrt(result[i])

    if verbose:
        auto_normalize(result, range(OBSERVATION_DIM), step)
        print("---")
        print(step)
        print("")
        print("x %s" % x)
        print("")
        print("result %s" % result)
        print("")
        print("min %s" % preprocess.min)
        print("")
        print("max %s" % preprocess.max)
        print("")
        print("mean %s" % preprocess.mean)
        print("")
        if step > 2: print("std %s" % map(lambda m: m / (step - 1), preprocess.m2))
        print("")

    assert len(result) == OBSERVATION_DIM
    return result

preprocess.old = None
preprocess.min = [float('inf')] * OBSERVATION_DIM
preprocess.max = [float('-inf')] * OBSERVATION_DIM
preprocess.mean = [0] * OBSERVATION_DIM
preprocess.m2 = [0] * OBSERVATION_DIM


def traj_segment_generator(pi, env, horizon, stochastic, verbose=False):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()
    ob = preprocess(ob, step=1)

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        ob = preprocess(ob, step=cur_ep_len + 2, verbose=verbose)

        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            ob = preprocess(ob, step=cur_ep_len + 1, verbose=verbose)
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(
    env,
    policy_func,
    timesteps_per_batch,  # timesteps per actor per update
    clip_param,  # clipping parameter epsilon
    entcoeff,  # entropy coeff
    optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
    gamma, lam,  # advantage estimation
    max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
    callback=None,  # you can do anything in the callback, since it takes locals(), globals()
    adam_epsilon=1e-5,
    schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
    verbose=False,
):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = U.mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
                                                   for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True, verbose=verbose)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 1
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    history = defaultdict(list)
    def log_and_store(key, value):
        """
            Logs info on stdout and stores value in history map

            TODO create "History" class for openai.baselines
        """
        logger.record_tabular(key, value)
        history[key].append(value)
        return history

    try:
        while True:
            if callback: callback(locals(), globals())
            if max_timesteps and timesteps_so_far >= max_timesteps:
                break
            elif max_episodes and episodes_so_far >= max_episodes:
                break
            elif max_iters and iters_so_far > max_iters:
                break
            elif max_seconds and time.time() - tstart >= max_seconds:
                break

            if schedule == 'constant':
                cur_lrmult = 1.0
            elif schedule == 'linear':
                cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
            else:
                raise NotImplementedError("Not available schedule: %s" % schedule)

            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.log("********** Iteration %i ************"%iters_so_far)

            seg = next(seg_gen)
            add_vtarg_and_adv(seg, gamma, lam)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
            optim_batchsize = optim_batchsize or ob.shape[0]

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

            assign_old_eq_new() # set old parameter values to new parameter values
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.log("Optimizing...")
                logger.log(fmt_row(13, loss_names))
            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    loss = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    newlosses = loss[:len(loss) - 1]
                    gradient = loss[len(loss) - 1]
                    adam.update(gradient, optim_stepsize * cur_lrmult)
                    losses.append(newlosses)
                if MPI.COMM_WORLD.Get_rank() == 0:
                    logger.log(fmt_row(13, np.mean(losses, axis=0)))

            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.log("Evaluating losses...")
            losses = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                losses.append(newlosses)
            meanlosses,_,_ = mpi_moments(losses, axis=0)
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.log(fmt_row(13, meanlosses))
            for (lossval, name) in zipsame(meanlosses, loss_names):
                log_and_store("loss_"+name, lossval)
            log_and_store("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
            lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
            lens, rews = map(flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            log_and_store("EpLenMean", np.mean(lenbuffer))
            log_and_store("EpRewMean", np.mean(rewbuffer))
            log_and_store("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1
            log_and_store("EpisodesSoFar", episodes_so_far)
            log_and_store("TimestepsSoFar", timesteps_so_far)
            log_and_store("TimeElapsed", time.time() - tstart)
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.dump_tabular()
    except KeyboardInterrupt:  # handles Ctrl + C
        pass

    return history

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
