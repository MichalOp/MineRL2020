#mostly from baseline

from logging import getLogger
import os

import tqdm
import numpy as np
from sklearn.cluster import KMeans
import joblib
import minerl

logger = getLogger(__name__)


class _KMeansCacheNotFound(FileNotFoundError):
    pass

default_n = 120
default_seed = 1

class BoundedLengthMemory:
    def __init__(self, maxlen, random_state):
        self.maxlen = maxlen
        self.t = 0
        self._rand = np.random.RandomState(random_state)
        self.memory = []

    def __call__(self):
        return np.array(self.memory)

    def append(self, action):
        self.t += 1
        if self.maxlen is None or len(self.memory) < self.maxlen:
            self.memory.append(action)
        else:
            idx = self._rand.randint(self.t)
            if idx < self.maxlen:
                self.memory[idx] = action


def cached_kmeans(cache_dir, env_id, n_clusters=default_n, random_state=default_seed,
                  subtask_reward_max=None, maxlen_each=None,
                  only_vector_converter=False):
    if cache_dir is None:  # ignore cache
        logger.info('Load dataset & do kmeans')
        kmeans = _do_kmeans(env_id=env_id, n_clusters=n_clusters,
                            random_state=random_state,
                            subtask_reward_max=subtask_reward_max,
                            only_vector_converter=only_vector_converter)
    else:
        if subtask_reward_max is None:
            name_subtask_reward_max = ''
        else:
            name_subtask_reward_max = '_{}'.format(subtask_reward_max)
        if only_vector_converter:
            filename = 'kmeans_vector_converter{}.joblib'.format(name_subtask_reward_max)  # noqa
        elif maxlen_each is not None:
            filename = 'kmeans_balanced_{}{}.joblib'.format(
                           maxlen_each, name_subtask_reward_max)
        else:
            filename = 'kmeans{}.joblib'.format(name_subtask_reward_max)
        filepath = os.path.join(cache_dir, env_id, f'n_clusters_{n_clusters}', f'random_state_{random_state}', filename)
        try:
            kmeans = _load_kmeans_result_cache(filepath)
            logger.info('found kmeans cache')
        except _KMeansCacheNotFound:
            logger.info('kmeans cache not found. Load dataset & do kmeans & save result as cache')
            kmeans = _do_kmeans(env_id=env_id, n_clusters=n_clusters,
                                random_state=random_state,
                                subtask_reward_max=subtask_reward_max,
                                maxlen_each=maxlen_each,
                                only_vector_converter=only_vector_converter)
            _save_kmeans_result_cache(kmeans, filepath)
    return kmeans


def _do_kmeans(env_id, n_clusters, random_state, subtask_reward_max,
               maxlen_each, only_vector_converter):
    logger.debug(f'loading data...')
    dat = minerl.data.make(env_id)
    if subtask_reward_max is None and maxlen_each is None:
        act_vectors = []
        for ob, act, _, next_ob, _ in tqdm.tqdm(dat.batch_iter(batch_size=16, seq_len=32, num_epochs=1, preload_buffer_size=32, seed=random_state)):
            if only_vector_converter:
                if np.allclose(ob['vector'], next_ob['vector']):
                    # Ignore the case when the action does not change observation$vector.
                    continue
            act_vectors.append(act['vector'])
        acts = np.concatenate(act_vectors).reshape(-1, 64)
    else:
        episode_names = dat.get_trajectory_names()
        mem_normal = BoundedLengthMemory(maxlen=maxlen_each, random_state=random_state)
        mem_vc = BoundedLengthMemory(maxlen=maxlen_each, random_state=random_state)
        for episode_name in episode_names:
            traj = dat.load_data(episode_name)
            dn = False
            current_reward_sum = 0
            while not dn:
                ob, act, rw, next_ob, dn = next(traj)
                current_reward_sum += rw
                if subtask_reward_max is not None and current_reward_sum >= subtask_reward_max:
                    dn = True
                if np.allclose(ob['vector'], next_ob['vector']):
                    # Ignore the case when the action does not change observation$vector.
                    mem_normal.append(act['vector'])
                else:
                    mem_vc.append(act['vector'])
        if only_vector_converter:
            acts = mem_vc().reshape(-1, 64)
        else:
            acts = np.concatenate((mem_normal(), mem_vc()), axis=0).reshape(-1, 64)
    logger.debug(f'loading data... done.')
    logger.debug(f'executing keamns...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(acts)
    logger.debug(f'executing keamns... done.')
    return kmeans


# def _describe_kmeans_result(kmeans):
#     result = [(obf_a, minerl.herobraine.envs.MINERL_TREECHOP_OBF_V0.unwrap_action({'vector': obf_a})) for obf_a in kmeans.cluster_centers_]
#     logger.debug(result)
#     return result


def _save_kmeans_result_cache(kmeans, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(kmeans, filepath)
    logger.info(f'saved kmeans {filepath}')


def _load_kmeans_result_cache(filepath):
    if not os.path.exists(filepath):
        raise _KMeansCacheNotFound
    logger.debug(f'loading kmeans {filepath}')
    return joblib.load(filepath)
