#mostly from baseline

from logging import getLogger
import os

import tqdm
import numpy as np
from sklearn.cluster import KMeans
import joblib
import minerl
from minerl.data import DataPipeline

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

def absolute_file_paths(directory):
    return [os.path.join(directory, path) for path in os.listdir(directory)]


def _do_kmeans(env_id, n_clusters, random_state, subtask_reward_max,
               maxlen_each, only_vector_converter):
    logger.debug(f'loading data...')
    files = absolute_file_paths('data/MineRLObtainDiamondVectorObf-v0')
    # dat = minerl.data.make(env_id)
    act_vectors = []
    for f in files:
        try:
            d = DataPipeline._load_data_pyfunc(f, -1, None)
        except:
            continue
        obs, act, reward, nextobs, done = d
        act_vectors.append(act['vector'])
    acts = np.concatenate(act_vectors).reshape(-1, 64)

    logger.debug(f'loading data... done.')
    logger.debug(f'executing keamns...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(acts)
    logger.debug(f'executing keamns... done.')
    return kmeans



def _save_kmeans_result_cache(kmeans, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(kmeans, filepath)
    logger.info(f'saved kmeans {filepath}')


def _load_kmeans_result_cache(filepath):
    if not os.path.exists(filepath):
        raise _KMeansCacheNotFound
    logger.debug(f'loading kmeans {filepath}')
    return joblib.load(filepath)
