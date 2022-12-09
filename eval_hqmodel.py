# ------------------------------------------------------------------------------------
# the code is modified from
# https://github.com/kakaobrain/rq-vae-transformer/blob/main/compute_metrics.py
# ------------------------------------------------------------------------------------

import logging
import argparse
import numpy as np
from pathlib import Path
from hqvae.utils.fid_utils import compute_activations_from_files, frechet_distance, mean_covar_numpy
from hqvae.utils.prdc import compute_prdc


def compute_fid_prdc(result_path,
                     ref_stat_path=None,
                     ref_feature_path=None,
                     ):

    act_path = Path(result_path) / 'acts.npz'
    if not act_path.exists():
        acts = compute_activations_from_files(result_path)
        mu, sigma = mean_covar_numpy(acts)
        np.savez(act_path, acts=acts, mu=mu, sigma=sigma)
        logging.info(f'activations saved to {act_path.as_posix()}')
    else:
        logging.info(f'precomputed activations found: {act_path.as_posix()}')

    acts_fake = np.load(act_path)

    metrics = {}

    if ref_stat_path:
        stats_ref = np.load(ref_stat_path)
        mu_ref, sigma_ref = stats_ref['mu'], stats_ref['sigma']
        logging.info(f'dataset stats loaded from {ref_stat_path}')

        mu_fake, sigma_fake = acts_fake['mu'], acts_fake['sigma']

        logging.info('computing fid...')
        fid = frechet_distance(mu_ref, sigma_ref, mu_fake, sigma_fake)
        metrics['fid'] = fid

        logging.info('FID: {fid:.4f}'.format(fid=fid))

    if ref_feature_path:

        fake_features = acts_fake['acts']
        logging.info(f'activations loaded from {act_path.as_posix()}')
        logging.info(f'shape: {fake_features.shape}')

        ref_features = np.load(ref_feature_path)['acts']
        logging.info(f'activations loaded from {ref_feature_path}')
        logging.info(f'shape: {ref_features.shape}')

        logging.info('computing prdc...')
        prdc = compute_prdc(ref_features, fake_features, nearest_k=3)

        logging.info(
            'P={p:.4f}, R={r:.4f}, D={d:.4f}, C={c:.4f}'.format(
                p=prdc['precision'],
                r=prdc['recall'],
                d=prdc['density'],
                c=prdc['coverage'],
            )
        )
        metrics.update(prdc)

    return metrics


DATASET_STATS_FOR_FID = {
    'imagenet': 'assets/inception_stats/imagenet_256_train.npz',
    'ffhq': 'assets/inception_stats/ffhq_256_train.npz',
    'cc3m': 'assets/inception_stats/cc3m_256_val.npz',
}

DATASET_ACTS_FOR_PRDC = {
    'imagenet': 'assets/inception_features/adm/imagenet_256_10000.npz',
    'ffhq': 'assets/inception_features/shuffled/ffhq_256_train_50000.npz',
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result-path', type=str, required=True)
    parser.add_argument('-d', '--ref-dataset', type=str, default='imagenet', required=False)
    parser.add_argument('-l', '--log-postfix', type=str, default='', required=False)
    parser.add_argument('-m', '--metrics', nargs='+', default=['fid', 'prdc'])
    args = parser.parse_args()

    log_path = Path(args.result_path)
    if args.log_postfix:
        log_filename = f'fid_prdc_{args.log_postfix}.log'
    else:
        log_filename = 'fid_prdc.log'
    log_path = log_path / log_filename

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )

    logging.info('=' * 80)
    logging.info(f'{args}')

    ref_stat_path = DATASET_STATS_FOR_FID[args.ref_dataset] if 'fid' in args.metrics else None
    ref_feature_path = DATASET_ACTS_FOR_PRDC[args.ref_dataset] if 'prdc' in args.metrics else None

    results = compute_fid_prdc(args.result_path,
                               ref_stat_path=ref_stat_path,
                               ref_feature_path=ref_feature_path)

    logging.info('=' * 80)

    print("path, top-k, top-p, fid, precision, recall, density, coverage")
    if 'fid' in args.metrics:
        metric_list = ['fid']
    else:
        metric_list = []
    if 'prdc' in args.metrics:
        metric_list += ['precision', 'recall', 'density', 'coverage']

    path = args.result_path.split('/')[-2]
    # path example: 'epoch100_model_temp_1.0_top_k_256_top_p_0.9'
    try:
        epoch = path.split('_')[0]
        top_k = path.split('top_k')[-1].split('_')[1]
        top_p = path.split('top_p')[-1].split('_')[1]
    except Exception as e:
        print(f"[Except] Use alternative top-k and top-p value:\n{e}")
        epoch = 100
        top_k = 2048
        top_p = 1.0

    metrics = set(results.keys()) & set(metric_list)

    result_string = f"results: {epoch}, {top_k}, {top_p}, "
    for metric_name in metric_list:
        if metric_name in results.keys():
            result_string = result_string + f"{results[metric_name]}, "
        else:
            result_string = result_string + ", "

    logging.info("path, top-k, top-p, fid, precision, recall, density, coverage")
    logging.info(result_string)  # add 'results:' for easy parsing such as str.split('results: ')[-1]
