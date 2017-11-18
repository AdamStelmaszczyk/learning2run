import datetime
import json
import os
import multiprocessing as mp
import es.es_distributed.main as es_main
import logging

import click


def highlight(x):
    if not isinstance(x, str):
        x = json.dumps(x, sort_keys=True, indent=2)
    click.secho(x, fg='green')

def start_master(exp_str):
    es_main.master(exp_str, None, '/var/run/redis/redis.sock', './logs')

def start_worker(num_workers):
    es_main.workers('localhost', 6379, '/var/run/redis/redis.sock', num_workers)

@click.command()
@click.argument('exp_files', nargs=-1, type=click.Path(), required=True)
@click.option('--num_workers', type=int, default=1, help='Number of workers')
@click.option('--yes', is_flag=True, help='Skip confirmation prompt')
def main(exp_files,
         num_workers,
         yes):

    logging.basicConfig(level=logging.INFO)
    highlight('Launching:')
    highlight(locals())

    for i_exp_file, exp_file in enumerate(exp_files):
        with open(exp_file, 'r') as f:
            exp = json.loads(f.read())
        highlight('Experiment [{}/{}]:'.format(i_exp_file + 1, len(exp_files)))
        highlight(exp)
        if not yes:
            click.confirm('Continue?', abort=True)

        exp_prefix = exp['exp_prefix']
        exp_str = json.dumps(exp)

        exp_name = '{}_{}'.format(exp_prefix, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

        master_process = mp.Process(target=start_master, args=(exp_str,))
        worker_process = mp.Process(target=start_worker, args=(num_workers,))
        master_process.start()
        worker_process.start()
        worker_process.join()
        master_process.join()
        highlight("%s launched successfully." % exp_name)


if __name__ == '__main__':
    main()
