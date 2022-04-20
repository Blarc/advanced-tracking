import os

from examples.correlation_filter_tracker import CorrelationFilterTracker
from utils.dataset import load_dataset
from utils.export_utils import export_measures
from utils.io_utils import read_regions, read_vector
from utils.tracker import Tracker
from utils.utils import trajectory_overlaps, count_failures, average_time


def evaluate_tracker(
        enlarge_factor=1,
        gaussian_sigma=4,
        filter_lambda=1,
        update_factor=0.2,
        workspace_path='',
        parameter_name='default'
):
    tracker: Tracker = CorrelationFilterTracker(
        enlarge_factor=enlarge_factor,
        gaussian_sigma=gaussian_sigma,
        filter_lambda=filter_lambda,
        update_factor=update_factor
    )

    dataset = load_dataset(workspace_path)

    results_path = os.path.join(workspace_path, 'results', tracker.name(), parameter_name)
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    analysis_path = os.path.join(workspace_path, 'analysis', tracker.name(), parameter_name)
    if not os.path.exists(analysis_path):
        os.mkdir(analysis_path)

    tracker.evaluate(dataset, results_path)

    per_seq_overlaps = len(dataset.sequences) * [0]
    per_seq_failures = len(dataset.sequences) * [0]
    per_seq_time = len(dataset.sequences) * [0]

    for i, sequence in enumerate(dataset.sequences):

        results_seq_path = os.path.join(workspace_path, 'results', tracker.name(), parameter_name, sequence.name,
                                        '%s_%03d.txt' % (sequence.name, 1))
        if not os.path.exists(results_seq_path):
            print('Results does not exist (%s).' % results_path)

        time_seq_path = os.path.join(workspace_path, 'results', tracker.name(), parameter_name, sequence.name,
                                     '%s_%03d_time.txt' % (sequence.name, 1))
        if not os.path.exists(time_seq_path):
            print('Time file does not exist (%s).' % time_seq_path)

        regions = read_regions(results_seq_path)
        times = read_vector(time_seq_path)

        overlaps, overlap_valid = trajectory_overlaps(regions, sequence.groundtruth)
        failures = count_failures(regions)
        t = average_time(times, regions)

        per_seq_overlaps[i] = sum(overlaps) / sum(overlap_valid)
        per_seq_failures[i] = failures
        per_seq_time[i] = t

    return export_measures(workspace_path, dataset, tracker, per_seq_overlaps, per_seq_failures, per_seq_time)


def compare_update_factors():
    update_factors = [.01, .05, .1, .2, .5, .8]
    workspaces = ['../workspace-dir-vot13', '../workspace-dir-vot14']
    with open('results/update_factor_comparison.txt', 'w', encoding='UTF-8') as f:
        for update_factor in update_factors:
            print(f'\\multirow{{2}}{{*}}{{{update_factor}}}')
            print(f'\\multirow{{2}}{{*}}{{{update_factor}}}', file=f)
            for i, workspace in enumerate(workspaces):
                output = evaluate_tracker(1, 4, 1, update_factor, workspace, f'update_factor_{update_factor}')
                workspace_name = workspace.split('-')[2]
                avg_overlap = output['average_overlap']
                failures = output['total_failures']
                fps = output['average_speed']
                if i == len(workspaces) - 1:
                    print(f'& {workspace_name} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\hline')
                    print(f'& {workspace_name} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\hline',
                          file=f)
                else:
                    print(
                        f'& {workspace_name} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\cline{{2-5}}')
                    print(
                        f'& {workspace_name} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\cline{{2-5}}',
                        file=f)


def compare_sigmas():
    sigmas = [.5, 1, 2, 3, 4, 5]
    workspaces = ['../workspace-dir-vot13', '../workspace-dir-vot14']
    with open('results/sigma_comparison.txt', 'w', encoding='UTF-8') as f:
        for sigma in sigmas:
            print(f'\\multirow{{2}}{{*}}{{{sigma}}}')
            print(f'\\multirow{{2}}{{*}}{{{sigma}}}', file=f)
            for i, workspace in enumerate(workspaces):
                output = evaluate_tracker(1, sigma, 1, 0.1, workspace, f'sigma_{sigma}')
                workspace_name = workspace.split('-')[2]
                avg_overlap = output['average_overlap']
                failures = output['total_failures']
                fps = output['average_speed']
                if i == len(workspaces) - 1:
                    print(f'& {workspace_name} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\hline')
                    print(f'& {workspace_name} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\hline',
                          file=f)
                else:
                    print(
                        f'& {workspace_name} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\cline{{2-5}}')
                    print(
                        f'& {workspace_name} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\cline{{2-5}}',
                        file=f)


def compare_enlarge_factors():
    enlarge_factors = [1, 1.25, 1.5, 2, 3]
    workspaces = ['../workspace-dir-vot13', '../workspace-dir-vot14']
    with open('results/enlarge_factor_comparison.txt', 'w', encoding='UTF-8') as f:
        for enlarge_factor in enlarge_factors:
            print(f'\\multirow{{2}}{{*}}{{{enlarge_factor}}}')
            print(f'\\multirow{{2}}{{*}}{{{enlarge_factor}}}', file=f)
            for i, workspace in enumerate(workspaces):
                output = evaluate_tracker(1, enlarge_factor, 1, 0.1, workspace, f'enlarge_factor_{enlarge_factor}')
                workspace_name = workspace.split('-')[2]
                avg_overlap = output['average_overlap']
                failures = output['total_failures']
                fps = output['average_speed']
                if i == len(workspaces) - 1:
                    print(f'& {workspace_name} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\hline')
                    print(f'& {workspace_name} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\hline',
                          file=f)
                else:
                    print(
                        f'& {workspace_name} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\cline{{2-5}}')
                    print(
                        f'& {workspace_name} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\cline{{2-5}}',
                        file=f)


if __name__ == '__main__':
    compare_update_factors()
    compare_sigmas()
    compare_enlarge_factors()
