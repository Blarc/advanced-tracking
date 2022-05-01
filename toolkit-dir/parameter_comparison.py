import os

from examples.particle_filter_tracker import ParticleFilterTracker
from utils.dataset import load_dataset
from utils.export_utils import export_measures
from utils.io_utils import read_regions, read_vector
from utils.tracker import Tracker
from utils.utils import trajectory_overlaps, count_failures, average_time


def evaluate_tracker(
        kernel_sigma=0.5,
        histogram_bins=6,
        n_of_particles=150,
        enlarge_factor=2,
        distance_sigma=0.11,
        update_alpha=0.05,
        color='HSV',
        dynamic_model='NCV',
        q=None,
        workspace_path='../workspace-dir-vot13',
        parameter_name='default'
):
    # q, particles, color space
    tracker: Tracker = ParticleFilterTracker(
        kernel_sigma=kernel_sigma,
        histogram_bins=histogram_bins,
        n_of_particles=n_of_particles,
        enlarge_factor=enlarge_factor,
        distance_sigma=distance_sigma,
        update_alpha=update_alpha,
        color=color,
        dynamic_model=dynamic_model,
        q=q
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

        init_time_path = os.path.join(workspace_path, 'results', tracker.name(), sequence.name,
                                      '%s_%03d_init_time.txt' % (sequence.name, 1))
        if not os.path.exists(init_time_path):
            print('Time file does not exist (%s).' % init_time_path)

        regions = read_regions(results_seq_path)
        times = read_vector(time_seq_path)

        overlaps, overlap_valid = trajectory_overlaps(regions, sequence.groundtruth)
        failures = count_failures(regions)
        t = average_time(times, regions)

        per_seq_overlaps[i] = sum(overlaps) / sum(overlap_valid)
        per_seq_failures[i] = failures
        per_seq_time[i] = t

    return export_measures(workspace_path, dataset, tracker, per_seq_overlaps, per_seq_failures, per_seq_time,
                           per_seq_time)


def compare_qs():
    qs = [1, 3, 5, 10, 50, 100]
    models = ['RW', 'NCV']
    with open('results/qs_comparison.txt', 'w', encoding='UTF-8') as f:
        for q in qs:
            print(f'\\multirow{{2}}{{*}}{{{q}}}')
            print(f'\\multirow{{2}}{{*}}{{{q}}}', file=f)
            for i, model in enumerate(models):
                output = evaluate_tracker(
                    q=q,
                    parameter_name=f'q_{q}_{model}',
                    dynamic_model=model
                )
                avg_overlap = output['average_overlap']
                failures = output['total_failures']
                fps = output['average_speed']
                if i == len(models) - 1:
                    print(f'& {model} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\hline')
                    print(f'& {model} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\hline',
                          file=f)
                else:
                    print(
                        f'& {model} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\cline{{2-5}}')
                    print(
                        f'& {model} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\cline{{2-5}}',
                        file=f)


def compare_particles():
    particles = [10, 50, 100, 200]
    models = ['RW', 'NCV']
    with open('results/particles_comparison.txt', 'w', encoding='UTF-8') as f:
        for particle in particles:
            print(f'\\multirow{{2}}{{*}}{{{particle}}}')
            print(f'\\multirow{{2}}{{*}}{{{particle}}}', file=f)
            for i, model in enumerate(models):
                output = evaluate_tracker(
                    n_of_particles=particle,
                    parameter_name=f'particle_{particle}_{model}',
                    dynamic_model=model
                )
                avg_overlap = output['average_overlap']
                failures = output['total_failures']
                fps = output['average_speed']
                if i == len(models) - 1:
                    print(f'& {model} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\hline')
                    print(f'& {model} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\hline',
                          file=f)
                else:
                    print(
                        f'& {model} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\cline{{2-5}}')
                    print(
                        f'& {model} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\cline{{2-5}}',
                        file=f)


def compare_color_spaces():
    color_spaces = ['HSV', 'LAB', 'RGB', 'YCRCB']
    models = ['RW', 'NCV']
    with open('results/color_spaces_comparison.txt', 'w', encoding='UTF-8') as f:
        for color_space in color_spaces:
            print(f'\\multirow{{2}}{{*}}{{{color_space}}}')
            print(f'\\multirow{{2}}{{*}}{{{color_space}}}', file=f)
            for i, model in enumerate(models):
                output = evaluate_tracker(
                    color=color_space,
                    parameter_name=f'color_space_{color_space}_{model}',
                    dynamic_model=model
                )
                avg_overlap = output['average_overlap']
                failures = output['total_failures']
                fps = output['average_speed']
                if i == len(models) - 1:
                    print(f'& {model} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\hline')
                    print(f'& {model} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\hline',
                          file=f)
                else:
                    print(
                        f'& {model} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\cline{{2-5}}')
                    print(
                        f'& {model} & {round(avg_overlap, 2)} & {failures} & {round(fps)} \\\\\n\\cline{{2-5}}',
                        file=f)


if __name__ == '__main__':
    # compare_qs()
    compare_particles()
    compare_color_spaces()
