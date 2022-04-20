import argparse
import os

from utils.utils import load_tracker, trajectory_overlaps, count_failures, average_time, average_init_time
from utils.dataset import load_dataset
from utils.io_utils import read_regions, read_vector
from utils.export_utils import export_measures


def tracking_analysis(workspace_path, tracker_id):
    dataset = load_dataset(workspace_path)

    tracker_class = load_tracker(workspace_path, tracker_id)
    tracker = tracker_class()

    print('Performing evaluation for tracker:', tracker.name())

    per_seq_overlaps = len(dataset.sequences) * [0]
    per_seq_failures = len(dataset.sequences) * [0]
    per_seq_time = len(dataset.sequences) * [0]
    per_seq_init_time = len(dataset.sequences) * [0]

    for i, sequence in enumerate(dataset.sequences):

        results_path = os.path.join(workspace_path, 'results', tracker.name(), sequence.name,
                                    '%s_%03d.txt' % (sequence.name, 1))
        if not os.path.exists(results_path):
            print('Results does not exist (%s).' % results_path)

        time_path = os.path.join(workspace_path, 'results', tracker.name(), sequence.name,
                                 '%s_%03d_time.txt' % (sequence.name, 1))
        if not os.path.exists(time_path):
            print('Time file does not exist (%s).' % time_path)
            
        init_time_path = os.path.join(workspace_path, 'results', tracker.name(), sequence.name,
                                 '%s_%03d_init_time.txt' % (sequence.name, 1))
        if not os.path.exists(time_path):
            print('Time file does not exist (%s).' % init_time_path)

        regions = read_regions(results_path)
        times = read_vector(time_path)

        overlaps, overlap_valid = trajectory_overlaps(regions, sequence.groundtruth)
        failures = count_failures(regions)
        t = average_time(times, regions)
        init_t = average_init_time(times, regions)

        per_seq_overlaps[i] = sum(overlaps) / sum(overlap_valid)
        per_seq_failures[i] = failures
        per_seq_time[i] = t
        per_seq_init_time[i] = init_t

    return export_measures(workspace_path, dataset, tracker, per_seq_overlaps, per_seq_failures, per_seq_time, per_seq_init_time)


def main():
    parser = argparse.ArgumentParser(description='Tracking Visualization Utility')

    parser.add_argument('--workspace_path', help='Path to the VOT workspace', required=True, action='store')
    parser.add_argument('--tracker', help='Tracker identifier', required=True, action='store')

    args = parser.parse_args()

    tracking_analysis(args.workspace_path, args.tracker)


if __name__ == "__main__":
    main()
