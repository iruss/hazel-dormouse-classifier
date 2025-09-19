# Author: Peter Prince, Open Acoustic Devices

import os
import sys
import subprocess

if len(sys.argv) < 5:
    print('Usage: python batch_goertzel_filter.py <folder> <frequency> <filter_length> [<threshold> <min_trigger_duration_secs>] [<plot_enabled>]')
    sys.exit(1)

folder = sys.argv[1]
args = sys.argv[2:]

# Find all .wav files in the folder (non-recursive)
wav_files = [f for f in os.listdir(folder) if f.lower().endswith('.wav')]

if not wav_files:
    print('No .wav files found in the specified folder.')
    sys.exit(0)

files_with_events = []
for wav_file in wav_files:
    wav_path = os.path.join(folder, wav_file)
    cmd = [sys.executable, 'goertzel_filter.py', wav_path] + args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'Error processing {wav_file}')
        continue
    # Check if .events.txt exists and has more than just the header
    events_txt = os.path.splitext(wav_path)[0] + '.events.txt'
    if os.path.exists(events_txt):
        with open(events_txt, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                files_with_events.append(wav_file)
# Write summary file
summary_path = os.path.join(folder, 'files_with_events.txt')
with open(summary_path, 'w') as f:
    for fname in files_with_events:
        f.write(fname + '\n')
