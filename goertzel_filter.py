# Author: Peter Prince, Open Acoustic Devices

import numpy as np
import wave
import sys
import math
import matplotlib.pyplot as plt
import os

# Constants
INT16_MAX = 32767
TWO_PI = 2 * math.pi
GOERTZEL_THRESHOLD_BUFFER_LENGTH = 16384

def generate_hamming_values(N):
    hamming_values = np.zeros(N)
    for i in range(N):
        hamming_values[i] = 0.54 - 0.46 * math.cos(TWO_PI * i / (N - 1))
    hamming_mean = np.mean(hamming_values)
    return hamming_values, hamming_mean

def apply_goertzel_filter(samples, sample_rate, freq, N):
    hamming_values, hamming_mean = generate_hamming_values(N)
    c = 2.0 * math.cos(2.0 * math.pi * freq / sample_rate)
    maximum = N * (INT16_MAX + 1) * hamming_mean / 2.0
    scaler = pow(maximum, -1)
    output = []
    d1 = 0.0
    d2 = 0.0
    index = 0
    for i, sample in enumerate(samples):
        y = hamming_values[i % N] * sample + c * d1 - d2
        d2 = d1
        d1 = y
        if i % N == N - 1:
            magnitude = (d1 * d1) + (d2 * d2) - c * d1 * d2
            goertzel_value = 0 if magnitude < 0 else math.sqrt(magnitude)
            output.append(min(goertzel_value * scaler, 1.0))
            d1 = 0.0
            d2 = 0.0
            index += 1
    return output

def apply_goertzel_threshold(goertzel_values, threshold, window_length, min_trigger_duration_samples):
    min_trigger_duration_buffers = math.ceil(min_trigger_duration_samples / GOERTZEL_THRESHOLD_BUFFER_LENGTH)
    trigger_duration = 0
    above_threshold = False
    n = 0
    index = 0
    thresholded_value_count = 0
    goertzel_buffer_length = GOERTZEL_THRESHOLD_BUFFER_LENGTH // window_length
    output = []
    while index < len(goertzel_values):
        limit = min(len(goertzel_values), index + goertzel_buffer_length)
        while index < limit:
            if goertzel_values[index] > threshold:
                above_threshold = True
                trigger_duration = min_trigger_duration_buffers
            index += 1
        output.append(above_threshold)
        n += 1
        if above_threshold:
            thresholded_value_count += 1
            if trigger_duration > 1:
                trigger_duration -= 1
            else:
                above_threshold = False
    thresholded_value_count *= GOERTZEL_THRESHOLD_BUFFER_LENGTH
    thresholded_value_count = min(thresholded_value_count, len(goertzel_values) * window_length)
    return output, thresholded_value_count

def read_wav_samples(path):
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)
        dtype = np.int16 if sample_width == 2 else np.uint8
        samples = np.frombuffer(frames, dtype=dtype)
        if n_channels > 1:
            samples = samples[::n_channels]  # Take first channel
        return samples, sample_rate

def main():
    if len(sys.argv) not in [4, 6, 7]:
        print('Usage: python goertzel_filter.py <wav_path> <frequency> <filter_length> [<threshold> <min_trigger_duration_secs>] [<plot_enabled>]')
        sys.exit(1)
    wav_path = sys.argv[1]
    print(f'Filtering file: {wav_path}')
    freq = float(sys.argv[2])
    N = int(sys.argv[3])
    allowed_lengths = [16, 32, 64, 128, 256, 512, 1024]
    if N not in allowed_lengths:
        print(f'Error: filter_length must be one of {allowed_lengths}, but got {N}')
        sys.exit(1)
    samples, sample_rate = read_wav_samples(wav_path)
    responses = apply_goertzel_filter(samples, sample_rate, freq, N)
    if len(sys.argv) >= 6:
        threshold = float(sys.argv[4])
        min_trigger_duration_sec = int(sys.argv[5])
        allowed_seconds = [0, 1, 2, 5, 10, 15, 30, 60]
        if min_trigger_duration_sec not in allowed_seconds:
            print(f'Error: min_trigger_duration_seconds must be one of {allowed_seconds}, but got {min_trigger_duration_sec}')
            sys.exit(1)
        # Threshold must be a percentage between 0 and 100
        if not (0 <= threshold <= 100):
            print('Error: threshold must be a percentage between 0 and 100.')
            sys.exit(1)
        threshold = threshold / 100.0  # Convert to 0-1 scale
        min_trigger_duration_samples = min_trigger_duration_sec * sample_rate
        thresholded, count = apply_goertzel_threshold(responses, threshold, N, min_trigger_duration_samples)
        # Find contiguous events above threshold and print their timestamps
        goertzel_buffer_length = GOERTZEL_THRESHOLD_BUFFER_LENGTH // N
        events = []
        in_event = False
        event_start = None
        for i, val in enumerate(thresholded):
            if val and not in_event:
                in_event = True
                event_start = i
            elif not val and in_event:
                in_event = False
                event_end = i - 1
                # Calculate start and end time in seconds using buffer mapping
                start_time = event_start * goertzel_buffer_length * N / sample_rate
                end_time = ((event_end + 1) * goertzel_buffer_length * N) / sample_rate
                events.append((start_time, end_time))
        # Handle case where event goes till the end
        if in_event:
            event_end = len(thresholded) - 1
            start_time = event_start * goertzel_buffer_length * N / sample_rate
            end_time = ((event_end + 1) * goertzel_buffer_length * N) / sample_rate
            events.append((start_time, end_time))
        for idx, (start, end) in enumerate(events, 1):
            # Calculate the average Goertzel response within the event
            # Map event start/end time to response indices
            resp_start = int(round(start * sample_rate / N))
            resp_end = int(round(end * sample_rate / N))
            delta = end - start
            avg_response = np.mean(responses[resp_start:resp_end]) if resp_end > resp_start else 0.0
            print(f'{idx}:\tStart: {start:.3f}, End: {end:.3f}, Delta: {delta:.3f}, Avg Goertzel: {avg_response:.4f}')
        # Write event info to .events.txt file
        filtered_txt_name = wav_path.rsplit('.', 1)[0] + '.events.txt'
        if os.path.exists(filtered_txt_name):
            os.remove(filtered_txt_name)
        with open(filtered_txt_name, 'w') as f:
            f.write('Begin Time (s)\tEnd Time (s)\tDelta Time (s)\tAverage Goertzel\n')
            for start, end in events:
                resp_start = int(round(start * sample_rate / N))
                resp_end = int(round(end * sample_rate / N))
                avg_response = np.mean(responses[resp_start:resp_end]) if resp_end > resp_start else 0.0
                delta = end - start
                f.write(f'{start:.6f}\t{end:.6f}\t{delta:.6f}\t{avg_response:.6f}\n')
        # Plot Goertzel response with time axis and threshold line
        if len(sys.argv) == 7:
            plot_arg = sys.argv[6].lower()
            plot_enabled = plot_arg == 'true'
        else:
            plot_enabled = True  # Default to True if not provided
        if plot_enabled:
            times = np.arange(len(responses)) * N / sample_rate
            waveform_times = np.arange(len(samples)) / sample_rate
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            # Goertzel response
            ax1.plot(times, responses, label='Goertzel Response')
            ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold*100:.0f}%)')
            ax1.set_ylabel('Goertzel Response')
            ax1.set_title('Goertzel Response Over Time')
            ax1.legend()
            ax1.set_ylim(0, 1)
            ax1.set_xlim(0, max(times))
            # Waveform
            # Normalize waveform to [-1, 1] for percentage axis
            if samples.dtype == np.int16:
                norm_samples = samples / 32767.0
            elif samples.dtype == np.uint8:
                norm_samples = (samples - 128) / 128.0
            else:
                norm_samples = samples / np.max(np.abs(samples))
            ax2.plot(waveform_times, norm_samples, linewidth=0.5)
            ax2.set_xlabel('Time (seconds)')
            # Set y-ticks and labels for percentage axis (increments of 25%)
            yticks = np.arange(-1, 1.01, 0.25)
            yticklabels = [f'{int(abs(y)*100)}%' if y != 0 else '0%' for y in yticks]
            ax2.set_yticks(yticks)
            ax2.set_yticklabels(yticklabels)
            ax2.set_ylabel('Amplitude (%)')
            ax2.set_title('Waveform')
            # Highlight non-event regions with translucent grey
            last_end = 0.0
            for start, end in events:
                if start > last_end:
                    ax1.axvspan(last_end, start, color='grey', alpha=0.3, zorder=0)
                    ax2.axvspan(last_end, start, color='grey', alpha=0.3, zorder=0)
                last_end = end
            # Cover region after last event if any
            if events and last_end < max(times):
                ax1.axvspan(last_end, max(times), color='grey', alpha=0.3, zorder=0)
                ax2.axvspan(last_end, max(times), color='grey', alpha=0.3, zorder=0)
            elif not events:
                # If no events, cover the whole plot
                ax1.axvspan(0, max(times), color='grey', alpha=0.3, zorder=0)
                ax2.axvspan(0, max(times), color='grey', alpha=0.3, zorder=0)
            plt.tight_layout()
            # Save plot as PNG with the wav file's name
            png_name = wav_path.rsplit('.', 1)[0] + '.png'
            plt.savefig(png_name)
            plt.close()

if __name__ == '__main__':
    main()
