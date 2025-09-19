# Update model version here before running
model_v = "v33"

# OpenSoundscape imports
from opensoundscape import Audio, Spectrogram
from opensoundscape.annotations import BoxedAnnotations
from opensoundscape.data_selection import resample
from opensoundscape import CNN
from opensoundscape.ml.datasets import AudioFileDataset
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.preprocess.actions import Action, SpectrogramToTensor
from opensoundscape.preprocess.action_functions import random_wrap_audio, scale_tensor, frequency_mask
from opensoundscape.preprocess.utils import show_tensor, show_tensor_grid
import opensoundscape.ml
from opensoundscape.preprocess import preprocessors
from opensoundscape.ml import cnn, cnn_architectures
from opensoundscape.ml.cnn import use_resample_loss

# transfer learning tools
# from opensoundscape.ml.shallow_classifier import MLPClassifier, quick_fit, fit_classifier_on_embeddings
# import bioacoustics_model_zoo as bmz
import torch

# General-purpose packages
import os
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
import copy
import wandb
import random
from matplotlib import pyplot as plt
from dotenv import load_dotenv

# Set random seed
#NOTE: Will want to remove this for actual model training, but useful for initial coding and debugging
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Get selection table file paths
# Load local file paths from .env file and create variables
load_dotenv()

all_sample_dirs = os.getenv('all_sample_dirs')

# Define function to get .txt file names from comma-separated list of directories
def list_files(dirs, ext):
    
    # Split directories into separate file paths where there are multiple
    dirs_list = dirs.split(",")

    # Get list of relevant file paths from all specified folders
    file_paths = [file for x in dirs_list for file in glob.glob(f"{x}/*{ext}")]

    # Return list
    return file_paths

# Get list of selection table files
annotations = list_files(all_sample_dirs, ext='.txt')

# Get list of audio files
audio = list_files(all_sample_dirs, ext='.WAV') + list_files(all_sample_dirs, ext='.wav')

# Load annotated audio
labelled_audio = BoxedAnnotations.from_raven_files(
    sorted(annotations), annotation_column="Annotation", audio_files = sorted(audio)
)

# Make a separate copy of labelled audio to filter out non-dormouse annotations before tidying up annotation column
labelled_audio_hdor = copy.deepcopy(labelled_audio)
labelled_audio_hdor.df = labelled_audio_hdor.df[labelled_audio_hdor.df['annotation'].str.startswith('hdor')].copy()

# Split cam/ncam annotation into separate column
labelled_audio_hdor.df['cam'] = [x.rsplit('_', 1)[1] for x in labelled_audio_hdor.df['annotation']]

# Split call type into separate column
labelled_audio_hdor.df['call_type'] = [x.rsplit('_', 1)[0] for x in labelled_audio_hdor.df['annotation']]

# Replace original annotations with call type
labelled_audio_hdor.df['annotation'] = labelled_audio_hdor.df['call_type']

# Parameters to use for label creation
clip_duration = 1
clip_overlap = 0  # Consider changing this to squeeze more clips out of dataset
min_label_overlap = 0.01 #0.25  # Arch calls are ~0.01-0.29s long (avg 0.027s) - 0.01s minimum unlikely to discard clips containing only arch calls
min_label_fraction = 0.5 #0.5  # With 0.5s each annotated call should appear in at least one clip
# calls_of_interest = ["hdor", "hdor_arch", "hdor_asc", "hdor_wig"]

# Create dataframe of one-hot labels
clip_labels = labelled_audio_hdor.clip_labels(
    clip_duration = clip_duration,
    clip_overlap = clip_overlap,
    min_label_overlap = min_label_overlap,
    min_label_fraction = min_label_fraction,
    # class_subset = calls_of_interest,  #NOTE: Optional - including for now as example code, but currently includes all as data is already filtered to hdor
    final_clip = None  # Where length of clip doesn't divide by 1s, discards the final <1s clip
)

# Roll all call types into one column and drop wiggled and general calls, as these aren't common/precise enough in my data for training on specifically
clip_labels['hdor_all'] = (clip_labels.hdor | clip_labels.hdor_arch | clip_labels.hdor_asc | clip_labels.hdor_wig)
clip_labels.drop(['hdor', 'hdor_wig'], axis=1, inplace = True)

# Split data into train and test

# Make indices into columns for easier manipulation
clip_labels_reset_ind = clip_labels.reset_index()

# Clips from cage 3 go into validation dataset
val_df = clip_labels_reset_ind[clip_labels_reset_ind.file.str.contains("03_NR23")]

# Remaining clips go into training dataset
train_df = clip_labels_reset_ind[~clip_labels_reset_ind['file'].isin(val_df['file'])]

# Make first 3 columns indices again
val_df.set_index(['file', 'start_time', 'end_time'], inplace=True)
train_df.set_index(['file', 'start_time', 'end_time'], inplace=True)

# upsample positive samples and downsample 'empty' clips to make them less overrepresented
# The downsampling will also make model training faster
balanced_train_df = resample(train_df, n_samples_per_class=1000, 
                             n_samples_without_labels=5000, 
                             random_state=0)

# Initialise model with resnet18 architecture
classes = clip_labels.columns
model = cnn.CNN("resnet18", classes, sample_duration=1.0)

# Use modified loss function, recommended for multi-class model
use_resample_loss(model, train_df=balanced_train_df)

# Freeze feature extractor so I'm only training the classification head, to prevent over-fitting
# model.freeze_feature_extractor()

# Customise preprocessor pipeline

# Get dataframe containing only non-dormouse Calke Abbey clips, with file as the only index
background_df = clip_labels.reset_index()
background_df = background_df[background_df['file'].str.contains("CalkeAbbey")]
background_df = background_df[background_df['hdor_all'] == False]
background_df.set_index('file', inplace=True)
background_df.drop(columns=['start_time', 'end_time'], inplace=True)

# Initialise preprocessor with overlay df containing background noise from Calke Abbey recordings
my_preprocessor = SpectrogramPreprocessor(sample_duration=1, overlay_df=background_df)

# Remove step to add noise
# Random noise can have a detrimental effect on model performance, esp where SNR is low (MacIsaac et al., 2024)
my_preprocessor.remove_action("add_noise")

# Remove random affine transformations
# These include rotation and mirroring (Mumuni & Mumuni, 2022) which are likely to distort vocalisations beyond their naturally occurring characteristics, producing unrealistic training data
my_preprocessor.remove_action("random_affine")

# Remove random trim audio step as I want all clips to have the same length
my_preprocessor.remove_action("random_trim_audio")

# Remove rescaling step
# No documentation that I can find on this function
my_preprocessor.remove_action("rescale")

# Resample all audio to the same sample rate as the Calke Abbey AudioMoths (192kHz)
my_preprocessor.pipeline.load_audio.set(sample_rate=192000)

# Set min and max frequency to include ultrasonic range (up to 60kHz)
# Choosing not to exclude sub-ultrasonic frequencies. Want to include bird songs due to harmonics potentially causing confusion.
my_preprocessor.pipeline.bandpass.set(min_f = 0, max_f = 60000)

# Set window length to 1024 in line with spectrograms in Middleton, Newson & Pearce (2023)
my_preprocessor.pipeline.to_spec.set(window_samples = 1024)

# Set max length and number of time masks
# Avg length of arch calls is ~0.026s so I've set max_width to 0.01s to avoid cutting out too much of a single call.
# max_width is actually the fraction of the sample, not the number of seconds, but my sample length is 1s which makes the maths simple
my_preprocessor.pipeline.time_mask.set(max_width = 0.01, max_masks = 5)

# Set max width and number of frequency masks
# Minimum change in frequency is <3kHz for annotated dormouse calls, so max_width needs to be small to avoid removing entire calls
my_preprocessor.pipeline.frequency_mask.set(max_width = 0.02, max_masks = 5)

# Function to check if overlay should be applied - to use this, uncomment and add criterion_fn=from_sound_library to overlay settings
#NOTE: Decided not to use criterion_fn and instead apply background noise to all clips
# def from_sound_library(AudioSample):
#     return 'sound_library' in AudioSample.source
               # , 'annotated_bat_data' in AudioSample.source)

# Set overlay to blend clips (70%) with background noise from Calke Abbey (30%)
my_preprocessor.pipeline.overlay.set(update_labels=False, overlay_prob=0.75, overlay_weight=0.3, overlay_class=None)

# Add step to randomly wrap audio, which acts as a time-shift augmentation
my_preprocessor.insert_action(
    action_index="random_wrap_audio",
    action=Action(random_wrap_audio, probability=0.5, max_shift=None),
    after_key="load_audio",
)

# Add step to normalize tensor
my_preprocessor.insert_action(
    action_index="scale_tensor",
    action=Action(scale_tensor),
    after_key="frequency_mask",
)

# Replace default preprocessor with my preprocessor
model.preprocessor = my_preprocessor

# Create folder to save data and model in
model_save_dir = Path("model_training_checkpoints/model_" + model_v)

os.mkdir(model_save_dir)

# Save train and val datasets to csv
train_df.to_csv("model_training_checkpoints/model_" + model_v + "/train_df_" + model_v + ".csv")
balanced_train_df.to_csv("model_training_checkpoints/model_" + model_v + "/balanced_train_df_" + model_v + ".csv")
val_df.to_csv("model_training_checkpoints/model_" + model_v + "/val_df_" + model_v + ".csv")

# Set up Weights and Biases model logging
try:
    wandb.login()
    wandb_session = wandb.init(
        entity="isobel-russ-ucl",  # replace with your entity/group name
        project="hazel-dormouse-classifier",
        name="Train CNN " + model_v,
    )
except:  # if wandb.init fails, don't use wandb logging
    print("failed to create wandb session. wandb session will be None")
    wandb_session = None

    # Train model
model.train(
    balanced_train_df,
    val_df,
    epochs=50,
    batch_size=16,
    log_interval=100,  # log progress every 100 batches
    num_workers=12,  # parallelized cpu tasks for preprocessing
    wandb_session=wandb_session,
    save_interval=1,  # save checkpoint every 10 epochs
    save_path=model_save_dir,  # location to save checkpoints
)