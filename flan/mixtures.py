# Copyright 2021 The FLAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add Mixtures to the registry."""

import functools
import seqio
from collections import defaultdict
import few_shot
import task_splits
import tasks  # pylint: disable=unused-import
import templates  # pylint: disable=unused-import
import json

mixing_rate_3k = functools.partial(seqio.mixing_rate_num_examples, maximum=3000)

all_splits = []

shot_config = few_shot.ShotConfig.ZERO
'''
# Add inter cluster splits.
all_splits += task_splits.generate_inter_cluster_splits(
    shot_config=shot_config)

# Add intra cluster splits.
all_splits += task_splits.generate_intra_cluster_splits(
    shot_config=shot_config)

# Add all overlap split.
all_splits += [
    task_splits.generate_all_overlap_split(shot_config=shot_config)
]

# Add superglue ablation on number of templates.
all_splits += task_splits.generate_superglue_num_templates_ablation(
    shot_config=shot_config)

# Add superglue ablation on number of tasks.
all_splits += task_splits.generate_superglue_num_tasks_ablation(
    shot_config=shot_config)

# Add diversity ablation on number of clusters in finetuning.
all_splits += task_splits.generate_inter_ablation(shot_config=shot_config)

# Add test dataset
all_splits = task_splits.generate_test_cluster_splits()
'''

# Add my inter cluster splits.
all_splits += task_splits.generate_my_inter_ablation(
    shot_config=shot_config)

for split in all_splits:
  seqio.MixtureRegistry.add(
      name=split.train_mixture_name,
      tasks=split.train_tasks,
      default_rate=mixing_rate_3k)
  seqio.MixtureRegistry.add(
      name=split.eval_mixture_name,
      tasks=split.test_tasks,
      default_rate=seqio.mixing_rate_num_examples)

for split in all_splits:
  print(split.train_mixture_name)


# to get all the data from splits
for split in all_splits:
    INPUT_SEQ_LEN = 2056
    TARGET_SEQ_LEN = 512

    selected_mixture = seqio.get_mixture_or_task(split.train_mixture_name)
    dataset_train = selected_mixture.get_dataset(
        sequence_length={"inputs": INPUT_SEQ_LEN, "targets": TARGET_SEQ_LEN},
        num_epochs=1,
        shuffle=True,
        copy_pretokenized=True,
        # The passthrough features let you track the source/task/template metadata for the example
        passthrough_features=["_template_idx", "_task_source", "_task_name", "_template", "_template_type"]
    )

    selected_mixture = seqio.get_mixture_or_task(split.eval_mixture_name)
    dataset_eval = selected_mixture.get_dataset(
        sequence_length={"inputs": INPUT_SEQ_LEN, "targets": TARGET_SEQ_LEN},
        num_epochs=1,
        shuffle=True,
        copy_pretokenized=True,
        # The passthrough features let you track the source/task/template metadata for the example
        passthrough_features=["_template_idx", "_task_source", "_task_name", "_template", "_template_type"]
    )

    # To read out the data you can do something like this:
    save_data_train = []
    save_data_eval = []
    source_counter = defaultdict(lambda: 0)
    NUM_SAMPLES_TRAIN = 10000
    NUM_SAMPLES_EVAL = 1000
    # If you would like to take min(1 epoch, NUM_SAMPLES) then use dataset.take(NUM_SAMPLES)
    # Or if you would like to gather a full epoch, simply `enumerate(dataset)` until completion.
    for i, ex in enumerate(dataset_train.take(NUM_SAMPLES_TRAIN)):
        #source_counter[ex["_task_source"].numpy()] += 1
        save_data_train.append((ex["inputs_pretokenized"].numpy().decode(),
                        ex["targets_pretokenized"].numpy().decode()))
    for i, ex in enumerate(dataset_eval.take(NUM_SAMPLES_EVAL)):
        #source_counter[ex["_task_source"].numpy()] += 1
        save_data_eval.append((ex["inputs_pretokenized"].numpy().decode(),
                        ex["targets_pretokenized"].numpy().decode()))
    #print(f"Data Submixture Counts: {source_counter}")

    # print(save_data)
    with open('/content/datasets/'+split.train_mixture_name+'.json', 'w') as f:
      for d in save_data_train:
        json.dump(d, f)
        f.write('\n')
    with open('/content/datasets/'+split.train_mixture_name+'.json', 'w') as f:
      for d in save_data_eval:
        json.dump(d, f)
        f.write('\n')
