# World-Act

**Note:** Human demos have already been collected and expanded with MimicGen. A sped-up preview of the generated data is in [`data_playback.mp4`](data_playback.mp4). The full HDF5 dataset is on Google Drive: [demo.hdf5](https://drive.google.com/file/d/19fS1VDBsYhsn97lgNLD-3DZoUsSWmKdB/view?usp=sharing).

## Dataset collection

Pipeline for the robosuite **Stack** task (Panda + OSC_POSE), ending in a MimicGen-augmented training set under `data/robot/mimicgen/`.

1. **Teleop collection** — `src/datasets/robot/collect_human_data.py`  
   Keyboard demos of Stack. Successful, approved episodes are saved as per-demo HDF5s in `data/robot/raw_demo/`.

2. **Process demos** — `src/datasets/robot/process_demos.py`  
   Merges the raw demos and runs robomimic’s `convert_robosuite` conversion → `data/robot/processed/demo.hdf5` (with train/valid masks).

3. **MimicGen generation** — `src/datasets/robot/generate_dataset.py`  
   Annotates the processed source for MimicGen (`MG_Stack`), writes a Stack config, and generates ~1000 successful demos with `agentview` + `robot0_eye_in_hand` cameras. Failed demos/playback are dropped and the output is flattened into `data/robot/mimicgen/` (including a 90/10 train/valid mask).

4. **Training loader** — `src/datasets/robot/primary_dataset.py`  
   PyTorch dataset over `data/robot/mimicgen/demo.hdf5` (windowed frames, frame skip, proprio, actions).

Example:

```bash
python src/datasets/robot/collect_human_data.py
python src/datasets/robot/process_demos.py
python src/datasets/robot/generate_dataset.py 1000
```

## Installation

```bash
conda create -n lewm python=3.10
conda activate lewm
pip install -r requirements.txt
```

`robosuite`, `robomimic`, and `mimicgen` are git submodules (pinned commits). After cloning this repo:

```bash
git submodule update --init --recursive
```

Then install them editable from the local checkouts (no need to re-clone):

```bash
# robosuite (MimicGen-compatible pin)
cd robosuite
pip install -e .
pip install mujoco==2.3.2
cd ..

# robomimic
cd robomimic
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -e .
cd ..

# mimicgen
cd mimicgen
pip install -e .
cd ..
```

Note for Markers: This codebase also includes some code I wrote when I was just trying to replicate the results of the pushT experiment in the original leworldmodel paper so there is some extra code. The important files for the actual experiment I am doing for the project are:
- the files in src/datasets/robot
- src/train/train_baseline.py and src/train/train_primary
- all the files in src/architectures
- src/losses.py

