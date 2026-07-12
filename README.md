# leworld

## Installation

```bash
conda create -n lewm python=3.10
conda activate lewm
pip install -r requirements.txt
```

### robosuite (MimicGen-compatible)

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
git checkout b9d8d3de5e3dfd1724f4a0e6555246c460407daa
pip install -e .
pip install mujoco==2.3.2
cd ..
```

### robomimic

```bash
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
git checkout d0b37cf214bd24fb590d182edb6384333f67b661
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install -e .
cd ..
```

### mimicgen

```bash
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen
pip install -e .
cd ..
```

### roboverse

Vendored under `./roboverse` (includes a small gym 0.25 compatibility fix).

```bash
conda install -c conda-forge pybullet
pip install "gym==0.25.2"
pip install -e ./roboverse
```

Smoke test:

```bash
python scripts/scripted_collect.py -n 1 -t 30 -e Widow250DoubleDrawerOpenNeutral-v0 -pl drawer_open_transfer -a drawer_opened_success --noise=0.1 --gui
```

Run that from `./roboverse`.
