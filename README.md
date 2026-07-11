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
