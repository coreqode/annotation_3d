# annotation_3d

### Prerequisites
- This script has been tested on Mac and Linux
- Blender should be installed 

## Usage

```bash
git clone https://github.com/coreqode/annotation_3d.git
cd annotation_3d
```

- Place `starter.blend` file from this [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/chandradeep_pokhariya_research_iiit_ac_in/EcPIPlqYX49AnTsULcDiioQBxiVLOAC05yaCrWdgFBOTyw?e=StDTff) to `data/blend_files`. (~127mb)
- Place the frame sequence (video) in the `data/to_annotate` folder similar to abhi.
- Define the seq_name, gender and other parameters in the `run.sh`
    -   `sh run.sh abhi male resume` to resume already saved sequence(abhi) which uses (male) smplx
    -   `sh run.sh abhi male restart` to restart sequence(abhi) which uses (male) smplx

