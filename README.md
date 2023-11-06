# OVIR-3D
> [!IMPORTANT]
> We will be presenting our poster at [CoRL'23](https://www.corl2023.org) on Wednesday, Nov. 8. from 5:15 - 6:00 pm in [session 4](https://www.corl2023.org/papers#:~:text=Poster%204%3A%20LLM/VLM/HRI). Hope to see you in person!

> [!WARNING]
>For those who cloned this repo before Oct 25, 2023, please update the repo by running `git pull` and `git submodule update --init --recursive`. We fixed a major bug that caused very bad segmentation for ScanNet200. We reran the results for ScanNet200 and the prediction files could be found at [here](https://drive.google.com/file/d/1_4cATwib3UyNax5iRgI1mW524ignx9_5/view?usp=sharing). Sorry for the inconvenience.

**OVIR-3D: Open-Vocabulary 3D Instance Retrieval Without Training on 3D Data**.\
Shiyang Lu, Haonan Chang, Eric Jing, Yu Wu, Abdeslam Boularias, Kostas Bekris

To appear at [CoRL'23](https://www.corl2023.org/). Also presented as an extended abstract at [OpenSUN3D (ICCV-W)](https://opensun3d.github.io/).\
[[Full Paper (8-page)](https://openreview.net/pdf?id=gVBvtRqU1_)][[Extended Abstract (4-page)](https://github.com/shiyoung77/OVIR-3D/blob/main/ovir3d_iccvw_extended_abstract.pdf)][[ICCV-W Poster](https://github.com/shiyoung77/OVIR-3D/blob/main/ovir3d_iccvw_poster.pdf)]

## Intro
Recent progress on open-vocabulary (language-driven, without a predefined set of categories) 3D segmentation addresses the problem mainly at the semantic level (by mid-2023). Nevertheless, robotic applications, such as manipulation and navigation, often require 3D object geometries at the instance level. This work provides a straightforward yet effective solution for open-vocabulary 3D instance retrieval, which returns a ranked set of 3D instance segments given a 3D point cloud reconstructed from an RGB-D video and a language query.

<img src="figures/ovir3d-demo.png" alt="drawing" width="700"/>

## Key Takeaways
Directly training an open-vocabulary 3D segmentation model is hard due to the lack of annotated 3D data with enough category
varieties. Instead, this work views this problem as a 3D fusion problem from language-guided 2D region proposals, which could be trained with extensive 2D datasets, and provides a method to project and fused 2D instance information in the 3D space for fast retrieval.

## Pipeline Structure
<img src="figures/pipeline.png" alt="drawing" width="700"/>


## Usage
### Download the repo
```
git clone git@github.com:shiyoung77/OVIR-3D.git --recurse-submodules
```

### Install Dependencies
```
conda create -n ovir3d python=3.10
conda activate ovir3d

# install pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# install other dependencies
pip install -r requirements.txt

# Download Detic pretrained model
cd Detic
mkdir models
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -O models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

### Demo
A preprossed sample scene from YCB-Video dataset could be downloaded [here](https://drive.google.com/file/d/1HqEUVIb1fNpjnaJH1VegylscRz3qRLEI/view?usp=sharing) (~1.3G). Extract it in this repo and then run `./demo.sh`.

### Dataset Preparation
For **CUSTOM** dataset, make your RGB-D video data in the following format. We have opensourced our [video recording scripts](https://github.com/shiyoung77/realsense-video-capture) for realsense cameras and [KinectFusion implementation](https://github.com/shiyoung77/KinectFusion-python) in Python to help you record and reconstruct your custom 3D scene.
```
{dataset_path}/
    {video_name}/
        color/
            0000-color.jpg
            0001-color.jpg
            ...
        depth/
            0000-depth.png
            0001-depth.png
            ...
        poses/
            0000-pose.txt
            0001-pose.txt
            ...
        config.json  # camera information
        scan-{resolution}.pcd  # reconstructed point cloud, e.g. scan-0.005.pcd
```
`config.json` should contain the camera information. An example config.json is as follows.
```
{
    "id": "video0",
    "im_w": 640,
    "im_h": 480,
    "depth_scale": 1000,
    "cam_intr": [
        [ 1066.778, 0, 312.9869 ],
        [ 0, 1067.487, 241.3109 ],
        [ 0, 0, 1 ]
    ]
}
```
For **YCB-Video** dataset, we have already processed the validation set (0048~0059) and you can directly download from [here](https://drive.google.com/file/d/1ypTJSw0SRLbHpuOXimUwMSVO24XGuBqZ/view?usp=sharing) (~16G).

For **ScanNet200** dataset, please follow the instructions on their [website](https://kaldir.vc.in.tum.de/scannet_benchmark/documentation) to download the data and extract them in the following format. The color images were captured at a high resolution, which have to be resized to 480p to match with the depth image (they are already aligned so just `cv2.resize`).
```
{dataset_path}/
    {video_name}/
        color/
            0000-color.jpg
            0001-color.jpg
            ...
        depth/
            0000-depth.png
            0001-depth.png
            ...
        poses/
            0000-pose.txt
            0001-pose.txt
            ...
        config.json
        {video_name}.txt
        {video_name}_clean_2.ply
```
You need the following files for ScanNet200 preprocessing/evalution, they are included in this repo for your convenience.
```
scannet_preprocess.py  # copy files, resize images, and generate config.json
scannet200_constants.py
scannet200_splits.py
scannetv2-labels.combined.tsv
scannet200_instance_gt/
    validation/
        {video_name}.txt
        ...
```
You can visualize the ground truth annotation via `visualize_{scannet200/ycb_video}_gt.py`.


### Text-aligned 2D Region Proposal Generation
This works adopts Detic as a backbone 2D region proposal network. This repo contains a modified copy of the original repo as a submodule. To generate region proposals, `cd Detic`, change the dataset path in `file.py` and then run `python fire.py`. This script supports multi-gpu to inference multiple videos in parallel. By default, this scripts query all the categories in `imagenet21k` with confidence threshold at 0.3. The output masks and text-aligned features for each frame are stored in the `{dataset_path}/{video_name}/detic_output` folder. You can also save the 2D visualization using the `--save_vis` option, but this will make inference much slower.

```
cd Detic
python fire.py --dataset {dataset_path}
```

### 2D-to-3D Fusion
Once 2D region proposals are generated, you can fuse the results for the 3D scan using the proposed algorithm. The implementation of this algorithm is in `src/proposed_fusion.py`. Again, there is a script `src/fire.py` that supports parallel fusion for multiple 3D scenes if you have multiple gpus. The output is stored in `{dataset_path}/{video_name}/detic_output/{vocab}/predictions` folder. It is recommended to have at least 11GB memory (e.g. 2080Ti) to run this algorithm, otherwise you may run into memory issues for large scenes.

```
cd src
python fire.py --dataset {dataset_path}
```

### Inference
Once fusion is done, you will be able to interactively query 3D instances via `src/instance_query.py`. Here `out_filename` is the file outputed from last step, the default is `proposed_fusion_detic_iou-0.25_recall-0.50_feature-0.75_interval-300.pkl`.
```
python src/instance_query.py -d {dataset_path} -v {video_name} --prediction_file {out_filename}
```

## Additional Comments
You may wonder why we call it instance retrieval instead of instance segmentation. The reason is that we formulate this problem as an information retrieval problem, i.e. given a query, retrieve relevant documents (ranked instances) from a database (a 3D scene). The proposed method first tries to find all 3D instances in a scene (without knowing the testing categories), and then rank them based on the language query using CLIP feature similarity. This is also how we [evaluate](https://github.com/shiyoung77/OVIR-3D/blob/main/src/eval_instance_retrieval_scannet200.py) our method and baselines, i.e. [Standard mAP for information retrieval](https://stackoverflow.com/a/40834813). We believe that it is a more reasonable metric given our open-vocabulary problem setting, though it is slightly different from the mAP metric commonly used for closed-set instance segmentation, where each predicted instance has to be assiged with a category label and a confidence score. Nevertheless, if you use OVIR-3D as a baseline, feel free to use any metric you like on the [prediction files](https://drive.google.com/file/d/1_4cATwib3UyNax5iRgI1mW524ignx9_5/view?usp=sharing) that we provided for ScanNet200, which contains all 3D instance segments (likely more than what ScanNet200 annotated) and their corresponding CLIP features.

## Applications
We have a follow-up work [Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs (OVSG)](https://ovsg-l.github.io/), which will also appear at [CoRL'23](https://www.corl2023.org/). It uses OVIR-3D as backbone method to get all 3D instances in a scene, and then build a 3D scene graph for more precise object retrieval using natural language by considering object relationships. Please take a look if you are interested.


# Bibtex
For OVIR-3D:
```
@inproceedings{lu2023ovir,
  title={OVIR-3D: Open-Vocabulary 3D Instance Retrieval Without Training on 3D Data},
  author={Lu, Shiyang and Chang, Haonan and Jing, Eric Pu and Boularias, Abdeslam and Bekris, Kostas},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023}
}
```

For OVSG:
```
@inproceedings{chang2023context,
  title={Context-Aware Entity Grounding with Open-Vocabulary 3D Scene Graphs},
  author={Chang, Haonan and Boyalakuntla, Kowndinya and Lu, Shiyang and Cai, Siwei and Jing, Eric Pu and Keskar, Shreesh and Geng, Shijie and Abbas, Adeeb and Zhou, Lifeng and Bekris, Kostas and others},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023}
}
```
