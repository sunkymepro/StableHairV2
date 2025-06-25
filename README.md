# StableHair v2
**Stable-Hair v2: Real-World Hair Transfer via Multiple-View Diffusion Model**  
Kuiyuan Sun*, [Yuxuan Zhang*](https://xiaojiu-z.github.io/YuxuanZhang.github.io/), [Jichao Zhang*](https://zhangqianhui.github.io/), [Jiaming Liu](https://scholar.google.com/citations?user=SmL7oMQAAAAJ&hl=en), 
 [Wei Wang](https://weiwangtrento.github.io/), [Nicu Sebe](http://disi.unitn.it/~sebe/), Yao Zhao<br>
*Equal Contribution <br>
Beijing Jiaotong University, Shanghai Jiaotong University, Ocean University of China, Tiamat AI, University of Trento <br>
[Arxiv](https://ttgnerf.github.io/TT-GNeRF/), [Project](https://ttgnerf.github.io/TT-GNeRF/)<br>


Bald     |  Reference | Multiple View | Original Video
![](./imgs/multiview1.gif)  
Bald     |  Reference | Multiple View | Original Video
![](./imgs/multiview2.gif)

## Environments

```
conda create -n stablehairv2 python=3.6
```
```
pip install -r req.txt
```

## Results

<img src="./imgs/teaser.jpg" width="800"> 

## Pretrained Model
| Name        |   Model   | 
|-------------|:---------:|
| motion_module-454500.pth      | [:link:](https://drive.google.com/file/d/1B7xJ4uM6AbMb24XUeW_g-yzBFkkfMMxq/view?usp=drive_link) | 
| pytorch_model_1.bin      |  [:link:](https://drive.google.com/file/d/1FwKPZI8lvdlZqu8R1aJ-QbE55kxHPHjU/view?usp=drive_link) | 
| pytorch_model_2.bin |  [:link:](https://drive.google.com/file/d/1h3dXlo8lhZN3ee5aN0shZmpLfn5itVou/view?usp=drive_link) | 
| pytorch_model_3.bin   |  [:link:](https://drive.google.com/file/d/1jARfXaU6wiur85Vm1JxZ_xye0FfrUiqb/view?usp=drive_link) | 
| pytorch_model.bin      |  [:link:](https://drive.google.com/file/d/1zXXf13pV5IOn2vrV6DGI9hliEFvuPrYf/view?usp=drive_link) |

### Single View Hair Transfer

```
python test_kmeans.py --outdir=[output_path] \
            --network=[pretrained eg3d model] \
            --dataset_path [our dataset path] \
            --csvpath [label path] \
            --batch=1 \
            --gen_pose_cond=True \
            --resolution 512 \
            --label_dim 6 \
            --truncation_psi 0.7 \
            --file_id 66 \
            --lambda_normal 1.0
```

### Multiple-View Hair Transfer

```
python test_reference_geometry_editing.py --outdir=[output_path] --batch=1 \
            --gen_pose_cond=True --num_steps 100 \
            --faceid_weights [face_id_path] \
            --w_dir [our dataset path] \
            --resolution 512 --truncation_psi 0.7 --id 41 --ref_id 5235
```


# Our V1 version

StableHair v2 is an improved version of [StableHair](https://github.com/Xiaojiu-z/Stable-Hair) (AAAI 2025)