# KAN See In the Dark [[paper]](https://arxiv.org/abs/2409.03404)
## Get Started
### Dependencies and Installation
- Python 3.8
- Pytorch 1.11
1. Create Conda Environment

```
conda create --name KSID python=3.8
conda activate KSID
```

2. Install PyTorch

```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

3. Clone Repo

```
git clone https://github.com/AXNing/KSID.git
```

4. Install Dependencies

```
cd KSID
pip install -r requirements.txt
```

### Data Preparation
#### Download the raw training and evaluation datasets
[[Google Drive]](https://drive.google.com/drive/folders/1yAp7c-fQhU_KQkK7xk1KZ4YKAywwo-2z?usp=drive_link)

```
├── dataset
    ├── LOLv1
        ├── our485
            ├──low
            ├──high
	├── eval15
            ├──low
            ├──high
├── dataset
   ├── LOLv2
       ├── Real_captured
           ├── Train
           ├── Test
```


### Testing

1. Our pre-trained weights can be downloaded by clicking [here](https://drive.google.com/drive/folders/1AcfKzxens1mhs7IALtiPVyE60ZV-X_5n?usp=drive_link), and put it in the following folder:

```
├── checkpoints
    ├── lolv1_gen.pth
    ├── lolv2_real_gen.pth
    ├── lsrw_gen.pth
```
2. Modifying the pre-trained weight path in the json file.

```
    "path": {
        "log": "logs",
        "tb_logger": "tb_logger",
        "results": "results",
        "checkpoint": "checkpoint", 
        "resume_state": ""
    },
```
   
```
# LOLv1
python test.py --dataset ./config/lolv1.yml --config ./config/lolv1_test.json

# LOLv2-real
python test.py --dataset ./config/lolv2_real.yml --config ./config/lolv2_real_test.json
```
3. Evaluation metrics are referenced [here](https://github.com/chaofengc/IQA-PyTorch).
### To Do List
- [x] Release the testing code for KSID.
- [x] Upload the pretrained checkpoints.
- [ ] Release the training code for KSID.
