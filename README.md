# Final Project for MTH 9899

by Liang Shan, Junhan Wang, Quanzhi Bi

## The docker container

Please pull our docker image from docker hub (just `continuumio/anaconda3_w_pytorch` with `dill` and `xgboost` installed)

```bash
docker pull jordanwang1998/mth9899_2021
```

## The file structure of the project

```bash
> tree .
.
├── EDA.py
├── README.md
├── dataloader.py
├── drift-plot.ipynb
├── features.py
├── get_IC.py
├── main.py
├── models
│   └── xgboost.pickle
├── pipeline.py
├── project_2021.pdf
├── selection.py
├── slides
│   └── MTH+9899+project%2C+final.pdf
└── train.ipynb
```

## How to run the code

In mode 1, the following code will extract raw data from `./train` (downloaded from [here](https://www.dropbox.com/s/fe0ov6ip19b2z47/train.zip?dl=1)) and save the feature csv files into `./features`

```bash
python main.py -i ./train -o ./features -s 20170102 -e 20171229 -m 1
```

Then in mode 2, the following code will generate predictions from `start_date` to `end_date` and save to `./predictions`:

```bash
python main.py -i ./features -o ./predictions -p ./models/xgboost.pickle -s 20170102 -e 20171229 -m 2
```

Notice that in `./predictions`, `date.csv` is the prediction of next dat's return made on `date`.