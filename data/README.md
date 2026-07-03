# Data

The competition data is not redistributed in this repo. Download it into this
directory with the Kaggle CLI (from the repo root):

```
kaggle competitions download -c store-sales-time-series-forecasting -p data --unzip
```

`notebooks/02_preprocessing.ipynb` writes the model-ready feature tables to
`data/processed/`.
