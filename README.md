# GraphCORAlysis: Benchmarking GNNs on Cora

A simple educational exercise comparing three Graph Neural Network (GNN) architectures—GCN, GraphSAGE, and GAT—against a logistic-regression baseline on the Cora citation dataset. We measure test accuracy, training time, and memory usage, and include analyses of node embeddings via t-SNE, confusion matrices, cluster purity, and example misclassifications.

---

## Prerequisites

- Python 3.11  
- pip  

---

## Installation

1. Clone the repo:  
   ```bash
   git clone https://github.com/your-username/GraphCORAlysis.git
   cd GraphCORAlysis
```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Download data & prepare environment**
   The Cora dataset is fetched automatically when you run the scripts.

2. **Baseline (logistic regression)**

   ```bash
   python src/baseline.py
   ```

3. **Train GNN models**
   Run each model with the same settings:

   ```bash
   python src/train.py --model gcn
   python src/train.py --model sage
   python src/train.py --model gat
   ```

4. **View results in MLflow**

   ```bash
   mlflow ui --backend-store-uri mlruns
   ```

   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

5. **Explore the notebook**
   Open and run `notebooks/report.ipynb` to see code, tables, plots, and deeper analyses.

---

## Project Structure

````
GraphCORAlysis/
├── configs/  
│   └── config.yaml            # experiment parameters  
├── src/  
│   ├── data_loader.py         # Cora loading & preprocessing  
│   ├── models.py              # unified GCN/SAGE/GAT class  
│   ├── train.py               # training + MLflow logging  
│   └── baseline.py            # logistic regression baseline  
├── notebooks/  
│   └── report.ipynb          # executable report of results  
├── figures/                   # t-SNE plots and confusion matrices  
├── checkpoints/               # saved model weights  
├── mlruns/                    # MLflow tracking data  
├── requirements.txt           # Python dependencies  
└── README.md                  # this file  
````

---

## Key Findings

| Model     | Test Acc | Epoch Time (s) | Peak Mem (MB) |
| --------- | -------- | -------------- | ------------- |
| Baseline  | \~0.50   | –              | –             |
| GCN       | \~0.83   | Fast           | Small         |
| GraphSAGE | \~0.85   | \~3× GCN       | Medium        |
| GAT       | \~0.86   | \~2× GCN       | \~1.5× GCN    |

* All GNNs outperform the logistic‐regression baseline by a large margin.
* On this small dataset, GCN is both fastest and most accurate.
* GraphSAGE yields slightly tighter clusters at the cost of slower training.
* GAT balances accuracy and expressivity but uses more memory.

---

## License

This work is released under the MIT License.
```
