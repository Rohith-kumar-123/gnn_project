
# HR-GNN: Financial Fraud Detection

## About the Project

HR-GNN (Historical- and Relational-aware Graph Neural Network) is a machine learning project designed to identify suspicious financial transactions, particularly money-laundering transactions.

The model represents financial accounts as nodes and transactions as edges in a graph. It combines historical account behaviour, transaction relationships, and learned edge-reliability scores to predict the probability that a transaction is fraudulent.

## Project Objectives

- Detect suspicious financial transactions using graph neural networks.
- Use historical account activity to improve transaction representations.
- Learn edge-reliability scores to control information flow between connected accounts.
- Evaluate fraud detection under severe class imbalance.
- Study the contribution of different model components through ablation experiments.

## How the Model Works

1. **Input Processing:** Transaction records are sorted chronologically, and numerical features are standardized using training data.
2. **Node Encoding:** Account features are transformed into learned representations using two fully connected layers.
3. **Persistence Modelling:** Historical account features are used to calculate an optional persistence score.
4. **Edge Reliability:** The model estimates a reliability score for each transaction edge.
5. **Message Passing:** Information is aggregated between connected accounts in both directions, using temporal weights and optional reliability channels.
6. **Fraud Prediction:** A transaction classifier combines the learned endpoint representations and transaction features to estimate fraud probability.

## Dataset

This project uses the synthetic AMLSim HI-Small anti-money-laundering dataset.

- Transactions: 5,078,345
- Accounts: 515,080
- Labelled laundering transactions: 5,177
- Chronological split: 60% training, 20% validation, 20% testing

Dataset: https://huggingface.co/datasets/bbfizp/AMLSim-HI-Small

Only approximately 0.102% of transactions are labelled as laundering, making class imbalance a major challenge.

## Model Configuration

- Node encoder: 2 fully connected layers
- Graph message-passing layers: 2
- Hidden dimension: 32
- Classifier hidden dimension: 64
- Dropout: 0.2
- Optimizer: Adam
- Learning rate: 0.003
- Weight decay: 0.00001
- Training loss: Focal loss
- Negative sampling: Up to 50 negative transactions per positive transaction
- Early stopping: Validation Average Precision
- Evaluation: 5 random seeds

## Technologies Used

- Python
- PyTorch
- Graph Neural Networks
- Machine Learning
- Pandas and NumPy
- Matplotlib

## Experimental Results

| Model | Average Precision (AP) |
|---|---:|
| GCN baseline | 0.4479 |
| GCN + History | 0.5818 |
| GCN + Reliability | 0.4885 |
| HR-GNN without time decay | 0.6325 |
| HR-GNN without persistence | 0.6347 |
| HR-GNN single channel | 0.6400 |
| HR-GNN full model | 0.6385 |

The full HR-GNN model achieved an AP of 0.6385, compared with 0.4479 for the baseline GCN.

At a 0.5% transaction-review budget, the full model achieved a fraud recall of 0.7686, compared with 0.6080 for the baseline GCN.

The single-channel variant achieved slightly higher mean AP than the full dual-channel model. The improvement over the history-aware GCN was not statistically significant after multiple-comparison correction.

## Model Variants

- **GCN:** Conventional graph convolutional baseline.
- **GCN + History:** Baseline augmented with historical account features.
- **GCN + Reliability:** Baseline with reliability-aware learning.
- **HR-GNN without Time Decay:** Tests the contribution of temporal decay.
- **HR-GNN without Persistence:** Tests the contribution of persistence weighting.
- **HR-GNN Single Channel:** Uses only the reliability-weighted message channel.
- **HR-GNN Full:** Combines the proposed components, including dual-channel message passing.

## Installation and Usage

Clone the repository:

```bash
git clone https://github.com/Rohith-kumar-123/gnn_project.git
cd gnn_project
```

Create and activate a virtual environment if desired, then install the dependencies listed in the project's requirements file.

Run the training or evaluation script provided in the repository.

**Note:** Check the actual script names, dependency file, and dataset paths in the repository before running the project. Exact commands depend on the files included in the current version.

## Limitations

- The dataset is synthetic and may not represent all real-world financial transaction patterns.
- Graph snapshots are constructed at phase boundaries rather than at every individual training transaction timestamp.
- Reliability supervision uses weak targets derived from training fraud activity.
- Results are based on five random seeds and one chronological split.
- Additional evaluation on real-world data is needed.

## Future Improvements

- Evaluate the model on real-world financial transaction datasets.
- Improve strict time-causal graph construction.
- Test more random seeds and chronological splits.
- Investigate better edge-reliability supervision.
- Explore methods to reduce false alerts while maintaining fraud recall.

## Repository

GitHub: https://github.com/Rohith-kumar-123/gnn_project

## Disclaimer

This project is intended for academic and experimental use. Model predictions indicate potentially suspicious transactions and should not be treated as proof of financial crime.
