import argparse
from .wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2009")
    parser.add_argument("--model_name", type=str, default="PKT")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    
    parser.add_argument("--emb_size", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--n_label', type=int, default=2)
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--embed_dropout', type=float, default=0.3)
    parser.add_argument('--cell_dropout', type=float, default=0.5)
    parser.add_argument('--out_dropout', type=float, default=0.5)
    parser.add_argument('--final_dropout', type=float, default=0.5)
    parser.add_argument('--rnn_type', type=str, default="GRU", choices=["LSTM", "GRU"])
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    params = vars(args)
    main(params)