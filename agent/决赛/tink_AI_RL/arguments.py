import time
import torch
import argparse

def parse_args():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument('--task', type=str, default="3v3.scen")
    parser.add_argument('--device', default=device)
    parser.add_argument("--max_train_steps", type=int, default=int(5000), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=2, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--episode_limit", type=float, default=800, help="The max length of one eposide game")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")#8192 
    parser.add_argument("--mini_batch_size", type=int, default=1, help="Minibatch size")#
    parser.add_argument("--hidden_width", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")#
    parser.add_argument('--reward_threshold', type=float, default=None)
    parser.add_argument('--best_reward', type=float, default=None)
    parser.add_argument('--save_path', type=str, default='./CPPO_model_data')
    parser.add_argument("--num_agent", type=int, default=3, help="Number of agent")
    parser.add_argument("--share_state_dim", type=int, default=None, help="size of shareobs")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--yita", type=float, default=3, help="dual PPO clip parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument('--double_clip_inner_eps', type=float, default=0.1,help='help="Co_PPO  clip parameter')
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--use_BCRL", type=bool, default=True, help="use BC and RL train model")
    parser.add_argument("--use_Co_PPO", type=bool, default=True, help="use Co_PPO")
    parser.add_argument("--use_dual_clip", type=bool, default=True, help="use dual_clip_PPO")
    parser.add_argument("--use_centralized_V", action='store_false',
                        default=False, help="Whether to use centralized V function")
    parser.add_argument("--use_gru", type=bool, default=True, help="Whether to use GRU")
    parser.add_argument("--video_train", type=bool, default=False, help="Whether to use video to train")
   
    return parser.parse_args()