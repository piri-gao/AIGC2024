from agent.MAPPO_AIGC.MAPPO_AIGC_main import Runner_MAPPO_AIGC
from agent.QMIX_VDN_AIGC.QMIX_AIGC_main import Runner_QMIX_AIGC
import argparse
if __name__ == '__main__':
    algorithm = "QMIX"
    if algorithm=="QMIX":
        parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX and VDN in AIGC environment")
        parser.add_argument("--algorithm", type=str, default="QMIX", help="QMIX or VDN")
        parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
        parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
        parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
        parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

        
        parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
        parser.add_argument("--epsilon_decay_steps", type=float, default=50000, help="How many steps before the epsilon decays to the minimum")
        parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
        parser.add_argument("--buffer_size", type=int, default=500, help="The capacity of the replay buffer")
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
        parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
        parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
        parser.add_argument("--qmix_hidden_dim", type=int, default=32, help="The dimension of the hidden layer of the QMIX network")
        parser.add_argument("--hyper_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of the hyper-network")
        parser.add_argument("--hyper_layers_num", type=int, default=1, help="The number of layers of hyper-network")
        parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
        parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
        parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
        parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
        parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
        parser.add_argument("--use_lr_decay", type=bool, default=False, help="use lr decay")
        parser.add_argument("--use_RMS", type=bool, default=False, help="Whether to use RMS,if False, we will use Adam")
        parser.add_argument("--add_last_action", type=bool, default=True, help="Whether to add last actions into the observation")
        parser.add_argument("--add_agent_id", type=bool, default=True, help="Whether to add agent id into the observation")
        parser.add_argument("--use_double_q", type=bool, default=True, help="Whether to use double q-learning")
        parser.add_argument("--use_reward_norm", type=bool, default=False, help="Whether to use reward normalization")
        parser.add_argument("--use_hard_update", type=bool, default=True, help="Whether to use hard update")
        parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network")
        parser.add_argument("--tau", type=int, default=0.005, help="If use soft update")

        args = parser.parse_args()
        args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps

        env_names = ['Battle10v10']
        env_index = 0
        runner = Runner_QMIX_AIGC(args, env_name=env_names[env_index], number=1, seed=0)
        runner.run()
    elif algorithm=="MAPPO":
        parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in AIGC environment")
        parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
        parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
        parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
        parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

        parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
        parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
        parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
        parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
        parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
        parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
        parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
        parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
        parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
        parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
        parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
        parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
        parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
        parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
        parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
        parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
        parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
        parser.add_argument("--use_relu", type=float, default=True, help="Whether to use relu, if False, we will use tanh")
        parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
        parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
        parser.add_argument("--use_agent_specific", type=float, default=True, help="Whether to use agent specific global state.")
        parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

        args = parser.parse_args()
        env_names = ['Battle10v10']
        env_index = 0
        runner = Runner_MAPPO_AIGC(args, env_name=env_names[env_index], number=1, seed=0)
        runner.run()