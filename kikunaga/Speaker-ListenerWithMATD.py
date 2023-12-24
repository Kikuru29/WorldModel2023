import os

import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from tqdm import trange

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation

import datetime


import imageio
import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from PIL import Image, ImageDraw

from agilerl.algorithms.matd3 import MATD3

device = torch.device("mps")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the network configuration
def define_network_config():
    return {
        "arch": "mlp",  # Network architecture
        "h_size": [32, 32],  # Actor hidden size
    }

# Define the initial hyperparameters
def initialize_hyperparameters():
    return {
        "POPULATION_SIZE": 4,
        "ALGO": "MATD3",  # Algorithm
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "BATCH_SIZE": 32,  # Batch size
        "LR": 0.01,  # Learning rate
        "GAMMA": 0.95,  # Discount factor
        "MEMORY_SIZE": 100000,  # Max memory buffer size
        "LEARN_STEP": 5,  # Learning frequency
        "TAU": 0.01,  # For soft update of target parameters
        "POLICY_FREQ": 2,  # Policy frequnecy
        # Instantiate a tournament selection object (used for HPO)
        'TOURNAMENT_SIZE': 2,
        'ELITISM': True,
        # Instantiate a mutations object (used for HPO)
        'NO_MUTATION': 0.2,
        'ARCHITECTURE_MUTATION': 0.2,
        'NEW_LAYER_MUTATION': 0.2,
        'PARAMETER_MUTATION': 0.2,
        'ACTIVATION_MUTATION': 0,
        'RL_HP_MUTATION': 0.2,
        'RL_HP_SELECTION': ["lr", "learn_step", "batch_size"], # RL hyperparams selected for mutation
        'MUTATION_SD': 0.1,

    }

# Define the simple speaker listener environment as a parallel environment
def initialize_environment():
    env = simple_speaker_listener_v4.parallel_env(continuous_actions=True)
    env.reset()
    return env

# Configure the multi-agent algo input arguments
def set_action_and_state_dimensions(env, init_hp):
    """
    環境から行動次元と状態次元を設定し、初期ハイパーパラメータを更新する。
    env: 学習環境
    init_hp: 初期ハイパーパラメータの辞書
    """
    try:
        # まず、状態次元を設定する
        # 状態空間が離散的か連続的かに基づいて状態次元を取得する
        state_dim = [env.observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        # 連続的な状態空間の場合
        state_dim = [env.observation_space(agent).shape for agent in env.agents]
        one_hot = False

    try:
        # 次に、行動次元を設定する
        # 行動空間が離散的か連続的かに基づいて行動次元を取得する
        action_dim = [env.action_space(agent).n for agent in env.agents]
        init_hp["DISCRETE_ACTIONS"] = True
        init_hp["MAX_ACTION"] = None
        init_hp["MIN_ACTION"] = None
    except Exception:
        # 連続的な行動空間の場合
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        init_hp["DISCRETE_ACTIONS"] = False
        init_hp["MAX_ACTION"] = [env.action_space(agent).high for agent in env.agents]
        init_hp["MIN_ACTION"] = [env.action_space(agent).low for agent in env.agents]

    # 状態次元の調整（CHANNELS_LAST オプションが True の場合）
    if init_hp["CHANNELS_LAST"]:
        state_dim = [
            (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim
        ]

    return state_dim, action_dim, init_hp, one_hot


def list_initial_population(algo, state_dim, action_dim, one_hot, net_config, init_hp, population_size, device):
    """
    初期人口を生成する。
    algo: 使用する強化学習アルゴリズム
    state_dim: 状態次元
    action_dim: 行動次元
    one_hot: 状態がワンホットエンコードされているかどうか
    net_config: ネットワーク構成
    init_hp: 初期ハイパーパラメータ
    device: 使用するデバイス（例: "cuda"、"mps"、"cpu"）
    """
    pop_list = []
    for _ in range(init_hp["POPULATION_SIZE"]):
        agent = initialPopulation(algo, state_dim, action_dim, one_hot, net_config, init_hp, population_size, device)
        pop_list.append(agent)

    return pop_list

def create_initial_population(algo, state_dim, action_dim, one_hot, net_config, init_hp, population_size, device):
    pop = initialPopulation(
        init_hp["ALGO"],
        state_dim,
        action_dim,
        one_hot,
        net_config,
        init_hp,
        population_size=init_hp["POPULATION_SIZE"],
        device=device,
    )
    if pop is None:
        return []
    else:
        return pop


def configure_replay_buffer(init_hp, field_names, device):
    """
    リプレイバッファを設定する。
    init_hp: 初期ハイパーパラメータ
    agent_ids: エージェントのIDリスト
    device: 使用するデバイス（例: "cuda"、"mps"、"cpu"）
    """
    # リプレイバッファを格納するためのデータ構造を定義
    field_names = ["state", "action", "reward", "next_state", "done"]

    # リプレイバッファのインスタンスを作成
    memory = MultiAgentReplayBuffer(
        init_hp["MEMORY_SIZE"],  # バッファの最大サイズ
        field_names=field_names,  # 格納するフィールド名
        agent_ids=init_hp["AGENT_IDS"],      # エージェントのID
        device=device,             # 使用するデバイス
    )

    return memory


def tournament_selection(init_hp):
    """
    トーナメント選択の設定を行う。
    init_hp: 初期ハイパーパラメータ
    """
    tournament = TournamentSelection(
        tournament_size=init_hp['TOURNAMENT_SIZE'],
        elitism=init_hp['ELITISM'],
        population_size=init_hp['POPULATION_SIZE'],
        evo_step =1,
    )
    return tournament


def mutations_config(init_hp, net_config):
    """
    突然変異の設定を行う。
    init_hp: 初期ハイパーパラメータ
    net_config: ネットワーク構成
    """
    mutations = Mutations(
        algo=init_hp["ALGO"],
        no_mutation=init_hp['NO_MUTATION'],
        architecture=init_hp['ARCHITECTURE_MUTATION'],
        new_layer_prob=init_hp['NEW_LAYER_MUTATION'],
        parameters=init_hp['PARAMETER_MUTATION'],
        activation=init_hp['ACTIVATION_MUTATION'],
        rl_hp=init_hp['RL_HP_MUTATION'],
        rl_hp_selection=init_hp['RL_HP_SELECTION'],
        mutation_sd=init_hp['MUTATION_SD'],
        agent_ids=init_hp["AGENT_IDS"],
        arch=net_config["arch"],
        rand_seed=1,
        device=device
    )
    return mutations


def save_trained_model(model, path, filename):
    """
    model: 学習済みのモデル
    path: モデルを保存するディレクトリのパス
    filename: 保存するファイルの名前
    """
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, filename))


def training_loop(env, pop, memory, tournament, mutations, init_hp, net_config, max_episodes, max_steps):
    epsilon = 1.0
    eps_end = 0.1
    eps_decay = 0.995
    evo_epochs = 20
    evo_loop = 1
    elite = None

    if pop is not None and len(pop) > 0:
        elite = pop[0]

    for idx_epi in range(max_episodes):
        for agent in pop:
            state, info = env.reset()
            agent_reward = {agent_id: 0 for agent_id in env.agents}
            if init_hp["CHANNELS_LAST"]:
                state = {
                    agent_id: np.moveaxis(np.expand_dims(s, 0), [-1], [-3])
                    for agent_id, s in state.items()
                }

            for _ in range(max_steps):
                agent_mask = info.get("agent_mask")
                env_defined_actions = info.get("env_defined_actions")

                cont_actions, discrete_action = agent.getAction(
                    state, epsilon, agent_mask, env_defined_actions
                )
                action = discrete_action if agent.discrete_actions else cont_actions

                next_state, reward, termination, truncation, info = env.step(action)

                if init_hp["CHANNELS_LAST"]:
                    state = {agent_id: np.squeeze(s) for agent_id, s in state.items()}
                    next_state = {
                        agent_id: np.moveaxis(ns, [-1], [-3])
                        for agent_id, ns in next_state.items()
                    }

                memory.save2memory(state, cont_actions, reward, next_state, termination)

                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                if (memory.counter % agent.learn_step == 0) and (len(memory) >= agent.batch_size):
                    experiences = memory.sample(agent.batch_size)
                    agent.learn(experiences)

                if init_hp["CHANNELS_LAST"]:
                    next_state = {
                        agent_id: np.expand_dims(ns, 0)
                        for agent_id, ns in next_state.items()
                    }
                state = next_state

                if any(truncation.values()) or any(termination.values()):
                    break

            score = sum(agent_reward.values())
            agent.scores.append(score)

        epsilon = max(eps_end, epsilon * eps_decay)

        if (idx_epi + 1) % evo_epochs == 0:
            fitnesses = [
                agent.test(
                    env,
                    swap_channels=init_hp["CHANNELS_LAST"],
                    max_steps=max_steps,
                    loop=evo_loop,
                )
                for agent in pop
            ]

            print(f"Episode {idx_epi + 1}/{max_episodes}")
            print(f'Fitnesses: {["%.2f" % fitness for fitness in fitnesses]}')
            print(
                f'100 fitness avgs: {["%.2f" % np.mean(agent.fitness[-100:]) for agent in pop]}'
            )

            if len(pop) > 0:
                elite, pop = tournament.select(pop)
                pop = mutations.mutation(pop)

    if elite is not None:
        save_trained_model(elite, './models/MATD3', "MATD3_trained_agent.pt")

# Main code
if __name__ == "__main__":

    #現在日時を取得

    dt_now = datetime.datetime.now()
    str_dt_now = dt_now.strftime("%Y%m%d-%H%M")
    print(str_dt_now)
    #device = torch.device("mps")
    #
    print("===== AgileRL Online Multi-Agent Demo =====")

    net_config = define_network_config()
    init_hp = initialize_hyperparameters()
    env = initialize_environment()

    # Set the number of agents in the INIT_HP dictionary
    init_hp["N_AGENTS"] = env.num_agents  # Assuming env.agents gives the list of agents
    init_hp["AGENT_IDS"] = env.agents  # エージェントIDのリストを設定


    state_dim, action_dim, init_hp, one_hot= set_action_and_state_dimensions(env, init_hp)
    pop = create_initial_population(init_hp["ALGO"], state_dim, action_dim, one_hot, net_config, init_hp, init_hp["POPULATION_SIZE"], device)
    memory = configure_replay_buffer(init_hp, env.agents, device=device)
    tournament = tournament_selection(init_hp)
    mutations = mutations_config(init_hp, net_config)
    
    training_loop(env, pop, memory, tournament, mutations, init_hp, net_config, max_episodes=6000, max_steps=25)
    
    save_trained_model(pop[0], "./models/MATD3", "MATD3_trained_agent.pt")


# Define function to return image
def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(frame) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text(
        (im.size[0] / 20, im.size[1] / 18), f"Episode: {episode_num+1}", fill=text_color
    )

    return im


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the environment
    env = simple_speaker_listener_v4.parallel_env(
        continuous_actions=True, render_mode="rgb_array"
    )
    env.reset()
    try:
        state_dim = [env.observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dim = [env.observation_space(agent).shape for agent in env.agents]
        one_hot = False
    try:
        action_dim = [env.action_space(agent).n for agent in env.agents]
        discrete_actions = True
        max_action = None
        min_action = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        discrete_actions = False
        max_action = [env.action_space(agent).high for agent in env.agents]
        min_action = [env.action_space(agent).low for agent in env.agents]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    n_agents = env.num_agents
    agent_ids = env.agents

    # Instantiate an MADDPG object
    matd3 = MATD3(
        state_dim,
        action_dim,
        one_hot,
        n_agents,
        agent_ids,
        max_action,
        min_action,
        discrete_actions,
        device=device,
    )

    # Load the saved algorithm into the MADDPG object
    path = "./models/MATD3/MATD3_trained_agent.pt"
    matd3.loadCheckpoint(path)

    # Define test loop parameters
    episodes = 10  # Number of episodes to test agent on
    max_steps = 25  # Max number of steps to take in the environment in each episode

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards

    # Test loop for inference
    for ep in range(episodes):
        state, info = env.reset()
        agent_reward = {agent_id: 0 for agent_id in agent_ids}
        score = 0
        for _ in range(max_steps):
            agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
            env_defined_actions = (
                info["env_defined_actions"]
                if "env_defined_actions" in info.keys()
                else None
            )

            # Get next action from agent
            cont_actions, discrete_action = matd3.getAction(
                state,
                epsilon=0,
                agent_mask=agent_mask,
                env_defined_actions=env_defined_actions,
            )
            if matd3.discrete_actions:
                action = discrete_action
            else:
                action = cont_actions

            # Save the frame for this step and append to frames list
            frame = env.render()
            frames.append(_label_with_episode_number(frame, episode_num=ep))

            # Take action in environment
            state, reward, termination, truncation, info = env.step(action)

            # Save agent's reward for this step in this episode
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # Determine total score for the episode and then append to rewards list
            score = sum(agent_reward.values())

            # Stop episode if any agents have terminated
            if any(truncation.values()) or any(termination.values()):
                break

        rewards.append(score)

        # Record agent specific episodic reward
        for agent_id in agent_ids:
            indi_agent_rewards[agent_id].append(agent_reward[agent_id])

        print("-" * 15, f"Episode: {ep}", "-" * 15)
        print("Episodic Reward: ", rewards[-1])
        for agent_id, reward_list in indi_agent_rewards.items():
            print(f"{agent_id} reward: {reward_list[-1]}")
    env.close()

    # Save the gif to specified path
    gif_path = "./videos/"
    os.makedirs(gif_path, exist_ok=True)
    imageio.mimwrite(
        os.path.join("./videos/", "speaker_listener.gif"), frames, duration=10
    )