"""
This tutorial shows how to train an MATD3 agent on the simple speaker listener multi-particle environment.

Authors: Michael (https://github.com/mikepratt1), Nickua (https://github.com/nicku-a)
"""

import os
import pprint
import datetime
import json
import argparse
import dill #import pickle

import numpy as np
import torch
from pettingzoo.mpe import simple_reference_v3
from tqdm import trange

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation

if __name__ == "__main__":

    #引数を取得
    parser = argparse.ArgumentParser()

    parser.add_argument('-dt', '--datetime', help='年月日時分をYYYYMMDD-hhmmの形式で指定。')
    parser.add_argument('-p', '--param', default='./parameters.json', help='学習用パラメータJSONファイルのパスを指定。')
    parser.add_argument('-pop', '--population', help='population.pklファイルのパスを指定', )
    parser.add_argument('-r', '--replaybuffer', help='replaybuffer.pklファイルのパスを指定', )
    parser.add_argument('-t', '--tournament', help='tournament.pklファイルのパスを指定', )
    parser.add_argument('-m', '--mutations', help='mutations.pklファイルのパスを指定', )

    args = parser.parse_args()

    if not args.datetime:
        #現在日時を取得
        dt_now = datetime.datetime.now()
        str_dt_now = dt_now.strftime("%Y%m%d-%H%M")
        print(str_dt_now)
    else:
        str_dt_now = args.datetime

    
    #JSONファイルからパラメータを読み込み
    with open(args.param, mode="rt", encoding="utf-8") as f:
        param_json = json.load(f)
        
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Online Multi-Agent Demo =====")

    # Define the network configuration
    # ネットワークコンフィグレーションの定義
    #NET_CONFIG = {
    #    "arch": "mlp",  # Network architecture
    #    "h_size": [32, 32],  # Actor hidden size
    #}
    NET_CONFIG = param_json["NET_CONFIG"] # JSONファイルからの読み込み値を反映

    # Define the initial hyperparameters
    # 初期ハイパーパラメータの定義
    #INIT_HP = {
    #    "POPULATION_SIZE": 4,
    #    "ALGO": "MATD3",  # Algorithm
    #    # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
    #    "CHANNELS_LAST": False,
    #    "BATCH_SIZE": 32,  # Batch size
    #    "LR": 0.01,  # Learning rate
    #    "GAMMA": 0.95,  # Discount factor
    #    "MEMORY_SIZE": 100000,  # Max memory buffer size
    #    "LEARN_STEP": 5,  # Learning frequency
    #    "TAU": 0.01,  # For soft update of target parameters
    #    "POLICY_FREQ": 2,  # Policy frequnecy
    #}
    INIT_HP = param_json["INIT_HP"] # JSONファイルからの読み込み値を反映

    # Define the simple speaker listener environment as a parallel environment
    # シンプル・スピーカー・リスナー環境（並列）の定義
    env = simple_reference_v3.parallel_env(continuous_actions=True)
    env.reset()

    # Configure the multi-agent algo input arguments
    # マルチ・エージェントアルゴへの入力引数の設定
    try:
        state_dim = [env.observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dim = [env.observation_space(agent).shape for agent in env.agents]
        one_hot = False
    try:
        action_dim = [env.action_space(agent).n for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = True
        INIT_HP["MAX_ACTION"] = None
        INIT_HP["MIN_ACTION"] = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = False
        INIT_HP["MAX_ACTION"] = [env.action_space(agent).high for agent in env.agents]
        INIT_HP["MIN_ACTION"] = [env.action_space(agent).low for agent in env.agents]

    
    if False: # デバッグ出力
        print("state_dim", state_dim)
        print("one_hot", one_hot)
        print( 'INIT_HP["DISCRETE_ACTIONS"]', INIT_HP["DISCRETE_ACTIONS"])
        print( 'INIT_HP["MAX_ACTION"]', INIT_HP["MAX_ACTION"])
        print( 'INIT_HP["MIN_ACTION"]', INIT_HP["MIN_ACTION"])
    

    # Not applicable to MPE environments, used when images are used for observations (Atari environments)
    # MPE環境には適用されない。観測のために画像が使用される場合に使う（アタリ環境）
    if INIT_HP["CHANNELS_LAST"]:
        state_dim = [
            (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim
        ]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    # エージェントとエージェントIDの数を、ハイパーパラメータディクショナリの初期化のために加える
    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

    
    if True: # デバッグ出力
        print('state_dim', state_dim)
        print('action_dim', action_dim)
        print('one_hot', one_hot)
        print('NET_CONFIG')
        pprint.pprint(NET_CONFIG)
        print('INIT_HP')
        pprint.pprint(INIT_HP)
        print('device', device)
    
    # Create a population ready for evolutionary hyper-parameter optimisation
    # 進化的なハイパーパラメータ最適化のための母集団を作成する
    if not args.population:
        population = initialPopulation(
            INIT_HP["ALGO"],
            state_dim,
            action_dim,
            one_hot,
            NET_CONFIG,
            INIT_HP,
            population_size=INIT_HP["POPULATION_SIZE"],
            device=device,
        )
    else: #引数に指定されている場合はファイルから読み込む
        population = dill.load(open(args.population,'rb'))
        

    # Configure the multi-agent replay buffer
    # マルチ・エージェント・リプレイバッファを設定する
    if not args.replaybuffer:
        field_names = ["state", "action", "reward", "next_state", "done"]
        memory = MultiAgentReplayBuffer(
            INIT_HP["MEMORY_SIZE"],
            field_names=field_names,
            agent_ids=INIT_HP["AGENT_IDS"],
            device=device,
        )
    else: #引数に指定されている場合はファイルから読み込む
        memory = dill.load(open(args.replaybuffer,'rb'))

    # Instantiate a tournament selection object (used for HPO)
    # トーナメント選択オブジェクトのインスタンス化（HPOで使用）
    if not args.tournament:
        tournament = TournamentSelection(
            tournament_size=2,  # Tournament selection size
            elitism=True,  # Elitism in tournament selection
            population_size=INIT_HP["POPULATION_SIZE"],  # Population size
            evo_step=1,
        )  # Evaluate using last N fitness scores
    else: #引数に指定されている場合はファイルから読み込む
        tournament = dill.load(open(args.tournament,'rb'))

    # Instantiate a mutations object (used for HPO)
    # ミューテーション・オブジェクトのインスタンス化（HPOで使用）
    if not args.mutations:
        mutations = Mutations(
            algo=INIT_HP["ALGO"],
            no_mutation=0.2,  # Probability of no mutation
            architecture=0.2,  # Probability of architecture mutation
            new_layer_prob=0.2,  # Probability of new layer mutation
            parameters=0.2,  # Probability of parameter mutation
            activation=0,  # Probability of activation function mutation
            rl_hp=0.2,  # Probability of RL hyperparameter mutation
            rl_hp_selection=[
                "lr",
                "learn_step",
                "batch_size",
            ],  # RL hyperparams selected for mutation
            mutation_sd=0.1,  # Mutation strength
            agent_ids=INIT_HP["AGENT_IDS"],
            arch=NET_CONFIG["arch"],
            rand_seed=2,#1,
            device=device,
        )
    else: #引数に指定されている場合はファイルから読み込む
        mutations = dill.load(open(args.mutations,'rb'))

    # Define training loop parameters
    # 学習ループ・パラメータを定義
    max_episodes = param_json["PARAM_TRAIN"]["max_episodes"]
    max_steps    = param_json["PARAM_TRAIN"]["max_steps"]
    epsilon      = param_json["PARAM_TRAIN"]["epsilon"]
    eps_end      = param_json["PARAM_TRAIN"]["eps_end"]
    eps_decay    = param_json["PARAM_TRAIN"]["eps_decay"]
    evo_epochs   = param_json["PARAM_TRAIN"]["evo_epochs"]
    evo_loop     = param_json["PARAM_TRAIN"]["evo_loop"]
    #max_episodes = 10 # 6000 #500  # Total episodes (default: 6000)
    #max_steps = 100 #25  # Maximum steps to take in each episode
    #epsilon = 1.0  # Starting epsilon value
    #eps_end = 0.1  # Final epsilon value
    #eps_decay = 0.995  # Epsilon decay
    #evo_epochs = 20  # Evolution frequency
    #evo_loop = 1  # Number of evaluation episodes
    elite = population[0]  # Assign a placeholder "elite" agent

    # Training loop
    # 学習ループ
    for idx_epi in trange(max_episodes):
        
        for agent in population:  # Loop through population
            
            state, info = env.reset()  # Reset environment at start of episode
            agent_reward = {agent_id: 0 for agent_id in env.agents}
            if INIT_HP["CHANNELS_LAST"]:
                state = {
                    agent_id: np.moveaxis(np.expand_dims(s, 0), [-1], [-3])
                    for agent_id, s in state.items()
                }

            for _ in range(max_steps):
                agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
                env_defined_actions = (
                    info["env_defined_actions"]
                    if "env_defined_actions" in info.keys()
                    else None
                )

                # Get next action from agent
                # エージェントから次の行動を取得する
                cont_actions, discrete_action = agent.getAction(
                    state, epsilon, agent_mask, env_defined_actions
                )
                if agent.discrete_actions:
                    action = discrete_action
                else:
                    action = cont_actions

                next_state, reward, termination, truncation, info = env.step(
                    action
                )  # Act in environment

                # Image processing if necessary for the environment
                # 環境に応じた画像処理を行う
                if INIT_HP["CHANNELS_LAST"]:
                    state = {agent_id: np.squeeze(s) for agent_id, s in state.items()}
                    next_state = {
                        agent_id: np.moveaxis(ns, [-1], [-3])
                        for agent_id, ns in next_state.items()
                    }

                # Save experiences to replay buffer
                # 経験をリプレイバッファに保存する
                memory.save2memory(state, cont_actions, reward, next_state, termination)

                # Collect the reward
                # 報酬を受け取る
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                # Learn according to learning frequency
                # 学習周期に合わせて学習する
                if (memory.counter % agent.learn_step == 0) and (
                    len(memory) >= agent.batch_size
                ):
                    experiences = memory.sample(
                        agent.batch_size
                    )  # Sample replay buffer
                    agent.learn(experiences)  # Learn according to agent's RL algorithm

                # Update the state
                # 状態を更新する
                if INIT_HP["CHANNELS_LAST"]:
                    next_state = {
                        agent_id: np.expand_dims(ns, 0)
                        for agent_id, ns in next_state.items()
                    }
                state = next_state

                # Stop episode if any agents have terminated
                # いずれかのエージェントが終了したならば、エピソードを停止する
                if any(truncation.values()) or any(termination.values()):
                    break

            # Save the total episode reward
            # エピソードの合計報酬を保存する
            score = sum(agent_reward.values())
            agent.scores.append(score)

        # Update epsilon for exploration
        # 探索用のイプシロンを更新
        epsilon = max(eps_end, epsilon * eps_decay)

        # Now evolve population if necessary
        # 必要であれば、母集団を進化させる
        if (idx_epi + 1) % evo_epochs == 0:
            # Evaluate population
            fitnesses = [
                agent.test(
                    env,
                    swap_channels=INIT_HP["CHANNELS_LAST"],
                    max_steps=max_steps,
                    loop=evo_loop,
                )
                for agent in population
            ]

            print(f"Episode {idx_epi + 1}/{max_episodes}")
            print(f'Fitnesses: {["%.2f" % fitness for fitness in fitnesses]}')
            print(
                f'100 fitness avgs: {["%.2f" % np.mean(agent.fitness[-100:]) for agent in population]}'
            )

            # Tournament selection and population mutation
            # トーナメント選択と母集団の変異
            elite, population = tournament.select(population)
            #population = mutations.mutation(population)


        # 5000エピソードごとに、モデルを保存
        if (idx_epi + 1) % 5000 == 0:

            print("Save Model at ep ", (idx_epi+1))
            
            str_idx = f'{(idx_epi+1):07}'

            path = "./result/"+str_dt_now
            filename = "MATD3_trained_agent_"+str_idx+".pt"
            os.makedirs(path, exist_ok=True)
            save_path = os.path.join(path, filename)
            elite.saveCheckpoint(save_path)


            if False:
                # 最高のfitness が得られたモデルを保存
                print("The best fitness : ", max(elite.fitness))
                #print(elite.fitness)
                if elite.fitness[-1]==max(elite.fitness):

                        print("Save the best model! fitness : ", elite.fitness[-1] )
                
                        path = "./result/"+str_dt_now
                        filename = "MATD3_trained_agent_best.pt"
                        os.makedirs(path, exist_ok=True)
                        save_path = os.path.join(path, filename)
                        elite.saveCheckpoint(save_path)
                

    print()
    print("Saving Files...")
    
    # Save the trained algorithm
    # 学習アルゴリズムを保存する
    #path = "./models/MATD3"
    path = "./result/"+str_dt_now
    filename = "MATD3_trained_agent.pt"
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    elite.saveCheckpoint(save_path)

    # population を保存
    filename = "pickle-population.pkl"
    save_path = os.path.join(path, filename)
    with open(save_path, "wb") as f:
        dill.dump(population, f)

    # replaybuffer を保存
    filename = "pickle-replaybuffer.pkl"
    save_path = os.path.join(path, filename)
    with open(save_path, "wb") as f:
        dill.dump(memory, f)

    # tounament を保存
    filename = "pickle-tournament.pkl"
    save_path = os.path.join(path, filename)
    with open(save_path, "wb") as f:
        dill.dump(tournament, f)

    # mutation を保存
    filename = "pickle-mutations.pkl"
    save_path = os.path.join(path, filename)
    with open(save_path, "wb") as f:
        dill.dump(mutations, f)

    #JSONファイルも複製をとっておく
    with open(args.param, mode="rt", encoding="utf-8") as f:
        param_json = json.load(f)
    with open(path+"/parameters.json", mode="wt", encoding="utf-8") as f:
        json.dump(param_json, f, ensure_ascii=False, indent=2)

    print("Done.")