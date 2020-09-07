import numpy as np
import argparse
import random

from Registry.AlgRegistry import TD
from Registry.EnvRegistry import Chain, FourRoomGridWorld
from Registry.ProbRegistry import EightStateOffPolicyRandomFeat, LearnEightPoliciesTileCodingFeat

from Job.JobBuilder import default_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', '-a', type=float, default=default_params['meta_parameters']['alpha'])
    parser.add_argument('-lmbda', '-l', type=float, default=default_params['meta_parameters']['lmbda'])
    parser.add_argument('--algorithm', '-alg', type=str, default=default_params['agent'])
    parser.add_argument('--problem', '-p', type=str, default=default_params['problem'])
    parser.add_argument('--run_number', '-r', type=int, default=default_params['meta_parameters']['run'])
    parser.add_argument('--environment', '-e', type=str, default=default_params['environment'])
    parser.add_argument('--save_path', '-sp', type=str, default='Experiments/')
    args = parser.parse_args()
    np.random.seed(args.run_number)
    random.seed(a=args.run_number)

    environment_dict = {'FourRoomGridWorld': FourRoomGridWorld, 'Chain': Chain}
    env = environment_dict[args.environment]()

    prob_dict = {'EightStateOffPolicyRandomFeat': EightStateOffPolicyRandomFeat,
                 'LearnEightPoliciesTileCodingFeat': LearnEightPoliciesTileCodingFeat}
    prob = prob_dict[args.problem](run_number=args.run_number)

    alg_dict = {'TD': TD}
    alg_params = {
        'TD': {
            'alpha': args.alpha, 'lmbda': args.lmbda, 'run': args.run_number,
            'num_features': prob.num_features, 'GAMMA': prob.GAMMA
        },
        'TDMultiplePolicy': {
            'alpha': args.alpha, 'lmbda': args.lmbda, 'run': args.run_number,
            'num_features': prob.num_features, 'GAMMA': prob.GAMMA
        }
    }
    agent = alg_dict[args.algorithm](prob, **alg_params[args.algorithm])

    print(agent)
    print(prob)
    RMSVE = np.zeros((prob.num_policies, prob.num_steps))
    agent.state = env.reset()
    is_terminal = False
    for step in range(prob.num_steps):
        RMSVE[:, step] = agent.compute_rmsve()
        agent.action = agent.choose_behavior_action()
        agent.next_state, r, is_terminal, info = env.step(agent.action)
        agent.learn(agent.state, agent.next_state, r, is_terminal)
        if is_terminal:
            agent.state = env.reset()
            is_terminal = False
            continue
        agent.state = agent.next_state
    print(np.mean(RMSVE[:, :], axis=0))
