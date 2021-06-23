import argparse
import torch

from models.neuralprocess import NeuralProcess
from models.neuralprocess_rev import NeuralProcess_rev
from models.attentiveNP import AttentiveNP
from models.attentiveNP_rev import AttentiveNP_rev
from models.attentiveNP_GP import AttentiveNP_plus
from models.latentODE import LatentODE
from trainer import Trainer_sinusoid
from evaluator import Evaluator_sinusoid
from datasets.Sinusoid.sinusoid import SineData


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='temp_inter')
parser.add_argument('--test-phase', type=boolean_string, default=False)
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--gpu-num', type=int, default=0)

parser.add_argument('--model-type', type=str, default='anp', choices=['np', 'np_rev', 'anp',
                                                                     'anp_rev', 'anp_plus', 'latent_ode'])
parser.add_argument('--h-size', type=int, default=128)

parser.add_argument('--x-size', type=int, default=1)
parser.add_argument('--num-full-x', type=int, default=100)

parser.add_argument('--task', type=str, default='interpolation', choices=['extrapolation', 'interpolation'])
parser.add_argument('--sample-strategy', type=int, default=3, choices=[1, 2, 3])

# only the case when using saved data
parser.add_argument('--cv-idx', type=int, default=0, choices=[0, 1, 2, 3, 4])
parser.add_argument('--num-data', type=int, default=36, choices=[1000, 3000])
parser.add_argument('--noisy-data', type=boolean_string, default=True)
parser.add_argument('--diverse-data', type=boolean_string, default=True)



args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_num}")

if args.model_type == 'anp':
    NP = AttentiveNP(args).to(device)
elif args.model_type == 'anp_rev':
    NP = AttentiveNP_rev(args).to(device)
elif args.model_type == 'np':
    NP = NeuralProcess(args).to(device)
elif args.model_type == 'np_rev':
    NP = NeuralProcess_rev(args).to(device)
elif args.model_type == 'anp_plus':
    NP = AttentiveNP_plus(args).to(device)
elif args.model_type == 'latent_ode':
    NP = LatentODE(args).to(device)
else:
    print('Incorrect model type')
# pytorch_total_params = sum(p.numel() for p in NP.parameters())
# print(pytorch_total_params)
# exit()

if __name__ == '__main__':
    if args.test_phase:
        evaluator = Evaluator_sinusoid(args, NP)
        evaluator.run()
    else:
        trainer = Trainer_sinusoid(args, NP)
        trainer.run()
    print('Finished!')