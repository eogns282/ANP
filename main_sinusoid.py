import argparse
import torch

from models.neuralprocess import NeuralProcess
from models.neuralprocess_rev import NeuralProcess_rev
from models.attentiveNP import AttentiveNP
from models.attentiveNP_rev import AttentiveNP_rev
from models.attentiveNP_det import AttentiveNP_det
from trainer import Trainer_sinusoid
from evaluator import Evaluator_sinusoid


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='temp')
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--gpu-num', type=int, default=0)
parser.add_argument('--cv-idx', type=int, default=0, choices=[0, 1, 2, 3, 4])
parser.add_argument('--x-size', type=int, default=1)
parser.add_argument('--h-size', type=int, default=20)
parser.add_argument('--test-phase', type=boolean_string, default=False)
parser.add_argument('--num-data', type=int, default=3000, choices=[1000, 3000])
parser.add_argument('--noisy-data', type=boolean_string, default=True)
parser.add_argument('--diverse-data', type=boolean_string, default=True)
parser.add_argument('--model-type', type=str, default='anp_det', choices=['np', 'np_rev', 'anp', 'anp_rev',
                                                                          'anp_det'])

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_num}")

if args.model_type == 'anp':
    NP = AttentiveNP(args).to(device)
elif args.model_type == 'anp_rev':
    NP = AttentiveNP_rev(args).to(device)
elif args.model_type == 'anp_det':
    NP = AttentiveNP_det(args).to(device)
elif args.model_type == 'np':
    NP = NeuralProcess(args).to(device)
else:
    NP = NeuralProcess_rev(args).to(device)
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