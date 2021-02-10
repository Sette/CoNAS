import os
import sys
import glob
import json
import numpy as np
import torch
import utils
import logging
import argparse
import pickle
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torch.autograd import Variable
from model import NetworkCIFAR as Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../data/imagenette', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10

def infer(test_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  logits_all = []
  for step, (input, target) in enumerate(test_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)
    
    logits = model(input)
    logits_all.append(logits)

    loss = criterion(logits, target)
    data = {"preds":logits, "loss":loss}
    

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return logits_all, top1.avg, objs.avg



def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  #genotype = eval("genotypes.%s" % args.arch)
  #model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  #model = model.cuda()
    

model_name = "weights_249.pt"

print("-----------------------------------------Carregando modelo: ",model_name)
model = utils.load_from_all(model_name)
#utils.load(model, args.model_path)

logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()
validdir = os.path.join(args.data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
]))

valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

model.drop_path_prob = args.drop_path_prob
logits_all, test_acc, test_obj = infer(valid_queue, model, criterion)
logging.info('test_acc %f', test_acc)
pickle.dump(logits_all, open( "logits_"+str(i)+".p", "wb" ))


if __name__ == '__main__':
  main() 

