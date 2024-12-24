import torch
import torch.nn.functional as F
# from log import  print_and_log
import numpy as np
import random

from mia_lib.mia_util import obtain_membership_feature
from sklearn.metrics import confusion_matrix

def mia_evaluate(args, model, adversary, device, infset_loader, is_test_set=False):

    model.eval()
    adversary.eval()
    correct = 0
    n = 0
    gain = 0
    binary_confusion_matrix = 0
    N = args.local_crops_number
    for batch_idx, ((tr_inputs, _), (te_inputs, _)) in infset_loader:
        # measure data loading time

        tr_inputs = [tr_input.to(device) for tr_input in tr_inputs]
        te_inputs = [te_input.to(device) for te_input in te_inputs]
        if args.fp16: model_input = model_input.half()

        with torch.no_grad():
            tr_outputs = model(tr_inputs)
            te_outputs = model(te_inputs)

            tr_outputs[1] = torch.cat(tr_outputs[1].chunk(N), dim=1)
            te_outputs[1] = torch.cat(te_outputs[1].chunk(N), dim=1)

            tr_features = obtain_membership_feature(tr_outputs[0], tr_outputs[1], feature_type=args.feature)
            # print(tr_features[0])
            te_features = obtain_membership_feature(te_outputs[0], te_outputs[1], feature_type=args.feature) 
            # print(te_features[0])

            v_is_member_labels = torch.from_numpy(
            np.reshape(np.concatenate((np.ones(tr_features.size(0)), np.zeros(te_features.size(0)))), [-1, 1])).to(device).float()

            attack_model_input = torch.cat((tr_features, te_features))
            
        ## train NN model
        r = np.arange(v_is_member_labels.size()[0]).tolist()
        random.shuffle(r)
        attack_model_input = attack_model_input[r]
        v_is_member_labels = v_is_member_labels[r]

        member_output = adversary(attack_model_input)
        
        correct += ((member_output > 0.5) == v_is_member_labels).sum().item()
        binary_confusion_matrix += confusion_matrix(v_is_member_labels.cpu(), (member_output.cpu() > 0.5)+0)
        n += member_output.size()[0]
        gain += ((v_is_member_labels==1)*(member_output-0.5)).sum() + ((0.5-member_output)*(v_is_member_labels==0)).sum()

    print(binary_confusion_matrix)
    TP, FP, FN = binary_confusion_matrix[1,1], binary_confusion_matrix[0,1], binary_confusion_matrix[1,0]
    print('\n{}: MIA accuracy: {}/{} ({:.3f}%) MIA Gain: {:.3f}% MIA Precision:{}/{} ({:.3f}%) MIA Recall:{}/{} ({:.3f}%) \n'.format(
        'MIA Test evaluation' if is_test_set else 'MIA Evaluation',
        correct, n, 100. * correct / float(n), 100. *gain/float(n), TP, (TP+FP), 100. *TP/float(TP+FP), TP, (TP+FN), 100. *TP/float(TP+FN)))

    return correct / float(n)
