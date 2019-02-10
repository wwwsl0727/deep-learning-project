import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import inception_v3


# def inception_score(images, batch_size=5):
#     net = inception_v3(pretrained=True).cuda()
#     scores = []
#     for i in range(int(math.ceil(float(len(images)) / float(batch_size)))):
#         batch = Variable(torch.cat(images[i * batch_size: (i + 1) * batch_size], 0))
#         s, _ = net(batch)  # skipping aux logits
#         scores.append(s)
#     p_yx = F.softmax(torch.cat(scores, 0), 1)
#     p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
#     KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
#     final_score = KL_d.mean()
#     return final_score

def inception_score(images):
    net = inception_v3(pretrained=True).cuda()
    scores = 0.0
    s,_ = net(images)  # skipping aux logits
    scores = s
    p_yx = F.softmax(torch.cat(scores, 0), 1)
    p_y = p_yx.mean(0).unsqueeze(0).expand(p_yx.size(0), -1)
    KL_d = p_yx * (torch.log(p_yx) - torch.log(p_y))
    final_score = KL_d
    return final_score