import torch
import torch.nn as nn
import torch.nn.functional as F
from bicubic import BicubicDownSample
from lpips import PerceptualLoss
from InsightFace_Pytorch.model import MobileFaceNet

class Arc_loss(torch.nn.Module):
    """ Arc loss to preserve identity.
    """
    def __init__(self,
                weight_path = 'weights/model_mobilefacenet.pth'):
        super(Arc_loss, self).__init__()
        self.pretrained_arc_face = MobileFaceNet(512)
        self.pretrained_arc_face.load_state_dict(torch.load(weight_path))
        self.pretrained_arc_face.eval()
        self.loss = torch.nn.CosineSimilarity()

    def forward(self, x, y):
        self.pretrained_arc_face = self.pretrained_arc_face.to(x)
        x = F.interpolate(x, (112,112))
        y = F.interpolate(y, (112,112))
        with torch.no_grad():
            x_arc_face = self.pretrained_arc_face(x)
            y_arc_face = self.pretrained_arc_face(y)
        return (1 - self.loss(x_arc_face, y_arc_face)).mean()

class LossBuilder(torch.nn.Module):
    def __init__(self, ref_im, loss_str, eps):
        super(LossBuilder, self).__init__()
        assert ref_im.shape[2]==ref_im.shape[3]
        im_size = ref_im.shape[2]
        factor=1024//im_size
        assert im_size*factor==1024
        self.D = BicubicDownSample(factor=factor)
        self.ref_im = ref_im
        self.parsed_loss = [loss_term.split('*') for loss_term in loss_str.split('+')]
        self.eps = eps
        self.PL = PerceptualLoss()
        self.ArcL = Arc_loss()

    # Takes a list of tensors, flattens them, and concatenates them into a vector
    # Used to calculate euclidian distance between lists of tensors
    def flatcat(self, l):
        l = l if(isinstance(l, list)) else [l]
        return torch.cat([x.flatten() for x in l], dim=0)

    def _loss_l2(self, gen_im_lr, ref_im, **kwargs):
        if 0: # check
            print(f"in L2: --------------> type(gen_im_lr) {type(gen_im_lr)}")
            print(f"in L2: --------------> type(ref_im) {type(ref_im)}")
            print(f"in L2: --------------> gen_im_lr.shape, {gen_im_lr.shape}")
            print(f"in L2: --------------> ref_im.shape, {ref_im.shape}")
            assert 1==2
        return ((gen_im_lr - ref_im).pow(2).mean((1, 2, 3)).clamp(min=self.eps).sum())

    def _loss_l1(self, gen_im_lr, ref_im, **kwargs):
        return 10*((gen_im_lr - ref_im).abs().mean((1, 2, 3)).clamp(min=self.eps).sum())

    # Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    def _loss_geocross(self, latent, **kwargs):
        if(latent.shape[1] == 1):
            return 0
        else:
            X = latent.view(-1, 1, 18, 512)
            Y = latent.view(-1, 18, 1, 512)
            A = ((X-Y).pow(2).sum(-1)+1e-9).sqrt()
            B = ((X+Y).pow(2).sum(-1)+1e-9).sqrt()
            D = 2*torch.atan2(A, B)
            D = ((D.pow(2)*512).mean((1, 2))/8.).sum()
            return D

    def _loss_perceptual(self, gen_im_lr, ref_im, **kwargs):
        plLoss = self.PL(gen_im_lr, ref_im)
        return plLoss
    
    def _loss_arc(self, gen_im_lr, ref_im, **kwargs):
        arcLoss = self.ArcL(gen_im_lr, ref_im)
        return arcLoss

    def forward(self, latent, gen_im):
        var_dict = {'latent': latent,
                    'gen_im_lr': self.D(gen_im),
                    'ref_im': self.ref_im,
                    }
        loss = 0
        loss_fun_dict = {
            'L2': self._loss_l2,
            'L1': self._loss_l1,
            'GEOCROSS': self._loss_geocross,
            'PERCEPTUAL': self._loss_perceptual,
            'ARC': self._loss_arc,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            tmp_loss = loss_fun_dict[loss_type](**var_dict).min()
            losses[loss_type] = tmp_loss
            if 0:#check
                print(f"================================> {loss_type} {type(tmp_loss)} {tmp_loss}")
            loss += float(weight)*tmp_loss
        return loss, losses

