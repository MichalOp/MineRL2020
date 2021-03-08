import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
import math
from kmeans import cached_kmeans, default_n

import numpy as np
np.set_printoptions(precision=2, suppress=True)

gamma = 0.995

class ResidualBlock(nn.Module):

    def __init__(self,in_layers, size):
        super().__init__()
        self.c1 = nn.Conv2d(in_layers,in_layers,(3,3),padding=1)
        self.c1 = nn.Sequential(nn.Conv2d(in_layers,in_layers,(3,3),padding=1),nn.ReLU())
        self.n1 = nn.LayerNorm((in_layers,size,size))
        self.c2 = nn.Conv2d(in_layers,in_layers,(3,3),padding=1)
        self.c2 = nn.Sequential(nn.Conv2d(in_layers,in_layers,(3,3),padding=1),nn.ReLU())
        self.n2 = nn.LayerNorm((in_layers,size,size))

    def forward(self,x):
        old_x = x
        x = self.n1(x)
        x = torch.relu(x)
        x = self.c1(x)
        x = self.n2(x)
        x = torch.relu(x)
        x = self.c2(x)

        return x+old_x


class ResidualReduceBlock(nn.Module):

    def __init__(self,in_layers, out_layers,size):
        super().__init__()
        self.c1 = nn.Conv2d(in_layers,in_layers,(3,3),padding=1)
        self.n1 = nn.LayerNorm((in_layers,size,size))
        self.c2 = nn.Conv2d(in_layers,out_layers,(4,4),stride=2,padding=1)
        self.n2 = nn.LayerNorm((in_layers,size,size))
        self.bypass = nn.Conv2d(in_layers,out_layers,(4,4),stride=2,padding=1)

    def forward(self,x):
       # x = self.n1(x)
        old_x = self.bypass(x) #lets try it
        x = torch.relu(x)
        x = self.c1(x)
        #x = self.n2(x)
        x = torch.relu(x)
        x = self.c2(x)
        return x+old_x

class FixupResNetCNN(nn.Module):
    """source: https://github.com/unixpickle/obs-tower2/blob/master/obs_tower2/model.py"""

    class _FixupResidual(nn.Module):
        def __init__(self, depth, num_residual):
            super().__init__()
            self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
            self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
            for p in self.conv1.parameters():
                p.data.mul_(1 / math.sqrt(num_residual))
            for p in self.conv2.parameters():
                p.data.zero_()
            self.bias1 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias2 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias3 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.bias4 = nn.Parameter(torch.zeros([depth, 1, 1]))
            self.scale = nn.Parameter(torch.ones([depth, 1, 1]))

        def forward(self, x):
            x = F.relu(x)
            out = x + self.bias1
            out = self.conv1(out)
            out = out + self.bias2
            out = F.relu(out)
            out = out + self.bias3
            out = self.conv2(out)
            out = out * self.scale
            out = out + self.bias4
            return out + x

    def __init__(self, input_channels, double_channels=False):
        super().__init__()
        depth_in = input_channels

        layers = []
        if not double_channels:
            channel_sizes = [32, 64, 64]
        else:
            channel_sizes = [64, 128, 128]
        for depth_out in channel_sizes:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                self._FixupResidual(depth_out, 8),
                self._FixupResidual(depth_out, 8),
            ])
            depth_in = depth_out
        layers.extend([
            self._FixupResidual(depth_in, 8),
            self._FixupResidual(depth_in, 8),
        ])
        self.conv_layers = nn.Sequential(*layers, nn.ReLU())
        self.output_size = math.ceil(64 / 8) ** 2 * depth_in

    def forward(self, x):
        return self.conv_layers(x)

class InputProcessor(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layers = FixupResNetCNN(3,double_channels=True)
        self.spatial_reshape = nn.Sequential(nn.Linear(128*8*8, 896),nn.ReLU(),nn.LayerNorm(896))
        self.nonspatial_reshape = nn.Sequential(nn.Linear(66,128),nn.ReLU(),nn.LayerNorm(128))

    def forward(self, spatial, nonspatial):
        shape = spatial.shape
        spatial = spatial.view((shape[0]*shape[1],)+shape[2:])/255.0
        #print(spatial.shape)
        spatial = self.conv_layers(spatial)

        #print(spatial.shape)

        new_shape = spatial.shape
        spatial = spatial.view(shape[:2]+(-1,))
        features_true = spatial.view(shape[:2]+(64,8,8))
        nonspatial = self.nonspatial_reshape(nonspatial)
        spatial = self.spatial_reshape(spatial)

        return torch.cat([spatial, nonspatial],dim=-1), features_true

class Core(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.input_proc = InputProcessor()
        self.lstm = nn.LSTM(1024, 1024, 1)
        #self.hidden = nn.Sequential(nn.Linear(256+64, 256),nn.ReLU())
        

    def forward(self, spatial, nonspatial, state):
        
        processed, features = self.input_proc.forward(spatial, nonspatial)
        lstm_output, new_state = self.lstm(processed, state)
        #lstm_output = self.hidden(processed)


        return lstm_output+processed, new_state

class SubPolicies(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_proc = InputProcessor()
        self.reflexes = nn.Sequential(nn.Linear(256+64, 256), nn.ReLU(), nn.Linear(256, 64*10))
    
    def forward(self, spatial, nonspatial):
        shape = spatial.shape
        processed = self.input_proc.forward(spatial, nonspatial)
        return self.reflexes(processed).view(shape[:2]+(10,64))


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.kmeans = cached_kmeans("train","MineRLObtainDiamondVectorObf-v0")
        self.core = Core()
        self.selector = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 120))
        #self.embedding = nn.Embedding(120, 32)
        #self.repeat = nn.Sequential(nn.Linear(256+32, 256), nn.ReLU(), nn.Linear(256, 40))
        
        #self.reflexes = SubPolicies()

    def get_zero_state(self, batch_size, device="cuda"):
        return (torch.zeros((1, batch_size, 1024), device=device), torch.zeros((1, batch_size, 1024), device=device))

    def compute_front(self, spatial, nonspatial, state):
        hidden, new_state, features = self.core(spatial, nonspatial, state)
        
        return hidden, self.selector(hidden), new_state, features

    def compute_auxiliary(self, hidden):
        
        front_s = hidden.shape[:2]

        hidden = hidden.view((-1,hidden.shape[2],1,1))
        aux = self.auxiliary(hidden)
        a_s = self.auxiliary_screen(aux)
        a_s_v = self.auxiliary_screen_v(aux)
        a_f = self.auxiliary_features(aux)
        a_f_v = self.auxiliary_features_v(aux)

        a_s = a_s - a_s.mean(dim=1,keepdim=True)
        a_f = a_s - a_f.mean(dim=1,keepdim=True)

        q_s = a_s + a_s_v
        q_f = a_f + a_f_v
        
        # print(q_s.shape)
        # print(q_f.shape)
        return q_s.view(front_s + q_s.shape[1:]), q_f.view(front_s + q_f.shape[1:])


    # def compute_repeat(self, hidden, action):
    #     em = self.embedding(action)
    #     return self.repeat(torch.cat([hidden, em], dim=-1))

    def forward(self, spatial, nonspatial, state, target):
        pass
        #reflex_outputs = self.reflexes(spatial, nonspatial)
        #print(reflex_outputs.sum())
        # hidden, new_state = self.selector(spatial, nonspatial, state, target)
        #print(selection.squeeze())
        #selection = selection.unsqueeze(-1)
        #result = selection*reflex_outputs
        #print(result.shape) 
        #result = result.sum(axis=2)
        # return selection, repeat, new_state

    def get_loss(self, spatial, nonspatial, prev_action, state, target, point, rewards):
        
        loss = nn.CrossEntropyLoss()
        
        hidden, d, state = self.compute_front(spatial, nonspatial, state)

        values = self.values(hidden)
        
        detached_values = values.detach()
        detached_probs = F.softmax(d.detach(),dim=-1)
        
        next_vals = (detached_values*detached_probs).sum(dim=-1)

        value_targets = rewards[:-1] + gamma*next_vals[1:]
        #print(values.shape, point.shape)
        chosen_values = torch.gather(values, 2, point.unsqueeze(-1)).squeeze()[:-1]
        loss2 = torch.nn.MSELoss()
        l2 = loss2(chosen_values, value_targets)
        # print(d.shape)
        # print(point.shape)
        #r = self.compute_repeat(hidden, point)

        l1 = loss(d.view(-1, d.shape[-1]), point.view(-1))
        #l2 = loss(r.view(-1, r.shape[-1]), repeat.view(-1))

        return l1+l2, {"action":l1.item(), "values":l2.item()}, state

    def soft_update(self, source, tau):
        for t, s in zip(self.parameters(), source.parameters()):
            t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

    def get_loss_rl(self, target_model, spatial, nonspatial, state, point, rewards):

        hidden, d, state, features = self.compute_front(spatial, nonspatial, state)
        probs = F.softmax(d,dim=-1)

        with torch.no_grad():
            hidden_t, target_d, _, _ = target_model.compute_front(spatial, nonspatial, state)
            target_values = target_model.values(hidden_t)
            target_probs = probs.detach()
            next_vals = (target_values*target_probs).sum(dim=-1)

            features = features.detach()
            aux_r_feat = (torch.abs(features[:-1] - features[1:])).sum(dim=2, keepdim=True)

            m = nn.MaxPool2d(8, stride=8)
            aux_r_screen = (torch.abs(spatial[:-1] - spatial[1:])).sum(dim=2, keepdim=True)
            aux_r_screen = m(aux_r_screen.view((-1,)+aux_r_screen.shape[2:])).view(aux_r_feat.shape)

            aux_screen_target, aux_feat_target = target_model.compute_auxiliary(hidden_t)
            aux_screen_target,_ = aux_screen_target.max(dim=2, keepdim=True)
            aux_feat_target,_ = aux_feat_target.max(dim=2, keepdim=True)

        loss = nn.CrossEntropyLoss()
        loss2 = torch.nn.MSELoss()

        aux_screen, aux_feat = self.compute_auxiliary(hidden)
        
        pick = point.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        pick = pick.repeat(1,1,1,8,8)

        aux_screen = torch.gather(aux_screen,2, pick)
        # print(aux_screen.shape)
        # print(aux_screen_target.shape)
        aux_feat = torch.gather(aux_feat,2, pick)
        # print(aux_r_screen.shape)
        loss_aux_screen = loss2(aux_screen[:-1], aux_r_screen + gamma*aux_screen_target[1:])*0.001
        loss_aux_feat = loss2(aux_feat[:-1], aux_r_feat + gamma*aux_feat_target[1:])*0.001

        values = self.values(hidden)
        
        detached_values = values.detach()
        

        value_targets = rewards[:-1] + gamma*next_vals[1:]
        #print(values.shape, point.shape)
        chosen_values = torch.gather(values, 2, point.unsqueeze(-1)).squeeze(dim=-1)[:-1]
        
        l2 = loss2(chosen_values, value_targets)

        shape = probs.shape
        l3 = loss(d.view(-1, d.shape[-1]), point.view(-1))
        l1 = -(probs * detached_values).sum()/(shape[0]*shape[1])
        entropy = (probs*torch.log(probs+0.00001)).sum()/(shape[0]*shape[1])*0.05

        return l1+l2+entropy+loss_aux_feat+loss_aux_screen,\
        {"action":l1.item(),
         "values":l2.item(),
         "stability":l3.item(), 
         "entropy":entropy.item(),
         "aux_screen":loss_aux_screen.item(),
         "aux_features":loss_aux_feat.item()},\
        state



    def sample(self, spatial, nonspatial, prev_action, state, target):
        hidden, d, state, _ = self.compute_front(spatial, nonspatial, state)
        dist = D.Categorical(logits = d)
        values = self.values(hidden)
        #print(values.cpu().numpy())
        #print(torch.softmax(d,dim=-1).squeeze().cpu().numpy())
        sam = dist.sample()
        # r = self.compute_repeat(hidden, s)
        # rep_dist = D.Categorical(logits = r)
        s = sam.squeeze().cpu().numpy()
        # rs = rep_dist.sample().squeeze().cpu().numpy()
        return self.kmeans.cluster_centers_[s], sam, state

class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        

        self.c1 = nn.Conv2d(3, 32, (8,8), stride=4, padding=2)
        self.c2 = nn.Conv2d(32, 64, (4,4), stride=2, padding=1)
        self.c3 = nn.Conv2d(64,64, (3,3), padding=1)
        self.c4 = nn.Conv2d(64,64, (3,3), padding=1)
        
        self.spatial_reshape = nn.Sequential(nn.Linear(64*8*8, 256),nn.ReLU())
        self.nonspatial_reshape = nn.Sequential(nn.Linear(64,64),nn.ReLU())

        self.hidden = nn.Sequential(nn.Linear(256+64, 256),nn.ReLU())
        self.output = nn.Sequential(nn.Linear(256,64),)

    def get_zero_state(self, batch_size, device="cuda"):
        return (torch.zeros((1, batch_size, 256), device="cuda"), torch.zeros((1, batch_size, 256), device="cuda"))

    def forward(self, spatial, nonspatial, state, target):
        
        shape = spatial.shape
        spatial = spatial.view((shape[0]*shape[1],)+shape[2:])/255.0
        #print(spatial.shape)
        spatial = F.relu(self.c1(spatial))
        #print(spatial.shape)
        spatial = F.relu(self.c2(spatial))
        old_spatial = spatial
        spatial = self.c3(spatial)
        spatial += old_spatial
        old_spatial = spatial
        spatial = F.relu(spatial)
        spatial = self.c4(spatial)
        spatial = F.relu(spatial+old_spatial)

        #print(spatial.shape)

        new_shape = spatial.shape
        spatial = spatial.view(shape[:2]+(-1,))
        nonspatial = self.nonspatial_reshape(nonspatial)
        spatial = self.spatial_reshape(spatial)
        
        h = self.hidden(torch.cat([spatial, nonspatial],dim=-1))
        result = self.output(h)
        return torch.tanh(result), state



class ProbModel(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv_layers = FixupResNetCNN(3)

        # self.c1 = nn.Conv2d(3, 32, (8,8), stride=4, padding=2)
        # self.c2 = nn.Conv2d(32, 64, (4,4), stride=2, padding=1)
        # self.c3 = nn.Conv2d(64,64, (3,3), padding=1)
        # self.c4 = nn.Conv2d(64,64, (3,3), padding=1)
        
        self.spatial_reshape = nn.Sequential(nn.Linear(64*8*8, 256),nn.ReLU())
        self.nonspatial_reshape = nn.Sequential(nn.Linear(128,128),nn.ReLU())
        self.lstm = nn.LSTM(256+128, 256, 1)
        #self.hidden = nn.Sequential(nn.Linear(256+64, 256),nn.ReLU())
        self.mu = nn.Sequential(nn.Linear(256,64*5),nn.Tanh())
        self.sigma = nn.Sequential(nn.Linear(256,64*64*5))
        self.scale = nn.Linear(256,5)

    def get_zero_state(self, batch_size, device="cuda"):
        return (torch.zeros((1, batch_size, 256), device="cuda"), torch.zeros((1, batch_size, 256), device="cuda"))

    def forward(self, spatial, nonspatial, prev_action, state, target):
        
        shape = spatial.shape
        spatial = spatial.view((shape[0]*shape[1],)+shape[2:])/255.0
        
        spatial = self.conv_layers(spatial)

        #print(spatial.shape)

        new_shape = spatial.shape
        spatial = spatial.view(shape[:2]+(-1,))
        nonspatial = self.nonspatial_reshape(torch.cat([nonspatial, prev_action], dim=-1))
        spatial = self.spatial_reshape(spatial)
        
        h, state = self.lstm(torch.cat([spatial, nonspatial],dim=-1), state)
        mu = self.mu(h).view(shape[:2]+(-1,64))
        #print(mu)
        sigma = self.sigma(h).view(shape[:2]+(-1,64,64))
        sigma_bot = torch.tril(sigma, diagonal=-1)
        sigma_diag = F.elu(torch.diagonal(sigma, dim1=-2, dim2=-1))+1
        #print(sigma_bot.shape)
        #print(sigma_diag.shape)
        sigma =torch.diag_embed(sigma_diag)# +  sigma_bot 
        scale = self.scale(h).view(shape[:2]+(-1,))
        return mu, sigma, scale, state

    def get_distribution(self, spatial, nonspatial, prev_action, state, target):
        
        mu, sigma, scale, state = self.forward(spatial, nonspatial, prev_action, state, target)
        mix = D.Categorical(logits=scale)
        comp = D.MultivariateNormal(loc=mu, scale_tril=sigma)
        d = D.MixtureSameFamily(mix, comp)
        return d, state

    def sample(self, spatial, nonspatial,prev_action, state, target):
        d, state = self.get_distribution(spatial, nonspatial,prev_action, state, target)
        return d.sample(), state

    def get_loss(self, spatial, nonspatial, prev_action, state, target, point):
        
        d, state = self.get_distribution(spatial, nonspatial, prev_action, state, target)
        #print(-d.log_prob(point))
        return -d.log_prob(point), state
    

if __name__ == "__main__":
    model = ProbModel()
    model.cuda()
    model.get_distribution(torch.zeros((1,1,3,64,64),device="cuda"), torch.zeros((1,1,64), device="cuda"), model.get_zero_state(1),torch.zeros((1,1,64), device="cuda") )

