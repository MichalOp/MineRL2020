import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
import math
from kmeans import cached_kmeans

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
        self.conv_layers = FixupResNetCNN(3)
        self.spatial_reshape = nn.Sequential(nn.Linear(64*8*8, 256),nn.ReLU())
        self.nonspatial_reshape = nn.Sequential(nn.Linear(64,64),nn.ReLU())

    def forward(self, spatial, nonspatial):
        shape = spatial.shape
        spatial = spatial.view((shape[0]*shape[1],)+shape[2:])/255.0
        #print(spatial.shape)
        spatial = self.conv_layers(spatial)

        #print(spatial.shape)

        new_shape = spatial.shape
        spatial = spatial.view(shape[:2]+(-1,))
        nonspatial = self.nonspatial_reshape(nonspatial)
        spatial = self.spatial_reshape(spatial)

        return torch.cat([spatial, nonspatial],dim=-1)

class Selector(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.input_proc = InputProcessor()
        self.lstm = nn.LSTM(256+64, 256, 1)
        self.selector = nn.Sequential(nn.Linear(256, 256),nn.ReLU(), nn.Linear(256, 120))

    def forward(self, spatial, nonspatial, state, target):
        
        processed = self.input_proc.forward(spatial, nonspatial)

        lstm_output, new_state = self.lstm(processed, state)
        #print(lstm_output.shape)
        #print(target.shape)
        #cat = torch.cat([lstm_output, target],dim=-1)
        
        return self.selector(lstm_output), new_state

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
        self.selector = Selector()
        #self.reflexes = SubPolicies()

    def get_zero_state(self, batch_size, device="cuda"):
        return (torch.zeros((1, batch_size, 256), device=device), torch.zeros((1, batch_size, 256), device=device))

    def forward(self, spatial, nonspatial, state, target):
        
        #reflex_outputs = self.reflexes(spatial, nonspatial)
        #print(reflex_outputs.sum())
        selection, new_state = self.selector(spatial, nonspatial, state, target)
        #print(selection.squeeze())
        #selection = selection.unsqueeze(-1)
        #result = selection*reflex_outputs
        #print(result.shape) 
        #result = result.sum(axis=2)
        return selection, new_state

    def get_loss(self, spatial, nonspatial, prev_action, state, target, point):
        
        loss = nn.CrossEntropyLoss()
        
        d, state = self.forward(spatial, nonspatial, state, target)
        # print(d.shape)
        # print(point.shape)
        l = loss(d.view(-1, d.shape[-1]), point.view(-1))

        return l, state

    def sample(self, spatial, nonspatial, prev_action, state, target):
        
        d, state = self.forward(spatial, nonspatial, state, target)
        dist = D.Categorical(logits = d)
        s = dist.sample().squeeze().cpu().numpy()

        return self.kmeans.cluster_centers_[s], state

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

