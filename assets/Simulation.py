import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class MLP1(torch.nn.Module):
    def __init__(self, device):
        super(MLP1, self).__init__()
        hidden_layers = [nn.Linear(200, 16), nn.Linear(16, 16)]
        output_layer = nn.Linear(16, 200)
        self.mlp = nn.Sequential(*hidden_layers, output_layer)
        self.mlp.to(device)
    def forward(self, x) -> torch.Tensor:
        return self.mlp(x)    
class Test():
    def __init__(self):
        self.device = 'cpu'
        self.model = MLP1(self.device)
        self.optimizer = optim.SGD(self.model.parameters(),lr=0.001)
        self.fig = None

    def run(self):
        data = torch.as_tensor(range(200)).to(self.device).to(torch.float)
        data[100:] = -10.
        data = data[None,None,:]
        self.add_img(0, data, 'gt')
        for i, condition in enumerate(['smoothness', 'smo_with_seg', '3D_planar_seg']):
            self.model = MLP1(self.device)
            self.optimizer = optim.SGD(self.model.parameters(),lr=0.001)
            tq_bar = tqdm(range(2000000))
            for epoch in tq_bar:
                output = self.model(data)
                loss = 0.
                
                # Simulation of occlusion-aware photometric loss. Set to zero at occlusion area (index 80-100)
                l1_loss = torch.norm(output-data, 1,1)
                index = torch.as_tensor(range(200)).to(self.device)
                bool_inx = (index>=80)&(index<=100)
                l1_loss = torch.where(bool_inx, torch.zeros_like(l1_loss), l1_loss)            
                loss += l1_loss.mean()
                
                # Simulation of smoothness loss, smoothness loss with segmentation, and our 3D planar segmentation loss
                if condition == 'smoothness':
                    grad = self.get_grad1(output)
                    grad_loss = grad.mean()
                    loss += grad_loss
                elif condition == 'smo_with_seg':
                    grad = self.get_grad1(output)
                    weight = torch.ones_like(grad)
                    weight[..., 99:101] = weight[..., 0] = weight[..., -1] = 0.
                    grad_loss = (grad*weight).mean()
                    loss += grad_loss
                elif condition == '3D_planar_seg': 
                    grad2 = self.get_grad2(output)
                    weight = torch.ones_like(grad2)
                    weight[..., 99:101] = weight[..., 0] = weight[..., -1] = 0.
                    grad2_loss = (grad2*weight).mean()
                    loss += grad2_loss              
                    
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.add_img(i+1, output, condition)
            # self.save_model(condition)

            
    def get_grad2(self, points):
        points = F.pad(points, (1,1),'constant',0.)
        dz_x = (points[:, :, 1:] - points[:, :, :-1])
        dz_x2 = (dz_x[..., 1:] - dz_x[..., :-1])
        grad_2order = torch.abs(dz_x2)
        return grad_2order
    
    def get_grad1(self, points):
        points = F.pad(points, (0,1),'constant',0.)
        dz_x = (points[:, :, 1:] - points[:, :, :-1])
        grad = torch.abs(dz_x)
        return grad
            
    def add_img(self, i: int, fig, name):
        fig = fig.clone().cpu().detach()
        if self.fig is None:
            self.fig, self.axs = plt.subplots(4, 1)
        self.axs[i].plot(list(range(fig.size()[-1])), fig.squeeze())
        self.axs[i].set_title(name)
        
    def save_model(self, name):
        checkpoint_data = {'self.model': self.model.state_dict(), 'self.optimizer': self.optimizer.state_dict()}
        torch.save(checkpoint_data, f"./{name}.ckpt")
    
if __name__ == "__main__":  
    test = Test()
    test.run()
    plt.tight_layout()
    plt.show()
