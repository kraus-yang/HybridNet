from thop import profile
from torchstat import stat
import torch
import model
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

# Graph = import_class()
graph = 'graph.ntu_rgb_d.Graph'


# model = agcn.Model
Model = import_class('model.agcn.Model')
model = Model(graph=graph).cuda(0)
model.training = False
input = torch.randn(1,3, 300, 25, 2).cuda(0)
flops, params = profile(model, inputs=(input,))
print(flops/(1e9),"GFLOPs")
print(params/(1e6),"M params")

# stat(model, (3,300, 25, 2))

# from torchsummary import summary
# torch.save(model.state_dict(), '临时文件MobileNet.pth')
# summary(model.cuda(), input_size=(3, 300, 25, 2), batch_size=-1)
