import torch
import torch.nn as nn


# Edge Reconstruction Network
class Net_Edge(nn.Module):
	def __init__(self, n_blocks = 8, n_channels = 64):
		super(Net_Edge, self).__init__()

		self.conv_in = nn.Conv2d(in_channels = 2,
		                         out_channels = n_channels,
		                         kernel_size = 3,
		                         stride = 1,
		                         padding = 1,
		                         bias = False)
		self.lrelu = nn.LeakyReLU(negative_slope = 0.2)

		self.res_layers = self.make_layers(block = Res_Block,
		                                   n_blocks = n_blocks,
		                                   n_channels = n_channels)

		self.conv_out = nn.Conv2d(in_channels = n_channels,
		                          out_channels = 1,
		                          kernel_size = 3,
		                          stride = 1,
		                          padding = 1,
		                          bias = False)
		self.tanh = nn.Tanh()

	def make_layers(self, block, n_blocks, n_channels):
		layers = []
		for _ in range(n_blocks):
			layers.append(block(n_channels))
		return nn.Sequential(*layers)

	def forward(self, image, event):
		out = torch.cat((image, event),
		                dim = 1)
		out = self.lrelu(self.conv_in(out))
		out = self.res_layers(out)
		out = self.conv_out(out)
		out += image
		out = self.tanh(out)

		return out


# Residual Block
class Res_Block(nn.Module):
	def __init__(self, n_channels):
		super(Res_Block, self).__init__()

		self.conv1 = nn.Conv2d(in_channels = n_channels,
		                       out_channels = n_channels,
		                       kernel_size = 3,
		                       stride = 1,
		                       padding = 1,
		                       bias = False)
		self.bn1 = nn.BatchNorm2d(num_features = n_channels)
		self.lrelu = nn.LeakyReLU(negative_slope = 0.2)

		self.conv2 = nn.Conv2d(in_channels = n_channels,
		                       out_channels = n_channels,
		                       kernel_size = 3,
		                       stride = 1,
		                       padding = 1,
		                       bias = False)
		self.bn2 = nn.BatchNorm2d(num_features = n_channels)

	def forward(self, x):
		out = self.lrelu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		return x + out


# Loss Function
class ER_Loss(nn.Module):
	def __init__(self, lambda_fid, lambda_tv):
		super(ER_Loss, self).__init__()
		self.lambda_fid = lambda_fid
		self.lambda_tv = lambda_tv

		sobel_x = torch.Tensor([[[1,0,-1],[2,0,-2],[1,0,-1]]])
		sobel_y = torch.Tensor([[[1,2,1],[0,0,0],[-1,-2,-1]]])
		sobel = torch.stack((sobel_x, sobel_y))
		sobel = torch.nn.Parameter(data = sobel,
		                           requires_grad = False)

		self.pad = nn.ReplicationPad2d(padding = 1)
		self.conv_grad = nn.Conv2d(in_channels = 1,
		                           out_channels = 2,
		                           kernel_size = 3,
		                           stride = 1,
		                           padding = 0,
		                           bias = False)
		self.conv_grad.weight = sobel

	def forward(self, event, pred, label):
		event *= event

		# Fidelity term
		fid = (1 + self.lambda_fid * event) * (label - pred)
		fid = (fid * fid).mean()

		# TV regularizer
		tv = self.conv_grad(self.pad(pred))
		tv = (tv * tv)
		tv = tv[:,[0],:,:] + tv[:,[1],:,:]
		tv *= (self.lambda_tv * (1 - event))
		tv = (tv * tv).mean()

		return fid + tv


# Build model
def build_model(mode, path = None, lr = None, lambda_fid = 0, lambda_tv = 0, n_blocks = 4, n_channels = 32):
	net = None
	loss_fn = None
	optimizer = None

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	net = Net_Edge(n_blocks = n_blocks,
	               n_channels = n_channels).to(device)
	loss_fn = ER_Loss(lambda_fid = lambda_fid,
	                  lambda_tv = lambda_tv).to(device)

	if (mode == 'train' and path) or mode == 'test':
		net.load_state_dict(torch.load(path,
		                               map_location = device))

	if torch.cuda.device_count() > 1:
		net = nn.DataParallel(net)
		loss_fn = nn.DataParallel(loss_fn)

	if mode == 'train':
		optimizer = torch.optim.Adam(params = net.parameters(),
		                             lr = lr,
		                             betas = (0.9, 0.999))

	return {'network': net, 'loss_fn': loss_fn, 'optimizer': optimizer}


if __name__ == '__main__':
	model = Net_Edge()
	print(model)
	print(sum(p.numel() for p in model.parameters()))

###