import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Bottleneck(nn.Module):
	def __init__(self, nChannels, growthRate):
		super(Bottleneck, self).__init__()
		interChannels = 4*growthRate
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
							   bias=False)
		self.bn2 = nn.BatchNorm2d(interChannels)
		self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
							   padding=1, bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = self.conv2(F.relu(self.bn2(out)))
		out_concat = torch.cat((out,x), 1)
		return out_concat,out

class DenseBlock(nn.Module):
	def __init__(self, nChannels, growthRate, nDenseBlocks, bottleneck):
		super(DenseBlock, self).__init__()
		self.nBlocks = nDenseBlocks
		self.layers = ['dense_layer'+str(i) for i in range(int(nDenseBlocks))]
		for i in range(int(nDenseBlocks)):
			if bottleneck:
				setattr(self, self.layers[i], Bottleneck(nChannels, growthRate))
			else:
				setattr(self, self.layers[i], SingleLayer(nChannels, growthRate))
			nChannels += growthRate

	def forward(self, x):
		output = None
		out_concat = x
		out= None
		for i in range(int(self.nBlocks)):
			out_concat ,out = getattr(self,self.layers[i])(out_concat)
			if output is not None and i<int(self.nBlocks)-1:
				output = torch.cat((out,output),1)
			elif i==0:
				output=out
		return torch.cat((out,output),1)

class SingleLayer(nn.Module):
	def __init__(self, nChannels, growthRate):
		super(SingleLayer, self).__init__()
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
							   padding=1, bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = torch.cat((out,x), 1)
		return out

class Transition(nn.Module):
	def __init__(self, nChannels, nOutChannels):
		super(Transition, self).__init__()
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = F.avg_pool2d(out, 2)
		return out

class ChannelAttention(nn.Module):
	def __init__(self, in_planes):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
		self.relu1 = nn.ReLU()
		self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
		max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
		out = avg_out + max_out
		return self.sigmoid(out)

class SpatialAttention(nn.Module):
	def __init__(self, kernel_size=7):
		super(SpatialAttention, self).__init__()

		assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
		padding = 3 if kernel_size == 7 else 1

		self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = torch.mean(x, dim=1, keepdim=True)
		max_out, _ = torch.max(x, dim=1, keepdim=True)
		x = torch.cat([avg_out, max_out], dim=1)
		x = self.conv1(x)
		return self.sigmoid(x)

class DenseNet(nn.Module):
	def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
		super(DenseNet, self).__init__()

		dense_layers = ['dense'+str(i) for i in range(1,5)]
		trans_layers = ['trans'+str(i) for i in range(1,5)]
		# cAttn_layers = ['cAttn' + str(i) for i in range(1, 5)]
		# sAttn_layers = ['sAttn' + str(i) for i in range(1, 5)]

		nDenseBlocks = (depth-4) // 3
		if bottleneck:
			nDenseBlocks //= 2

		nChannels = 2*growthRate
		self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,bias=False)

		for i in range(len(dense_layers)):
			if i is not 0:
				nChannels = nOutChannels
			init_nChannels = nChannels
			setattr(self, dense_layers[i], DenseBlock(nChannels, growthRate, nDenseBlocks, bottleneck))
			# if bottleneck:
			# 	setattr(self,cAttn_layers[i] ,ChannelAttention(4 * growthRate))
			# else:
			# 	setattr(self,cAttn_layers[i] ,ChannelAttention(growthRate))
			# setattr(self,sAttn_layers[i], SpatialAttention())
			nChannels += nDenseBlocks * growthRate
			nOutChannels = int(math.floor(nChannels * reduction))
			setattr(self, trans_layers[i],  Transition(nChannels, nOutChannels))

		nChannels = nOutChannels
		self.dense5 = DenseBlock(nChannels, growthRate, nDenseBlocks, bottleneck)
		nChannels += nDenseBlocks * growthRate

		self.bn1 = nn.BatchNorm2d(nChannels)
		self.fc = nn.Linear(nChannels, nClasses)


	def forward(self, x):
		out = self.conv1(x)
		# out = self.dense1(out)
		# out = self.cAttn1(out)*out
		# out = self.sAttn1(out)*out
		# out = self.trans1(out)
		#
		# out = self.dense2(out)
		# out = self.cAttn2(out)*out
		# out = self.sAttn2(out)*out
		# out = self.trans2(out)
		#
		# out = self.dense3(out)
		# out = self.cAttn3(out)*out
		# out = self.sAttn3(out)*out
		# out = self.trans3(out)
		#
		# out = self.dense4(out)
		# out = self.cAttn4(out)*out
		# out = self.sAttn4(out)*out
		# out = self.trans4(out)

		out = self.trans1(torch.cat((self.dense1(out), out), 1))
		out = self.trans2(torch.cat((self.dense2(out), out), 1))
		out = self.trans3(torch.cat((self.dense3(out), out), 1))
		out = self.trans4(torch.cat((self.dense4(out), out), 1))
		out = torch.cat((self.dense5(out), out), 1)
		out = torch.squeeze(F.max_pool2d(F.relu(self.bn1(out)), 8))
		out = torch.sigmoid(self.fc(out))
		return out