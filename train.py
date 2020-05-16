import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score
from PIL import Image
import numpy as np
import glob
import os
import shutil
from densenet import DenseNet


batch_size = 13
v_batch_size = 3

epochs = 100
seed = 42
opt = 'adam'

if os.path.exists('/StudentData'):
	data_root = '/StudentData'
else:
	data_root = "data"

class MaskDataset(Dataset):
	"""Covid19 Mask dataset."""

	def __init__(self, list_IDs, labels, transform=None , train=True):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.labels = labels
		self.list_IDs = list_IDs
		self.transform = transform
		self.train = train

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, idx):
		# Load data and get label
		if self.train:
			X = Image.open(os.path.join(data_root,'train/') + self.list_IDs[idx] + '_' + str(self.labels[idx]) + '.jpg')
		else:
			X = Image.open(os.path.join(data_root,'test/') + self.list_IDs[idx] + '_' + str(self.labels[idx]) + '.jpg')
		y = self.labels[idx]

		if self.transform:
			X = self.transform(X)

		return X,y

def main():

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	save_path = 'model'

	torch.manual_seed(seed)

	if os.path.exists(save_path):
		shutil.rmtree(save_path)
	os.makedirs(save_path, exist_ok=True)

	# Parameters
	params = {'batch_size': batch_size,
			  'shuffle': True,
			  'num_workers': 4}

	v_params = {'batch_size': v_batch_size,
			  'shuffle': True,
			  'num_workers': 4}


	data = glob.glob(os.path.join(data_root,'train','*.jpg'))

	list_IDs = list(map(lambda x: x[-12:-6],data))
	labels = list(map(lambda x: int(x[-5:-4]),data))

	# mean = torch.zeros(3)
	# std = torch.zeros(3)
	# traindata = MaskDataset(list_IDs, labels, transforms.ToTensor())
	# for inputs, _labels in traindata:
	# 	print(inputs.shape)
	# 	for i in range(3):
	# 		mean[i] += inputs[ i, :, :].mean()
	# 		std[i] += inputs[ i, :, :].std()
	# mean.div_(len(traindata))
	# std.div_(len(traindata))
	# print(mean, std)

	X_train,X_val,y_train,y_val = train_test_split(list_IDs,labels,test_size=0.15,random_state=42)

	channel_means = (0.5226, 0.4494, 0.4206)
	channel_stds = (0.2411, 0.2299, 0.2262)

	trainTransform = transforms.Compose([
		transforms.Resize((128,128),interpolation=Image.BICUBIC),
	    transforms.RandomHorizontalFlip(),
	    transforms.ToTensor(),
	    transforms.Normalize(channel_means, channel_stds)
	])

	# Generators
	training_set = MaskDataset(X_train, y_train, trainTransform)
	training_generator = DataLoader(training_set, **params)
	validation_set = MaskDataset(X_val, y_val, trainTransform)
	validation_set_generator = DataLoader(validation_set, **v_params)

	net = DenseNet(growthRate=16, depth=60, reduction=0.5,bottleneck=True, nClasses=1)

	print('  + Number of params: {}'.format(
		sum([p.data.nelement() for p in net.parameters()])))

	net = net.to(device)

	if opt == 'sgd':
		optimizer = optim.SGD(net.parameters(),lr=1e-2,momentum=0.9, weight_decay=1e-4)
	elif opt == 'adam':
		optimizer = optim.Adam(net.parameters(),lr=1e-2 , weight_decay=1e-4)

	counter=0
	max_loss_val = 10**5
	for epoch in range(1, epochs + 1):
		counter = adjust_opt(counter, optimizer)
		train(device, epoch, net, training_generator, optimizer)
		curr_loss_val = _test(device, net, validation_set_generator)
		if curr_loss_val<max_loss_val:
			max_loss_val= curr_loss_val
			torch.save(net.state_dict(), os.path.join(save_path, 'latest.pkl'))
			counter = 0
		else:
			counter+=1


def train(device, epoch, net, trainLoader, optimizer):
	net.train()
	y_true = []
	y_pred = []
	nTrain = len(trainLoader.dataset)

	for batch_idx, (data, target) in enumerate(trainLoader):
		data, target = data.to(device), target.to(device, dtype= torch.float)

		output = net(data).view(-1)
		criterion = nn.BCELoss()
		loss = criterion(output, target)

		# BACKWARD AND OPTIMIZE
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# PREDICTIONS
		pred = np.round(output.detach().cpu())
		target = np.round(target.detach().cpu())
		y_pred.extend(pred.tolist())
		y_true.extend(target.tolist())

		if (batch_idx + 1) % 10 == 0:
			print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Accuracy: %.4f'
				  % (epoch , epochs, batch_idx + 1, nTrain // batch_size, loss.item(),accuracy_score(y_true,y_pred)))

def _test(device, net, testLoader):
	net.train()

	test_loss = []
	y_true = []
	y_pred = []
	with torch.no_grad():
		for data, target in testLoader:
			data, target = data.to(device), target.to(device, dtype= torch.float)
			output = net(data).view(-1)

			criterion = nn.BCELoss()
			test_loss.append(criterion(output, target).item())

			# PREDICTIONS
			pred = np.round(output.detach().cpu())
			target = np.round(target.detach().cpu())
			y_true.extend(target.tolist())
			y_pred.extend(pred.tolist())

		test_loss = np.mean(test_loss)
		print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.2f}%)\n'.format(test_loss, accuracy_score(y_true,y_pred)))
	return test_loss


def adjust_opt(counter, optimizer):
	if counter > 3:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.75
			print('Learning rate has been updates to: {}'.format(param_group['lr']))
		counter=0

	return counter


if __name__=='__main__':
	main()