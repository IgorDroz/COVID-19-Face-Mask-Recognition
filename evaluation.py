import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from densenet import DenseNet
from torch.autograd import Variable
from train import *

if os.path.exists('/StudentData'):
	data_root = '/StudentData'
else:
	data_root = "data"

def evaluate_hw2():
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cnn = DenseNet(growthRate=12, depth=48, reduction=0.5,bottleneck=True, nClasses=1)
	cnn = cnn.to(device)

	params = {'batch_size': 50,
			  'shuffle': False,
			  'num_workers': 4}

	data = glob.glob(os.path.join(data_root, 'test', '*.jpg'))
	list_IDs = list(map(lambda x: x[-12:-6], data))
	labels = list(map(lambda x: int(x[-5:-4]), data))

	channel_means = (0.5226, 0.4494, 0.4206)
	channel_stds = (0.2411, 0.2299, 0.2262)

	testTransform = transforms.Compose([
		transforms.Resize((64, 64), interpolation=Image.BICUBIC),
		transforms.ToTensor(),
		transforms.Normalize(channel_means, channel_stds)
	])

	# Generators
	testing_set = MaskDataset(list_IDs, labels, testTransform,train=False)
	testing_generator = DataLoader(testing_set, **params)

	cnn.load_state_dict(torch.load('latest_99train_97val.pkl', map_location=lambda storage, loc: storage))
	cnn.eval()

	# Test the Model
	correct = 0
	total = 0
	for idx, (images, labels) in enumerate(testing_generator):
		images = images.to(device)
		labels = labels.to(device,dtype= torch.float)
		outputs = cnn(images).view(-1)
		predicted = outputs.round()

		total += labels.size(0)
		correct += (predicted == labels).sum()
		# for i in range(params['batch_size']):
		# 	if predicted[i]!=labels[i]:
		# 		Image.open(data[idx*params['batch_size']+i]).save('./mistakes/'+data[idx*params['batch_size']+i][-12:])

	print('Accuracy of the model on the {} test images: {:.4f}%'.format(total,100 * float(correct) / total))





if __name__ == "__main__":
	evaluate_hw2()
