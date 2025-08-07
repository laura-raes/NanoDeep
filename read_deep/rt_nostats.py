import os
import re
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import matplotlib.pyplot as plt
from ont_fast5_api.multi_fast5 import MultiFast5File
from torch.utils.data import dataset
from torch.utils.data import dataloader
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import csv
from read_deep.preprocess import signal_preprocess
class loaddata_from_disk(dataset.Dataset):
	def __init__(self, data_path, label_path, input_length):
		super(loaddata_from_disk, self).__init__()
		self.data_path = data_path
		self.label_path = label_path
		self.lable_name = []
		self.lable = {}
		self.reads_ids = []
		self.data_statistics = {}
		self.input_length = input_length

		all_file = os.listdir(self.label_path)
		all_file = sorted(all_file, key=str.lower)

		for file_name in all_file:
			file_path = os.path.join(self.label_path, file_name)
			if file_name not in self.lable_name:
				self.lable_name.append(file_name)
				self.data_statistics[file_name] = 0
			file = open(file_path, 'r')
			for line in file:
				self.data_statistics[file_name] += 1
				id = re.split(r"[\n,\t, ]", line)[0]
				self.lable[id] = file_name
				self.reads_ids.append(id)
		for index,label in enumerate(self.lable_name):
			print('Remember Label_{} is:{}  '.format(index+1,self.lable_name[index]),end=' ')
		print('\n')
		all_file = os.listdir(self.data_path)
		self.reader_id_index = {}

		pbar = tqdm(total=len(all_file))
		for file_name in all_file:
			file_path = os.path.join(self.data_path, file_name)
			reader = MultiFast5File(file_path, 'r')
			id_list = reader.get_read_ids()
			for id in id_list:
				self.reader_id_index[id] = MultiFast5File(file_path, 'r')
			pbar.update()
		self.error_data = 0

	def data_status(self):
		all_file = os.listdir(self.data_path)
		data_id = []
		lable_id = self.lable.keys()
		for file_name in all_file:
			file_path = os.path.join(self.data_path, file_name)
			reader = MultiFast5File(file_path, 'r')
			id_list = reader.get_read_ids()
			data_id.extend(id_list)
			print(file_name, len(id_list))

		print('lable len:', len(lable_id))
		print('data len:', len(data_id))

		error_data = []
		for id in lable_id:
			if id not in data_id:
				error_data.append(id)
		print('error data num:', len(error_data))

	def __getitem__(self, index):
		id = self.reads_ids[index]
		label = np.zeros(len(self.lable_name))
		label[self.lable_name.index(self.lable[id])] = 1
		signal = self.reader_id_index[id].get_read(id).get_raw_data()
		signal = signal_preprocess(signal)
		if len(signal) > self.input_length:
			signal = signal[0:self.input_length]
		signal = np.pad(signal, ((0, self.input_length - len(signal))), 'constant', constant_values=0)
		signal = signal[np.newaxis,]
		signal = signal.astype(np.float32)
		return signal, label

	def __len__(self):
		return len(self.lable)

class loaddata_from_memory(dataset.Dataset):
	def __init__(self, data_path, input_length):
		super(loaddata_from_memory, self).__init__()
		self.data_path = data_path
		self.reads_ids = []
		self.data_statistics = {}
		self.input_length = input_length
		self.last_mean = 0
		self.mean_len = 0
		all_file = os.listdir(self.data_path)

		self.reader_raw_data = {}
		pbar = tqdm(total=len(all_file))
		for file_name in all_file:
			file_path = os.path.join(self.data_path, file_name)
			reader = MultiFast5File(file_path, 'r')
			id_list = reader.get_read_ids()

			for i, id in enumerate(id_list):
				signal = reader.get_read(id).get_raw_data()
				signal = signal_preprocess(signal)
				if len(signal) > self.input_length:
					
					# Print warning if read will not be processed correctly
					if len(signal) < self.input_length + 1000:
						print(f"Warning: {id} shorter than expected for slicing")
					
					signal = signal[1000:self.input_length + 1000]
				signal = np.pad(signal, ((0, self.input_length - len(signal))), 'constant', constant_values=0)
				signal = signal[np.newaxis,]
				signal = signal.astype(np.float32)
				self.reads_ids.append(id)

				self.reader_raw_data[id] = signal

			pbar.update()
		
	def __getitem__(self, index):
		id = self.reads_ids[index]
		signal = self.reader_raw_data[id]

		return signal

	def data_status(self):
		all_file = os.listdir(self.data_path)
		data_id = []
		for file_name in all_file:
			file_path = os.path.join(self.data_path, file_name)
			reader = MultiFast5File(file_path, 'r')
			id_list = reader.get_read_ids()
			data_id.extend(id_list)
			print(file_name, len(id_list))

		print('data len:', len(data_id))

	def __len__(self):
		return len(self.reads_ids)

class rt_deep:
	def __init__(self, model: nn.Module, input_length: int, device):
		'''

		:param model: 用来分类的模型
		:param signal_length: 输入信号的长度
		'''
		self.model = model
		self.length = input_length

		self.device = torch.device(device)
		self.model.to(self.device)
		self.validation = False
		self.test = False
		print("use:", self.device)

	def load_the_model_weights(self, model_path):
		'''
		:param model_path: 模型参数保存路径
		:return:
		'''
		self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

	def save_model(self, path):
		torch.save(self.model.state_dict(), path)

	def signal_classification(self, signal_data: list):
		self.model.eval()
		input = []
		for i in range(len(signal_data)):
			data = signal_data[i].flatten()
			if data.shape[0] < self.length:
				data = np.pad(data, ((0, self.length - data.shape[0])), 'constant', constant_values=0)
				data = data.reshape((1, -1))
				input.append(data)
			else:
				data = data[0:self.length]
				data = data.reshape((1, -1))
				input.append(data)
		input = np.array(input)
		input = input - np.average(input)
		input = torch.from_numpy(input)
		input = input.to(torch.float)
		input = input.to(self.device)
		output = self.model(input)
		output = torch.softmax(output, dim=1)
		output = output.cpu().detach().numpy()
		return output

	def load_data(self,
				  data_path,
				  dataset: str,  # 'train','validation','test'
				  load_to_mem=False,
				  ):

		if dataset == 'test':
			self.test = True
			if load_to_mem:
				self.test_dataset = loaddata_from_memory(data_path, self.length)
			else:
				self.test_dataset = loaddata_from_disk(data_path, self.length)
				self.test_dataloader = dataloader.DataLoader(
					dataset=self.test_dataset,
					batch_size=1,
					shuffle=False
				)

	def test_model(self,batch_size=50, **kwargs):
		if self.test == False:
			raise ValueError("no test data")
		self.test_dataloader = dataloader.DataLoader(
			dataset=self.test_dataset,
			batch_size=batch_size,
			shuffle=False
		)
		self.model.eval()
		pbar = tqdm(total=self.test_dataloader.__len__())
		predict_proba = []
		output_rows = []
		for input in self.test_dataloader:
			input = input.to(self.device)

			logit = self.model(input, **kwargs)
			
			# Store predictions per read
			start_idx = len(output_rows)
			end_idx = start_idx + len(input)
			batch_indices = self.test_dataset.reads_ids[start_idx:end_idx]
			predicted_classes = torch.argmax(logit, dim=1).cpu().numpy()
			for rid, pred in zip(batch_indices, predicted_classes):
				output_rows.append([rid, pred])
			
		# Write predictions per read to csv
		with open('predictions_per_read.csv', 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['read_id', 'predicted_class'])
			writer.writerows(output_rows)

		return 1


	def data_imformation(self):
		self.train_dataset.data_status()
