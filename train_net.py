import torch
import torch.nn as nn
from torch.utils import data
from data_process import simpling  # 导入数据加载类
import model.net as nets
import numpy as np
import os


class Trainer:
    """
    训练网络
    """

    def __init__(self, train_net, batch_size, data_path, save_model_path,epoch=100 ,lr=0.001, isCuda=True):
        """
        初始化类
        :param train_net: net
        :param batch_size: 批次大小
        :param data_path: 训练集地址
        :param isCuda： 是否使用CUDA，默认：True
        :param lr: 学习率 默认：0.001
        :param save_model_path: 保存模型地址
        """
        self.epoch = epoch
        self.model = train_net
        self.data_path = data_path
        self.batch_size = batch_size

        self.lr = lr
        self.isCuda = isCuda
        self.save_path = save_model_path

        if os.path.exists(self.save_path):  # 如果有保存的模型，加载模型
            self.model.load_state_dict(torch.load(self.save_path))

        if self.isCuda:
            self.model.cuda()

        self.face_loss = nn.BCELoss()
        self.offset_loss = nn.MSELoss()

        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        # self.train_net()  # 调用训练方法

    # def one_hot(self, data):
    #     """
    #     one_hot编码
    #     :param data:一个值，
    #     :return: one_hot编码后的值
    #     """
    #     hot = np.zeros([2])
    #     hot[data] = 1
    #     return hot

    def train_net(self):
        Epoch = self.epoch  # 记录训练次数
        IMG_DATA = simpling.FaceDataset(self.data_path)  # 获取数据
        for epoch in range(Epoch):  # 将所有数据训练1000次
            self.model.train()
            train_data = data.DataLoader(IMG_DATA, batch_size=self.batch_size, shuffle=True, num_workers=8)
            for i, train in enumerate(train_data):
                # 获取数据
                # img_data ：[512, 3, 24, 24]
                # label ：[512, 1]
                # offset ：[512, 4]
                img_data, label, box_offset = train
                total = len(train_data)
                # print(total)
                label_index = torch.lt(label, 2)  # 删除部分样本
                label = torch.masked_select(label, label_index).view(-1, 1)
                # label = torch.zeros(label.size(0), 2).scatter_(1, label.view(-1,1), 1)
                box_offset_index = torch.gt(label, 0)
                box_offset_index = torch.nonzero(box_offset_index)[:, 0]
                box_offset = box_offset[box_offset_index]  # 删除负样本

                # print(label.shape,box_offset.shape)
                if self.isCuda:
                    img_data = img_data.cuda()
                    box_offset = box_offset.cuda()
                    label = label.cuda()
                # 获取网络输出：P-net
                # face_out : [512, 2, 1, 1]
                # box_offset_out: [512, 4, 1, 1]
                # land_offset_out: [512,10,1,1]
                # R-net、O-net
                # face_out : [512, 2, 1, 1]
                # box_offset_out: [512, 4, 1, 1]
                # land_offset_out: [512,10,1,1]
                # print(img_data.shape)
                face_out, box_offset_out, land_offset_out = self.model(img_data)

                # 降维 [512, 2, 1, 1] => [512,2]
                face_out = face_out.view(-1, 1)
                # print(face_out.shape)
                box_offset_out = box_offset_out.squeeze()
                land_offset_out = land_offset_out.squeeze()

                # 获取1 和 0 做人脸损失
                # one = torch.ne(label, 2)  # one : torch.Size([512, 1])
                # one = one.squeeze()  # one : torch.Size([512]) 掩码输出： 1,0 int8

                # 获取1 和 2 做回归框损失
                # two = torch.ne(label, 0)  # two : [512,1]
                # two = two.squeeze()  # two : [512]

                # 将标签转为one_hot编码
                # label_10 = label[one]  # [batch,1]
                # label_10 = torch.Tensor([self.one_hot(int(i)) for i in label_10.squeeze().numpy()])  # [batch,2]

                face_out = torch.masked_select(face_out.cpu(), label_index).view(-1, 1).cuda()

                # face_out = torch.zeros(face_out.size(0),2).scatter_(1,face_out.view(-1,1),1).cuda()
                # print(label[:10], face_out[:10])
                # 得到人脸损失，和偏移量损失
                box_offset_out = box_offset_out[box_offset_index]
                # print(box_offset_out.shape,box_offset.shape)
                face_loss = self.face_loss(face_out, label)
                box_offset_loss = self.offset_loss(box_offset_out, box_offset)
                # land_offset_loss = self.offset_loss(land_offset_out[two],land_offset[two])
                # 损失相加
                self.loss = face_loss + box_offset_loss  # + land_offset_loss

                # 优化损失
                self.opt.zero_grad()
                self.loss.backward()
                self.opt.step()
                # 每训练100次，输出损失，并保存数据
                i += 1
                if i % 20 == 0:
                    print('Epoch:', epoch, '-', (i), "/", total, ' Loss：', self.loss.cpu().item())
                    torch.save(self.model.state_dict(), self.save_path)


if __name__ == '__main__':
    pnet = nets.PNet()
    rnet = nets.RNet()
    onet = nets.ONet()
    train = Trainer(pnet, 1024, r"C:\\celeba\12", r'model/log_P_train2w_2.pt')#loss 150 0.03
    # train = Trainer(rnet, 1024, r"C:\\celeba\24", r'model/log_R_train2w.pt')#loss 90 0.004
    # train = Trainer(onet, 1024, r"C:\\celeba\48", r'model/log_O_train2w_2.pt',epoch=100)#loss 100 0.001
    train.train_net()
