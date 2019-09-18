import numpy as np
import time
import torch
import data_process.utils as utils
import model.net as nets
from model.net import PNet, RNet,ONet
# from model.Module import O_net as ONet
import torchvision.transforms as trans
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
#o_model_path="./model/o_net.pkl"
class Detector:
    def __init__(self, p_model_path="./model/log_P_train2w.pt", r_model_path="./model/log_R_train2w.pt"
                 , o_model_path="./model/log_O_train2w.pt", cond=None, threshold=None,isCuda=True):
        if threshold is None:
            threshold = [0.6, 0.7, 0.7]
        if cond is None:
            cond = [0.6, 0.7, 0.7]
        self.isCuda=isCuda
        self.pnet, self.rnet, self.onet = PNet(), RNet(), ONet()
        self.pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))
        self.pnet.eval()
        self.rnet.load_state_dict(torch.load(r_model_path, map_location=lambda storage, loc: storage))
        self.rnet.eval()
        self.onet.load_state_dict(torch.load(o_model_path, map_location=lambda storage, loc: storage))
        self.onet.eval()
        self.transf = trans.Compose([trans.ToTensor(), ])
        self.cond = cond
        self.threshold=threshold
        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()
    def detect_face(self, img):
        start_time = time.time()
        p_boxes = self.detect_pnet(img)
        print(p_boxes.shape)
        # return p_boxes
        if len(p_boxes) == 0:
            return np.array([])
        end_time = time.time()
        p_time = end_time - start_time
        start_time = time.time()
        r_boxes = self.detect_rnet(img, p_boxes)
        print(r_boxes.shape)
        # return r_boxes
        if r_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        r_time = end_time - start_time
        start_time = time.time()
        o_boxes = self.detect_onet(img, r_boxes)
        print(o_boxes.shape)
        if o_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        o_time = end_time - start_time
        print("total:{},pnet:{},rnet:{},onet:{}".format(p_time + r_time + o_time, p_time, r_time, o_time))
        return o_boxes

    def detect_pnet(self, img):
        w, h = img.size
        min_side = min(w, h)
        scale = 1.0
        new_scale = 1.0
        p_boxes=[]
        while min_side > 12:
            image_tensor = self.transf(img)
            image_tensor = image_tensor.unsqueeze(0)
            if self.isCuda:
                image_tensor=image_tensor.cuda()
            # image_tensor = image_tensor
            # print(image_tensor.shape)
            with torch.no_grad():
                cond, offset,land = self.pnet(image_tensor)

            # cls_map_np = (1, 1,n, m ),reg_np.shape = (1, 4,n, m )
            cls = cond[0][0].cpu().data
            reg = offset[0].cpu().data.numpy()
            # print(map.shape,reg.shape)#torch.Size([257, 388]) torch.Size([4, 257, 388])
            indexs = torch.nonzero(torch.gt(cls, self.cond[0])).numpy()
            # print(indexs.shape)
            # if indexs.size == 0:
            #     continue

            boxes= self.box(indexs, reg, cls[indexs[:,0],indexs[:,1]], new_scale)
            # print(boxes.shape)
            # p_boxes.extend(boxes)
            p_boxes.extend(utils.NMS(boxes, self.threshold[0]))
            scale*=0.7
            nw = int(w*scale)
            nh = int(h*scale)
            new_scale=min(nw,nh)/min(w,h)
            img = img.resize((nw,nh))
            min_side = min(nw,nh)

        # print(np.array(p_boxes).shape)
        return utils.NMS(np.array(p_boxes), 0.7)
        # return np.array(p_boxes)

    def detect_rnet(self, img, p_boxes):
        # print(type(p_boxes))
        sq_p_boxes = utils.convert_to_square(p_boxes)
        # return sq_p_boxes
        img_datas=[]
        for _box in sq_p_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            img_crop = img.crop((_x1, _y1, _x2, _y2))
            img_resize = img_crop.resize((24, 24))
            img_data = self.transf(img_resize)
            img_datas.append(img_data)
        img_datasets = torch.stack(img_datas)
        if self.isCuda:
            img_datasets=img_datasets.cuda()
        with torch.no_grad():
            cond,offset,land=self.rnet(img_datasets)
        cls = cond.cpu().data.numpy()
        # print(cls)
        offset = offset.cpu().data.numpy()
        indexs,_ = np.where(cls>self.cond[1])#[0 1 2 3 4 5]
        # print(indexs.shape)
        # indexs = torch.nonzero(torch.gt(cls, 0.6))
        # print(indexs)
        boxes = sq_p_boxes[indexs]
        # print(sq_p_boxes.shape,boxes.shape)
        nx1,ny1,nx2,ny2 = boxes[:,0].astype(np.int32),boxes[:,1].astype(np.int32),boxes[:,2].astype(np.int32),boxes[:,3].astype(np.int32)
        dw = nx2-nx1
        dh = ny2-ny1
        # print(offset[indexs].shape)
        x1 = nx1+dw*offset[indexs][:,0]
        y1 = ny1+dh*offset[indexs][:,1]
        x2 = nx2+dw*offset[indexs][:,2]
        y2 = ny2+dh*offset[indexs][:,3]
        # print(cls[indexs].T[0].shape)
        boxes = np.array([x1, y1, x2, y2, cls[indexs].T[0]]).T
        # print(boxes.shape)
        return utils.NMS(boxes,self.threshold[1])
    def detect_onet(self, img, r_boxes):
        sq_r_boxes = utils.convert_to_square(r_boxes)
        # return sq_r_boxes
        img_datas = []
        for _box in sq_r_boxes:
            # img_np = self.transf(img).numpy()
            # img_trans = np.transpose(img_np,(1,2,0))
            # print(img_trans.shape)
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            # if _x1>_x2 or _y1>_y2:
            #     continue
            # img_crop = img_trans[_y1:_y2,_x1:_x2,:]
            # print(img_crop.shape)
            # if img_crop.size==0:
            #     continue
            # img_crop = img.crop((_x1, _y1, _x2, _y2))
            # img_resize = cv2.resize(img_crop,(48, 48))
            # img_retrans = np.transpose(img_resize,(2,0,1))
            img_crop = img.crop((_x1, _y1, _x2, _y2))
            img_re = img_crop.resize((48, 48))
            img_data = self.transf(img_re)
            img_datas.append(img_data)
            # img_datas.append(img_retrans)
        img_datasets = torch.stack(img_datas)
        # cond, offset = self.onet(img_datasets)
        if self.isCuda:
            img_datasets=img_datasets.cuda()
        with torch.no_grad():
            cond, offset,land= self.onet(img_datasets)
        cls = cond.cpu().data.numpy()
        # print("cls",cls)
        offset = offset.cpu().data.numpy()
        indexs, _ = np.where(cls > self.cond[2])  # [0 1 2 3 4 5]
        # indexs = torch.nonzero(torch.gt(cls, 0.6))
        # print(indexs)
        boxes = sq_r_boxes[indexs]
        # print(sq_p_boxes.shape,boxes.shape)
        nx1, ny1, nx2, ny2 = boxes[:, 0].astype(np.int32), boxes[:, 1].astype(np.int32), boxes[:, 2].astype(
            np.int32), boxes[:, 3].astype(np.int32)
        dw = nx2 - nx1
        dh = ny2 - ny1
        # print(offset[indexs].shape)
        x1 = nx1 + dw * offset[indexs][:, 0]
        y1 = ny1 + dh * offset[indexs][:, 1]
        x2 = nx2 + dw * offset[indexs][:, 2]
        y2 = ny2 + dh * offset[indexs][:, 3]
        # print(cls[indexs].T[0].shape)
        boxes = np.array([x1, y1, x2, y2, cls[indexs].T[0]]).T
        # print(boxes.shape)
        # return boxes
        return utils.NMS(boxes, self.threshold[2],IsUnion=False)

    # 将回归量还原到原图上去
    def box(self, indexs, reg, map, scale, stride=2.0, side_len=12.0):
        # stride = torch.tensor(stride)
        dx1 = ((stride * indexs[:, 1]) / scale)
        # print(dx1.shape)
        dy1 = ((stride * indexs[:, 0]) / scale)
        dx2 = ((stride * indexs[:, 1] + side_len) / scale)
        dy2 = ((stride * indexs[:, 0] + side_len) / scale)
        ow, oh = (dx2 - dx1), (dy2 - dy1)
        d_reg = reg[:, indexs[:, 0], indexs[:, 1]]

        x1 = dx1 + ow * d_reg[0]
        y1 = dy1 + oh * d_reg[1]
        x2 = dx2 + ow * d_reg[2]
        y2 = dy2 + oh * d_reg[3]
        # print(x1.shape,map.shape)
        boxes = np.array([x1, y1, x2, y2,map.numpy()]).T
        return boxes

if __name__ == '__main__':
    img_file = r"pic/11.jpg"
    detector = Detector(p_model_path="./model/log_P_train2w_2.pt",
                        r_model_path="./model/log_R_train2w.pt",
                        o_model_path="./model/log_O_train2w_2.pt",
                        cond=[0.9,0.7,0.99],
                        threshold=[0.7,0.7,0.7],
                        isCuda=True)
    with Image.open(img_file) as img:
        # print(img.size)#(418, 594)
        boxes = detector.detect_face(img)
        img_draw = ImageDraw.ImageDraw(img)
        for box in boxes:
            x1 = box[0].astype(int)
            y1 = box[1].astype(int)
            x2 = box[2].astype(int)
            y2 = box[3].astype(int)
            # print(box[4])
            img_draw.rectangle((x1, y1, x2, y2), outline="yellow",width=3)
        # img.show()
        plt.imshow(img)
        plt.pause(0)