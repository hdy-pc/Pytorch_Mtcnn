import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
from data_process import utils
import model.net as nets
from torchvision import transforms
import time

class Detector:
    def __init__(self, pnet_param="./model/log_P_train2w.pt", rnet_param="./model/log_R_train10w.pt", onet_param="./model/log_O_train20w.pt",
                 isCuda=True):

        self.isCuda = isCuda

        self.pnet = nets.PNet()
        self.rnet = nets.RNet()
        self.onet = nets.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_param))
        self.rnet.load_state_dict(torch.load(rnet_param))
        self.onet.load_state_dict(torch.load(onet_param))



        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def detect(self, image):

        start_time = time.time()
        pnet_boxes = self.__pnet_detect(image)
        print("p_end")
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time
        print(pnet_boxes.shape)
        # return pnet_boxes
        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        print(rnet_boxes.shape)
        # return rnet_boxes
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time
        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        print(onet_boxes.shape)
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        t_sum = t_pnet + t_rnet + t_onet

        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes

    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)

            _img_dataset.append(img_data)

        img_dataset =torch.stack(_img_dataset)
        # print(img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset,land = self.rnet(img_dataset)
        # print(_cls)
        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        # print(cls)
        boxes = []
        idxs, _ = np.where(cls > 0.7)
        # print(idxs.shape)
        for idx in idxs:
            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])
        # print(boxes)
        return utils.NMS(np.array(boxes), 0.7)

    def __onet_detect(self, image, rnet_boxes):

        datasets = []
        # print(rnet_boxes)
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        # print(_rnet_boxes.shape)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img_crop = image.crop((_x1, _y1, _x2, _y2))
            img_re = img_crop.resize((48, 48))
            img_data = self.__image_transform(img_re)
            datasets.append(img_data)

        img_dataset = torch.stack(datasets)
        if self.isCuda:
            img_dataset = img_dataset.cuda()
        # print(img_dataset.shape)
        _cls, _offset,land = self.onet(img_dataset)
        # print(_cls)
        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(cls > 0.7)
        for idx in idxs:
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([x1, y1, x2, y2, cls[idx][0]])
        # print(boxes)
        return utils.NMS(np.array(boxes), 0.7,IsUnion=False)

    def __pnet_detect(self, image):


        p_boxes=[]

        img = image
        w, h = img.size
        min_side_len = min(w, h)

        scale = 1.0
        boxes = []
        while min_side_len > 12:

            img_data = self.__image_transform(img)
            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)

            _cls, _offest ,land= self.pnet(img_data)

            cls, offest = _cls[0][0].cpu().data, _offest[0].cpu().data
            idxs = torch.nonzero(torch.gt(cls, 0.6)).numpy()

            for idx in idxs:
                boxes.append(self.__box(idx, offest, cls[idx[0], idx[1]], scale))
            # print(np.array(boxes).shape)
            # p_boxes.extend(utils.NMS(np.array(boxes), 0.5))

            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)

        # print(p_boxes)
        return np.array(utils.NMS(np.array(boxes),0.5))

    # 将回归量还原到原图上去
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):

        _x1 = (start_index[1] * stride) / scale
        _y1 = (start_index[0] * stride) / scale
        _x2 = (start_index[1] * stride + side_len) / scale
        _y2 = (start_index[0] * stride + side_len) / scale

        ow = _x2 - _x1
        oh = _y2 - _y1

        _offset = offset[:, start_index[0], start_index[1]]
        x1 = _x1 + ow * _offset[0]
        # print(_x1.dtype,ow.dtype,_offset.dtype,x1.dtype)
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]


if __name__ == '__main__':

    image_file = r"4.jpg"
    detector = Detector()

    with Image.open(image_file) as im:
        # boxes = detector.detect(im)
        # print("----------------------------")
        boxes = detector.detect(im)
        # print(boxes)
        imDraw = ImageDraw.Draw(im)
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            # print(x1,y1,x2,y2)
            imDraw.rectangle((x1, y1, x2, y2), outline='red',width=3)

        im.show()
