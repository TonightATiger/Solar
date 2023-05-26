import os
import cv2
import torch
import mmcv
import numpy as np
import torchvision
import shutil

class ClassifierApi(object):
    def __init__(self, **kwargs):

        self.model_path = kwargs.get('model_path') or 'frozen_model.pt'
        self.device = kwargs.get('device') or None
        self.img_long_size = kwargs.get('img_long_size') or 128
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._construct_model(self.model_path)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.to_rgb =True

        pass
    def _pre_process(self, raw_img):
        # raw_img = raw_img.astype(np.float32)
        # raw_img[:, :, 1] = 255 - raw_img[:, :, 1]

        # h, w = raw_img.shape[:2]
        # scale = self.img_long_size * 1.0 / max(h, w)
        # if h < w:
        #     oh, ow = int(scale * h), self.img_long_size
        # else:
        #     oh, ow = self.img_long_size, int(scale * w)
        raw_img = mmcv.imresize(
            raw_img,
            size=(self.img_long_size, self.img_long_size),
            interpolation='bilinear',
            return_scale=False,
            backend='cv2',
        )
        # raw_img = mmcv.impad(
        #     raw_img,
        #     shape=(self.img_long_size, self.img_long_size),
        #     pad_val=0,
        #     padding_mode='constant',
        # )
        raw_img = raw_img.astype(np.float32)
        raw_img = mmcv.imnormalize(
            raw_img, self.mean, self.std, self.to_rgb)
        raw_img = torch.from_numpy(raw_img.transpose(2, 0, 1))
        raw_img = torch.unsqueeze(raw_img, dim=0)
        return raw_img

    def _construct_model(self, model_path):
        model = torch.jit.load(model_path, map_location=self.device)
        #model = torch.load(model_path, map_location=self.device)
        model.eval()
        return model

    def _infer(self, x):
        with torch.no_grad():
            y = self.model(x)
            class_score, class_id = y.data.max(1)
            return class_score.cpu().numpy(), class_id.cpu().numpy()

    def exec(self, img):
        tensor = self._pre_process(img)
        class_score, class_id = self._infer(tensor.to(self.device))
        return dict(class_id=class_id, class_score=class_score)


if __name__ == '__main__':
    # model_path = r'D:\liushuo\DAIMA\mmclassification-master\work-dir\mbv2\cls-test-0315\best_accuracy_top-1_epoch_51-1.pt'
    model_path = r'D:\LSTEST\openMMLab\DCP-AIXU\DAIMA\YLClassifier\work-dir\resnet_4class_YL\cls-test-0414\epoch_44.pt'
   # model_path = r'D:\liushuo\DAIMA\mmclassification-master\work-dir\resnet_6class\cls-test-0317\best_accuracy_top-1_epoch_31.pt'
    my_inference = ClassifierApi(
        model_path=model_path,
    )
    class_name_list = ['0-Other', '1-YL', '2-BB', '3-Other']
    #是否保存分错类别数据
    IsSaveErrorImage=True
    saveErrorPath=r"D:\Error"
    if not os.path.exists(saveErrorPath):
        os.makedirs(saveErrorPath)
    IsSaveRightImage = True
    saveRightPath=r"D:\Right"
    if not os.path.exists(IsSaveRightImage):
        os.makedirs(IsSaveRightImage)
    #是否保存未拼接原图
    IsSaveOrighSrc=True
    root_dir = r'D:\LSTEST\openMMLab\DCP-AIXU\DATA\15-8-PLClassifier\data_split\val'#测试路径

    shutil.rmtree(saveErrorPath)
    if not os.path.exists(saveErrorPath):
        os.makedirs(saveErrorPath)

    for idx, class_name in enumerate(class_name_list):
        img_dir = os.path.join(root_dir, class_name)
        try:
            img_list = os.listdir(img_dir)
        except:
            print(f'{class_name}路径不存在')
            continue
        a_count, r_count = 0, 0
        a_count_all, r_count_all = 0, 0
        for name in img_list:
            img_data = cv2.imread(os.path.join(img_dir, name))

            res_data = my_inference.exec(img_data)
            if res_data['class_id'] == idx:
                r_count += 1
                r_count_all += 1
            # elif res_data['class_id'] in [0,4,5] and idx in [0,4,5]:
            #      r_count_all += 1
            else:
                if IsSaveErrorImage:
                    dir=os.path.join(saveErrorPath,class_name)
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                    imgErrPath=name.split('.bmp')[0]+'_T'+str(idx)+'_P'+str(res_data['class_id'])+str(res_data['class_score'])+'.bmp'
                    imgErrPathF=os.path.join(dir,imgErrPath)
                    mmcv.imwrite(img_data, imgErrPathF)
            a_count += 1
            a_count_all+=1
        acc = r_count / a_count
        acc_all=r_count_all/(a_count_all+1)
        print(f"single class_name:{class_name}, a:{a_count}, r:{r_count}, acc:{acc} \n")

        # print(f"merge single class_name:{class_name},a:{a_count_all},r:{r_count_all},acc:{acc_all} ")






















