import torch, torchvision
import PIL
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import  Image
import json
from torch.utils.data.dataset import Dataset
import numpy as np
import os, random
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

class CustomDatasetFromDepthOnline(Dataset):
    def __init__(self, image_arr, tfms):
        """
        an online dataset generator for any number of depth images
        Args:
            image_arr (numpy array): [x,w,h] depth image array
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.image_arr = image_arr
        self.data_len = image_arr.shape[0]
        self.tfms = tfms
    def __getitem__(self, index):
        """
        Args:
            index:

        Returns:
            img_as_tensor: [batch_size, channels(3), height, width], float32
            single_image_label: label 0 or 1, int64

        """
        img_as_img = self.image_arr[index]
        img_as_tensor = self.tfms(img_as_img)
        return img_as_tensor

    def __len__(self):
        return self.data_len


class GraspClassifier():
    def __init__(self, model_path, model_name, num_classes):
        self.device = torch.device('cuda')
        self.num_classes = num_classes
        self._load_class_name()
        self.model_name = model_name
        self.image_size = EfficientNet.get_image_size(self.model_name)  # 224
        self.model = EfficientNet.from_pretrained(model_name=self.model_name, advprop=False, num_classes=self.num_classes)
        self.model.cuda()
        self._load_ckpt(path=model_path)
        # TODO check image shape and dimension is (3,100,100)

    def _load_ckpt(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
    def _load_class_name(self):
        # Load class names
        self.labels_map = json.load(open('labels_map.txt'))
        self.labels_map = [self.labels_map[str(i)] for i in range(2)]

    def _preprocess_img(self,image):
        """
        preprocess data with center crop, transform to tensor and standardization

        Args:
        Returns:

        """
        normalize = transforms.Normalize(mean=[0.4719, 0.4719, 0.4719],
                                         std=[0.2379, 0.2379, 0.2379])
        tfms = transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize(self.image_size, interpolation=PIL.Image.BICUBIC),
                                   # transforms.CenterCrop(self.image_size),
                                   # torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                   # torchvision.transforms.RandomVerticalFlip(p=0.5),
                                   transforms.ToTensor(),
                                   normalize,
                                   ])
        self.test_loader = tfms(image)
        self.test_loader = self.test_loader.unsqueeze(0)

    def _preprocess_imgs(self,images):
        """
        preprocess data with center crop, transform to tensor and standardization

        Args:

        """
        normalize = transforms.Normalize(mean=[0.4719, 0.4719, 0.4719],
                                         std=[0.2379, 0.2379, 0.2379])
        tfms = transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize(self.image_size, interpolation=PIL.Image.BICUBIC),
                                   # transforms.CenterCrop(self.image_size),
                                   transforms.ToTensor(),
                                   normalize, ])
        test_dataset = CustomDatasetFromDepthOnline(images, tfms)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=images.shape[0], shuffle=False)
        self.image_len = len(self.test_loader)

    def evaluate(self,images, mode=1):
        """
        evaluation of the image.
        NOT FINISHED!

        Args:


        Returns:

        """
        if mode ==0:
            self._preprocess_img(images)
            with torch.no_grad():
                self.test_loader = self.test_loader.to(self.device)
                logits = self.model(self.test_loader)
                loss = torch.nn.CrossEntropyLoss().forward(logits, torch.tensor([1]).to(self.device))
                preds = torch.topk(logits, k=1).indices.squeeze(0).tolist()
                print(logits)
            for idx in preds:
                label = self.labels_map[idx]
                prob = torch.softmax(logits, dim=1)[0, idx].item()
                print('{:<75} ({:.2f}%)'.format(label, prob * 100))
            # preds = preds.cpu().numpy()

        elif mode ==1:
            self._preprocess_imgs(images)
            with torch.no_grad():
                for image in self.test_loader:
                    image = image.to(self.device)
                    logits = self.model(image)
                    # score, predicted = torch.max(logits, 1)
                    preds = torch.topk(logits, k=1).indices.squeeze(0).tolist()
                    print(logits)
            for i, idx in enumerate(preds):
                if isinstance(idx, int):
                    idx = idx
                elif isinstance(idx, list):
                    idx = idx[0]
                label = self.labels_map[idx]
                prob = torch.softmax(logits, dim=1)[i, idx].item()
                print('{:<75} ({:.2f}%)'.format(label, prob * 100))
            # preds = preds.cpu().numpy()
            # preds = predicted.cpu().numpy()
            # prob = score.cpu().numpy()
        # return preds
        return idx

if __name__== "__main__":
    Inference = GraspClassifier('/home/kb/MyProjects/peach_grasping/two_finger_grasping/grasp_peach_network/EfficientNet-PyTorch/examples/imagenet/ckpt/ckpt_imagenet_bs_64_lr_0.04_class_2_adam_with_pretrain/model_best.pth.tar', 'efficientnet-b0', 2)

    random.seed(666)
    torch.manual_seed(666)
    cudnn.deterministic = True

    files = []
    class_str = 1
    for r, d, f in os.walk('/home/kb/MyProjects/peach_grasping/two_finger_grasping/examples/image_dataset/val/{}/'.format(str(class_str))):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    debug_rate = []
    for file in files:
        # rand_image_0 = Image.open('./data_aoc_112_V2_2_class/train/1/aocCableGrasp112_2_000064.jpg')
        rand_image_0 = Image.open(file)
        rand_image_0 = rand_image_0.convert('RGB')

        npy_array_0 = np.moveaxis(np.asarray(rand_image_0, np.float32), 2, 0)
        npy_debug_0 = torch.from_numpy(npy_array_0)
        # npy_debug_0 = npy_debug_0.type(torch.FloatTensor)
        result = Inference.evaluate(images=npy_debug_0, mode=0)
        if result== class_str:
            debug_rate.append(1)
        else:
            plt.imshow(rand_image_0)
            plt.show()

            debug_rate.append(0)
    # print(result)
    # print(debug_rate)
    print(np.sum(debug_rate)/len(debug_rate))

    # test multi images
    # rand_image = Image.open('./data_aoc_112_V2_2_class/train/1/aocCableGrasp112_2_000400.jpg')
    # rand_image_1 = Image.open('./data_aoc_112_V2_2_class/train/1/aocCableGrasp112_2_001648_hv.jpg')

    rand_image = Image.open('/home/kb/MyProjects/peach_grasping/two_finger_grasping/examples/image_dataset/val/1/1683724818.png')
    rand_image_1 = Image.open('/home/kb/MyProjects/peach_grasping/two_finger_grasping/examples/image_dataset/val/1/1683724833.png')

    rand_image = rand_image.convert('RGB')
    rand_image_1 = rand_image_1.convert('RGB')

    npy_array = np.expand_dims(np.moveaxis(np.asarray(rand_image), 2, 0), 0)
    npy_array_1 = np.expand_dims(np.moveaxis(np.asarray(rand_image_1), 2, 0), 0)

    npy_debug_2 = torch.from_numpy(np.asarray(np.moveaxis(np.asarray(rand_image_1), 2, 0)))
    # npy_debug_2 = npy_debug_2.type(torch.FloatTensor)
    normalize = transforms.Normalize(mean=[0.4719, 0.4719, 0.4719],
                                     std=[0.2379, 0.2379, 0.2379])
    tfms = transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
                               # transforms.CenterCrop(self.image_size),
                               # torchvision.transforms.RandomHorizontalFlip(p=0.5),
                               # torchvision.transforms.RandomVerticalFlip(p=0.5),
                               transforms.ToTensor(),
                               normalize,
                               ])
    test_loader = tfms(npy_debug_2)

    result = Inference.evaluate(images=npy_debug_2, mode=0)

    # npy_array =

    npy_array_3 = np.concatenate((npy_array_1, npy_array_1))
    npy_debug_3 = torch.from_numpy(npy_array_3)
    npy_debug_3 = npy_debug_3.type(torch.FloatTensor)

    result = Inference.evaluate(images=npy_debug_3, mode=1)
    print(result)