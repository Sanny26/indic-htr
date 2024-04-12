from __future__ import absolute_import
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import sys
from PIL import Image
import numpy as np
import pdb
import cv2
import numpy as np
import random
import lmdb
import six
import re
import torch
from torch.utils import data
from torch.utils.data import sampler
from torchvision import transforms
from utils import get_vocabulary, to_numpy

ImageFile.LOAD_TRUNCATED_IMAGES = True


class lmdbDataset(data.Dataset):
    def __init__(self, root, voc, num_samples=np.inf,
                 transform=None, label_transform=None,
                 voc_type='string', lowercase=False,
                 alphanumeric=False, ctc_blank='<b>',
                 return_list=False):
        super(lmdbDataset, self).__init__()

        self.env = lmdb.open(root, max_readers=100, readonly=True)

        assert self.env is not None, "cannot create lmdb from %s" % root
        self.txn = self.env.begin()

        self.voc = voc
        self.transform = transform
        self.label_transform = label_transform
        self.nSamples = int(float(self.txn.get(b"num-samples")))
        self.nSamples = min(self.nSamples, num_samples)

        self.voc = get_vocabulary(voc, voc_type, lowercase, alphanumeric)
        self.char2id = dict(zip(self.voc, range(1, len(self.voc)+1))) # 0 reserved for ctc blank
        self.id2char = dict(zip(range(1, len(self.voc)+1), self.voc))
        self.char2id[ctc_blank] = 0
        self.id2char[0] = ctc_blank
        self.ctc_blank = ctc_blank
        self.lowercase = lowercase
        self.alphanumeric = alphanumeric
        self.rec_num_classes = len(self.id2char)
        self.return_list = return_list

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        img_key = b'image-%09d' % index
        imgbuf = self.txn.get(img_key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            img = Image.open(buf).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        label_key = b'label-%09d' % index
        word = self.txn.get(label_key).decode()

        # file_key = b'fname-%09d' % index
        # fname = self.txn.get(file_key).decode()

        if self.label_transform is not None:
            word = self.label_transform(word)

        # remove accented characters from labels
        out_of_char = f'[^{"".join(self.voc)}]'
        to_remove = re.search(out_of_char, word)
        if to_remove:
           print(f'Accented characters need to removed! -- |{word}|')
           pattern = re.compile(out_of_char)
           word = pattern.sub('', word)
           print(out_of_char, word, to_remove)

        if self.return_list:
            return [img, word, img.size(2)]
        return img, word, img.size(2)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        # img.sub_(0.5).div_(0.5)
        img = img.add(-128.0).div(128.0)
        return img


class resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        image = image.resize(self.size, self.interpolation)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2YUV)[:, :, 0]
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        return image


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1, multiscale=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.multiscale = multiscale

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW

        if self.multiscale:
            mimgH = [self.imgH, 48, 32, 24]
            mimgW = [self.imgW, 128, 88, 64]
            for image in images:
                rid = np.random.randint(0, 4)
                nimages = []
                t_w, t_h = mimgW[rid], mimgH[rid]
                w, h = image.size
                image = image.resize((t_w, t_h))
                pad_width = (w - t_w)//2
                pad_height = (h - t_h)//2
                nimages.append(transforms.Pad((pad_width, pad_height), fill=255))
            images = nimages

        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        return images, labels


class collateXwidth(object):
    def __init__(self, PADDING_CONSTANT=0, invert=False):
        self.PADDING_CONSTANT = (PADDING_CONSTANT-128.0)/128.0
        self.invert = invert

    def __call__(self, batch):
        images, labels, widths = zip(*batch)
        max_w = max(widths)
        _, h, w = images[0].size()
        pimages = torch.zeros((len(images), 1, h, max_w))
        for i in range(len(images)):
            d = torch.ones((h, max_w - widths[i]), dtype=torch.double)*self.PADDING_CONSTANT
            pimages[i, 0] = torch.cat([images[i][0], d], 1)
            if self.invert:
                pimages[i, 0] = 1 - pimages[i, 0]
        return pimages, labels


class collatedict(object):

    def __call__(self, batch):
        images, labels, _ = zip(*batch)
        bsize = len(batch)
        nimages = []
        for i in range(bsize):
            tmp = images[i]
            nimages.append(tmp)
        nimages = torch.stack(nimages)
        return nimages, labels
