"""CPU tests: geometry, matching, actual bbox crop input, and ranking statistic."""
import unittest
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
from PIL import Image
import torch

from cplid_shared_p3_core import affine, crop_box, iou, match, to_input
from cplid_shared_p3 import roc


class Tests(unittest.TestCase):
    def test_letterbox(self):
        self.assertEqual(affine((1152,864)),(1280/1152,(0,160)))
        torch.testing.assert_close(to_input([[0,0,1152,864]],(1152,864)),torch.tensor([[0.,160.,1280.,1120.]]))
        self.assertEqual(affine((864,1152)),(1280/1152,(160,0)))

    def test_matching(self):
        p=[dict(xyxy=[0,0,10,10]),dict(xyxy=[0,0,10,10]),dict(xyxy=[20,20,30,30])]
        self.assertEqual(match(p,[[0,0,10,10],[20,20,30,30]]),{0:0,2:1})
        self.assertEqual(match([],[[0,0,10,10]]),{})
        self.assertEqual(iou([0,0,10,10],[20,20,30,30]),0)

    def test_crop_content(self):
        array=np.zeros((20,40,3),np.uint8);array[:,20:]=255
        im=Image.fromarray(array)
        a=np.asarray(im.crop(crop_box([0,0,20,20],im.size)))
        b=np.asarray(im.crop(crop_box([20,0,40,20],im.size)))
        self.assertEqual(a.max(),0);self.assertEqual(b.min(),255)
        self.assertEqual(crop_box([-.2,-.1,40.2,20.2],im.size),[0,0,40,20])

    def test_roc(self):
        self.assertEqual(roc([dict(label=0,score=0),dict(label=1,score=1)]),1)
        self.assertEqual(roc([dict(label=0,score=1),dict(label=1,score=1)]),.5)


if __name__=='__main__':unittest.main()
