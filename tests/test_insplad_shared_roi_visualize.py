import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'scripts'))

#!/usr/bin/env python3
"""CPU-only checks for fixed selection and ROI display coordinates."""
import unittest
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import Normalize
import insplad_shared_roi_visualize as vis


class Tests(unittest.TestCase):
    def test_fixed_selection_without_scores(self):
        data, detections = {}, {}
        for i in range(5):
            asset=f'asset{i}'
            normal=[f'{asset}/n0',f'{asset}/n1']
            fault=[f'{asset}/f0',f'{asset}/f1',f'{asset}/f2']
            data[asset]=dict(normal=normal,defects={'rust':fault})
            detections[asset]={p:{'fallback':p.endswith('f2')} for p in normal+fault}
        first=vis.choose_examples({'data':data},detections)
        self.assertEqual(first,vis.choose_examples({'data':data},detections))
        self.assertEqual(len({e['path'] for e in first}),20)
        self.assertEqual(sum(e['fallback'] for e in first),5)
        self.assertEqual(sum(e['image_label']=='good' for e in first),5)

    def test_top8_native_coordinates(self):
        d=np.zeros((40,40),dtype=np.float32);d[39,39]=9
        points=vis.top_positions(d,[100,200,500,1000])
        self.assertEqual(len(points),8)
        self.assertEqual((points[0]['x'],points[0]['y']),(495,990))
        self.assertEqual((points[1]['row'],points[1]['col']),(0,0))
        self.assertEqual((points[1]['x'],points[1]['y']),(105,210))

    def test_overlay_is_limited_to_roi(self):
        record=dict(box_native_xyxy=[10,20,50,100],fallback=False)
        fig,ax=plt.subplots()
        vis.paint(ax,Image.new('RGB',(100,200)),record,np.ones((40,40)),Normalize(0,1))
        self.assertEqual(tuple(ax.images[1].get_extent()),(10,50,100,20))
        self.assertEqual(tuple(ax.get_xlim()),(0,100))
        self.assertEqual(tuple(ax.get_ylim()),(200,0))
        plt.close(fig)


if __name__=='__main__':
    unittest.main()
