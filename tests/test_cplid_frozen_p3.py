import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'scripts'))

#!/usr/bin/env python3
"""CPU tests for VOC adaptation, immutable derived labels and JSON output."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import cplid_frozen_p3 as subject


class Tests(unittest.TestCase):
    def xml(self,name='insulator',x2=102):
        return f'<annotation><size><width>100</width><height>80</height></size><object><name>{name}</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>{x2}</xmax><ymax>70</ymax></bndbox></object></annotation>'

    def test_voc_boundary_and_yolo_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.xml';p.write_text(self.xml())
            raw,boxes,clips=subject.voc_boxes(p,100,80)
            self.assertEqual(raw,[[10,20,102,70]])
            self.assertEqual(boxes,[[10,20,100,70]])
            self.assertEqual(clips,1)
            parts=subject.yolo_labels(boxes,100,80).split()
            self.assertEqual(parts[0],'0')
            self.assertAlmostEqual(float(parts[1]),.55)
            self.assertAlmostEqual(float(parts[3]),.9)

    def test_unknown_class_and_invalid_box_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.xml'
            for text in [self.xml(name='defect'),self.xml(x2=10),self.xml(x2='nan')]:
                p.write_text(text)
                with self.assertRaises(AssertionError):subject.voc_boxes(p,100,80)

    def test_numpy_json_serialization(self):
        value=subject.plain({'count':np.int64(12),'ap':np.float64(.8),'list':np.array([1,2])})
        self.assertEqual(json.loads(json.dumps(value,allow_nan=False)),{'count':12,'ap':.8,'list':[1,2]})

    def test_derived_file_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'labels/a.txt';subject.write_identical(p,'first\n')
            subject.write_identical(p,'first\n')
            with self.assertRaises(AssertionError):subject.write_identical(p,'changed\n')


if __name__=='__main__':unittest.main()
