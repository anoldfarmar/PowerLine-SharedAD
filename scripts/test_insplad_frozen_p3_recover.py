import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from insplad_frozen_p3_recover import corrected_recall,json_write


class RecoveryTest(unittest.TestCase):
    def test_numpy_summary_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'metrics.json'
            json_write(p,dict(per_class=[dict(Images=np.int64(2626),Instances=np.int64(5933),AP=np.float64(.821))]))
            result=json.loads(p.read_text())
            self.assertEqual(result['per_class'][0]['Instances'],5933)
            self.assertEqual(result['per_class'][0]['AP'],.821)
            before=p.read_bytes()
            with self.assertRaises(ValueError):
                json_write(p,dict(AP=np.float64(float('nan'))))
            self.assertEqual(p.read_bytes(),before)

    def test_one_based_json_ids_for_both_heads(self):
        ims={1:dict(file_name='frame.jpg',width=1280,height=720)}
        anns=[dict(image_id=1,category_id=17,bbox=[10,10,10,10])]
        # C17 maps to zero-based model class12 in the reduced 14-class head.
        pred=[dict(image_id='frame',category_id=13,bbox=[10,10,10,10],score=.9)]
        r=corrected_recall(anns,ims,pred,[17],{12:17})
        self.assertEqual(r['per_class']['17']['small']['recall'],1)
        # In the old 18-class head its model class16 exports JSON category17.
        pred[0]['category_id']=17
        r=corrected_recall(anns,ims,pred,[17],{16:17})
        self.assertEqual(r['per_class']['17']['small']['recall'],1)
        pred[0]['category_id']=0
        with self.assertRaises(AssertionError):
            corrected_recall(anns,ims,pred,[17],{16:17})

    def test_non_target_classes_from_old_head_are_filtered(self):
        ims={1:dict(file_name='frame.jpg',width=1280,height=720)}
        anns=[dict(image_id=1,category_id=17,bbox=[10,10,10,10])]
        mapping={cid-1:cid for cid in list(range(1,13))+[17,18]}
        pred=[dict(image_id='frame',category_id=cid,bbox=[10,10,10,10],score=.9) for cid in [13,14,15,16,17]]
        r=corrected_recall(anns,ims,pred,[17],mapping)
        self.assertEqual(r['per_class']['17']['all']['recall'],1)
        self.assertEqual(r['excluded_non_target_prediction_count'],4)
        self.assertEqual(r['excluded_non_target_category_ids'],[13,14,15,16])
        pred.append(dict(image_id='frame',category_id=19,bbox=[10,10,10,10],score=.9))
        with self.assertRaises(AssertionError):
            corrected_recall(anns,ims,pred,[17],mapping)


if __name__=='__main__':
    unittest.main()
