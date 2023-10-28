import pytest

import numpy as np

from qonnx.util.range_analysis import range_analysis
from qonnx.util.test import download_model, test_model_details

model_details_stuckchans = {
    "MobileNetv1-w4a4": {
        "stuck_chans": {
            "Quant_29_out0": [
                (0, 0.4813263),
                (4, 0.0),
                (6, 0.0),
                (10, 0.0),
                (13, 0.0),
                (15, 0.0),
                (16, 0.0),
                (19, 0.0),
                (26, 0.0),
                (28, 0.0),
            ],
            "Quant_30_out0": [
                (0, 0.0),
                (4, 0.0),
                (6, 0.0),
                (10, 0.0),
                (13, 0.15743902),
                (15, 0.0),
                (16, 0.47231707),
                (19, 0.0),
                (26, 0.0),
                (28, 0.0),
            ],
            "Quant_31_out0": [(42, 0.0)],
            "Quant_32_out0": [(42, 0.0)],
            "Quant_35_out0": [(102, 0.0)],
            "Quant_36_out0": [(102, 0.0)],
        }
    },
    "FINN-CNV_W2A2": {
        "stuck_chans": {
            "Quant_10_out0": [(5, -1.0), (10, 1.0), (26, 1.0), (30, -1.0), (34, -1.0), (54, -1.0)],
            "Quant_11_out0": [(30, 1.0), (35, 1.0), (37, -1.0), (42, 1.0), (45, -1.0), (57, -1.0)],
            "Quant_13_out0": [(40, -1.0)],
            "Quant_14_out0": [(4, 1.0), (175, 1.0), (209, -1.0)],
            "Quant_16_out0": [
                (5, -1.0),
                (50, 1.0),
                (77, -1.0),
                (95, -1.0),
                (153, 1.0),
                (186, 1.0),
                (199, 1.0),
                (209, -1.0),
                (241, 1.0),
                (329, 1.0),
                (340, 1.0),
                (465, -1.0),
                (478, -1.0),
                (510, -1.0),
            ],
            "Quant_17_out0": [(101, -0.0), (230, -0.0), (443, 0.0)],
        }
    },
}

# inherit basics for matching testcases from test util
model_details = {k: v for (k, v) in test_model_details.items() if k in model_details_stuckchans.keys()}
model_details = {**model_details, **model_details_stuckchans}


@pytest.mark.parametrize("model_name", model_details.keys())
def test_range_analysis(model_name):
    #model = download_model(model_name, return_modelwrapper=True)
    model = ModelWrapper(model)
    irange = test_model_details[model_name]["input_range"]
    ret = range_analysis(model, irange=irange, report_mode="stuck_channel", key_filter="Quant", do_cleanup=True)
    golden_stuck_channels = model_details[model_name]["stuck_chans"]
    for tname, ret_chans in ret.items():
        tg_chans = golden_stuck_channels[tname]
        for i in range(len(tg_chans)):
            tg_ind, tg_val = tg_chans[i]
            ret_ind, ret_val = ret_chans[i]
            assert tg_ind == ret_ind
            assert np.isclose(tg_val, ret_val)


from qonnx.core.modelwrapper import ModelWrapper

m_path = './.onnx'
model = ModelWrapper(m_path)
ret = range_analysis(model, irange=[-128,127], report_mode="stuck_channel", key_filter="Quant", do_cleanup=True)
print(ret)