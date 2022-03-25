import tensorflow.compat.v1 as tf

# Left indexとRight indexで指定されたLandmarkを結ぶ線が水平になり、Subset indicesで指定されたLandmrakをちょうど含むような範囲をcropするように、元の画像をAffine変換する行列
# の逆行列を求める。なぜ、逆行列かといういうと、後の計算で使うのが逆行列だから。
def test_landmark2transformMatrix():
    custom_options = {'left_rotation_idx': 61, 
                      'right_rotation_idx': 291, 
                      'output_height': 16,
                      'output_width': 16,
                      'scale_x': 1.5,
                      'scale_y': 1.5,
                      'target_rotation_radians': 0.0,
                      'subset_idxs': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415, 76, 77, 90, 180, 85, 16, 315, 404, 320, 307, 306, 184, 74, 73, 72, 11, 302, 303, 304, 408, 62, 96, 89, 179, 86, 15, 316, 403, 319, 325, 292, 183, 42, 41, 38, 12, 268, 271, 272, 407]
                      }
    operator = {
              "opcode_index": 8,
              "inputs": [191],
              "outputs": [196],
              "builtin_options_type": "NONE",
              "custom_options": [],
              "custom_options_format": "FLEXBUFFERS"
            }

    import numpy as np
    np.arange(468*3)
    input_tensor = tf.ones([1, 468, 3], tf.float32)
        #{
        #"shape": [
        #    1,
        #    468,
        #    3
        #    ],
        #"type": "FLOAT32",
        #"buffer": 0,
        #"name": "multiply_by_constant",
        #"is_variable": false
        #}

    tensors = { 192: input_tensor }