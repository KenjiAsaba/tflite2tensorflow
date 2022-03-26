# Copyright 2022 Akiya Research Institute, Inc. All Rights Reserved

import tensorflow.compat.v1 as tf

#TODO test
# Left indexとRight indexで指定されたLandmarkを結ぶ線が水平になり、Subset indicesで指定されたLandmrakをちょうど含むような範囲をcropするように、元の画像をAffine変換する行列
# の逆行列を求める。なぜ、逆行列かといういうと、後の計算で使うのが逆行列だから。
def landmarks2transformMatrix(operator, custom_options, tensors, interpreter):
    landmarks3d = tensors[operator['inputs'][0]] #float32[b,468,3] landmarks
    landmarks2d = landmarks3d[:,:,0:2] # [b,468,2]

    ######################################
    # get rotation
    ######################################
    rot90_t = tf.constant([[  0.0,  1.0],
                           [ -1.0,  0.0]]) #[2,2], already transposed

    idx_rot_l = custom_options['left_rotation_idx']
    idx_rot_r = custom_options['right_rotation_idx']
    left_ = landmarks2d[:,idx_rot_l,:] #[b,2]
    right = landmarks2d[:,idx_rot_r,:] #[b,2]

    delta = tf.subtract(right, left_) #[b,2]
    length = tf.norm(delta, axis=1) #[b]

    u = tf.divide(delta, length) #[b,2] = [[ dx, dy]]
    v = tf.matmul(u, rot90_t)    #[b,2] = [[-dy, dx]]

    # return values
    # mat_rot_inv = [[ dx,  dy],
    #                [-dy,  dx]]
    # mat_rot     = [[ dx, -dy],
    #                [ dy,  dx]]
    u = tf.expand_dims(u, axis=1) #[b,1,2]
    v = tf.expand_dims(v, axis=1) #[b,1,2]
    mat_rot_inv = tf.concat([u, v], axis=1) #[b,2,2]
    mat_rot = tf.transpose(mat_rot_inv, perm=[0,2,1]) #[b,2,2]

    ######################################
    # get crop size and center
    ######################################
    subset_idxs = custom_options['subset_idxs'] #[80]
    landmarks2d_subset = tf.gather(landmarks2d, indices=subset_idxs, axis=1) #[b,80,2]
    landmarks2d_subset_rotated = tf.matmul(landmarks2d_subset, mat_rot) #[b,80,2] use mat_rot to avoid transposing
    landmarks2d_subset_rotated_min = tf.reduce_min(landmarks2d_subset_rotated, axis=1) #[b,2]
    landmarks2d_subset_rotated_max = tf.reduce_max(landmarks2d_subset_rotated, axis=1) #[b,2]

    # return values
    crop_size = tf.subtract(landmarks2d_subset_rotated_max, landmarks2d_subset_rotated_min) #[b,2], max - min
    #center = tf.multiply(tf.add(landmarks2d_subset_rotated_min, landmarks2d_subset_rotated_max), tf.constant(0.5)) #[b,2], 1/2 * (max + min)

    ######################################
    # get mat
    ######################################
    output_w = custom_options['output_width']
    output_h = custom_options['output_height']
    scale_x = custom_options['scale_x']
    scale_y = custom_options['scale_y']
    scalingFactor_x = scale_x / output_w
    scalingFactor_y = scale_y / output_h
    scalingFactor = tf.constant([[scalingFactor_x, scalingFactor_y]]) #[b,2]
    scale = tf.multiply(scalingFactor, crop_size) #[b,2]
    scale = tf.expand_dims(scale, axis=1) #[b,1,2] 

    # mat = [[ sx*dx, -sy*dy, 0, tx],
    #        [ sx*dy,  sy*dx, 0, ty]]
    translation = tf.expand_dims(landmarks2d_subset_rotated_min, axis=1) #[b,1,2]
    b = translation.shape[0]
    zeros = tf.zeros([b, 1, 2])
    mat = tf.concat([tf.multiply(scale, u), tf.multiply(scale, v), zeros, translation], axis=1) #[b,4,2]
    mat = tf.transpose(mat, perm=[0,2,1]) #[b,2,4]

    # mat = [[ sx*dx, -sy*dy, 0, tx],
    #        [ sx*dy,  sy*dx, 0, ty],
    #        [     0,      0, 1,  0],
    #        [     0,      0, 0,  1]]
    unit_zw = tf.tile(tf.constant([[[0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]]]), [b,1,1]) #[b,2,4]
    mat = tf.concat([mat, unit_zw], axis=1) #[b,4,4]
    return mat

    #
    # u = tf.squeeze(u) #[2]
    # v = tf.squeeze(v) #[2]
    # mat_rot_inv = tf.stack([u, v]) #[2,2]

    # = [[sx, sy],
    #    [sx, sy]]
    # mat_scale_elementWise = tf.concat([scale, scale], axis=1) #[b,2,2]

    # mat = [[ sx*dx, -sy*dy],
    #        [ sx*dy,  sy*dx]]
    # mat = tf.multiply(mat_rot, mat_scale_elementWise)
    # tf.multiply(scale, u), tf.multiply(scale, v), 

    # mat = [[ sx*dx, -sy*dy, 0, minX],
    #        [ sx*dy,  sy*dx, 0, minY]]
    # mat = tf.transpose(mat)
    #mat = tf.concat([mat, tf.constant([[0.0], [0.0]]), translation], axis=0) #[4,2]
    # mat = tf.transpose(mat) #[2,4]
