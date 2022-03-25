import tensorflow.compat.v1 as tf

#TODO test
# Left indexとRight indexで指定されたLandmarkを結ぶ線が水平になり、Subset indicesで指定されたLandmrakをちょうど含むような範囲をcropするように、元の画像をAffine変換する行列
# の逆行列を求める。なぜ、逆行列かといういうと、後の計算で使うのが逆行列だから。
def landmark2transformMatrix(operator, custom_options, tensors, interpreter):
    landmarks3d = tensors[operator['inputs'][0]] #float32[1,468,3] landmarks
    landmarks2d = landmarks3d[:,:,0:2] # [1,468,2]

    ######################################
    # get rotation
    ######################################
    rot90 = tf.constant([[ 0, -1],
                         [ 1,  0]]) #[2,2]

    idx_rot_l = custom_options['left_rotation_idx']
    idx_rot_r = custom_options['right_rotation_idx']
    left_ = landmarks2d[:,idx_rot_l,:] #[1,1,2]
    right = landmarks2d[:,idx_rot_r,:] #[1,1,2]

    delta = tf.subtract(right, left_) #[1,1,2]
    length = tf.norm(delta, axis=2, keepdims=True) #[1,1,1]

    u = tf.divide(delta, length) #[1,1,2] = [[[dx, dy]]]
    u = tf.squeeze(u) #[2] #TODO? not needed?
    v = tf.matmul(rot90, u, transpose_b=True) #[2] = [-dy, dx]

    # return values
    # mat_rot_inv = [[ dx,  dy],
    #                [-dy,  dx]]
    # mat_rot     = [[ dx, -dy],
    #                [ dy,  dx]]
    mat_rot_inv = tf.stack([u, v]) #[2,2]
    mat_rot = tf.transpose(mat_rot_inv, perm=[2]) #[2,2]

    ######################################
    # get crop size and center
    ######################################
    subset_idxs = custom_options['subset_idxs'] #[80]
    landmarks2d_subset = tf.gather(landmarks3d, [[subset_idxs]]) #[1,80,2]
    landmarks2d_subset_rotated = tf.matmul(mat_rot_inv, landmarks2d_subset, transpose_b=True) #[1,80,2]
    landmarks2d_subset_rotated_min = tf.reduce_min(landmarks2d_subset_rotated, axis=1) #[1,2]
    landmarks2d_subset_rotated_max = tf.reduce_max(landmarks2d_subset_rotated, axis=1) #[1,2]

    # return values
    crop_size = tf.subtract(landmarks2d_subset_rotated_max, landmarks2d_subset_rotated_min) #[1,2], max - min
    center = tf.multiply(tf.add(landmarks2d_subset_rotated_min, landmarks2d_subset_rotated_max), tf.constant(0.5)) #[1,2], 1/2 * (max + min) #TODO? use tf.reduce_mean?

    ######################################
    # get mat
    ######################################
    output_w = custom_options['output_width']
    output_h = custom_options['output_height']
    scale_x = custom_options['scale_x']
    scale_y = custom_options['scale_y']
    scalingFactor_x = scale_x / output_w
    scalingFactor_y = scale_y / output_h
    scalingFactor = tf.constant([[scalingFactor_x, scalingFactor_y]]) #[1,2]
    scale = tf.multiply(scalingFactor, crop_size) #[1,2]
    scale = tf.squeeze(scale) #[2] #TODO? not needed?

    # = [[sx, sy],
    #    [sx, sy]]
    mat_scale_elementWise = tf.stack([scale, scale], axis=1) #[2,2]

    # mat_rot = [[ sx*dx, -sy*dy],
    #            [ sx*dy,  sy*dx]]
    mat_ret = tf.multiply(mat_rot, mat_scale_elementWise)

    # mat_rot = [[ sx*dx, -sy*dy, 0, minX],
    #            [ sx*dy,  sy*dx, 0, minY]]
    translate = tf.squeeze(landmarks2d_subset_rotated_min) #[2] #TODO? not needed?
    mat_ret = tf.stack([mat_ret, tf.constant([0,0]), translate], axis=1) #[2,4]

    # mat_rot = [[ sx*dx, -sy*dy, 0, minX],
    #            [ sx*dy,  sy*dx, 0, minY],
    #            [     0,      0, 1,    0],
    #            [     0,      0, 0,    1]]
    mat_ret = tf.stack([mat_ret, tf.constant([0,0,1,0]), tf.constant([0,0,0,1])]) #[4,4]
    return mat_ret