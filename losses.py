"""Define functions to create the triplet loss with online triplet mining."""

import tensorflow as tf



def _pairwise_angular_distances(embeddings):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    # dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    d = tf.split(embeddings, 3, 0)
    positive = d[0]
    ancor = d[1]
    negitive = d[2]



    Eu_norm_p = tf.sqrt(tf.reduce_sum(tf.square(positive), 1, keep_dims=True))
    Eu_norm_a = tf.sqrt(tf.reduce_sum(tf.square(ancor), 1, keep_dims=True))
    Eu_norm_n = tf.sqrt(tf.reduce_sum(tf.square(negitive), 1, keep_dims=True))
    AAT_p_a = tf.abs(tf.matmul(positive, ancor, transpose_a=False, transpose_b=True))
    AAT_a_n = tf.abs(tf.matmul(ancor, negitive, transpose_a=False, transpose_b=True))
    Eu_norm_p_a = tf.matmul(Eu_norm_p, Eu_norm_a, transpose_a=False, transpose_b=True)
    Eu_norm_a_n = tf.matmul(Eu_norm_a, Eu_norm_n, transpose_a=False, transpose_b=True)

    distances_p_a = 1.0 - tf.divide(AAT_p_a,Eu_norm_p_a)
    distances_a_n = 1.0 - tf.divide(AAT_a_n, Eu_norm_a_n)



    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances_p_a = tf.maximum(distances_p_a, 0.0)
    distances_a_n = tf.maximum(distances_a_n, 0.0)

    # if not squared:
    #     # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
    #     # we need to add a small epsilon where distances == 0.0
    #     mask = tf.to_float(tf.equal(distances, 0.0))
    #     distances = distances + mask * 1e-16
    #
    #     distances = tf.sqrt(distances)
    #
    #     # Correct the epsilon added: set the distances on the mask to be exactly 0.0
    #     distances = distances * (1.0 - mask)

    return distances_p_a, distances_a_n

# def angular_loss(anchor, positive, negative, alpha=45, in_degree=True,
#                  reduce='mean'):
#     '''
#     Features, y = dnn(x), must be l2 normalized.
#     '''
#     if in_degree:
#         alpha = np.deg2rad(alpha)
#     # tan(x)^2: [0, ..., pi/4, ..., pi/3] -> [0, ..., 1, ..., 3]
#     # strictly increaseing convex function
#     sq_tan_alpha = np.tan(alpha) ** 2
#     c = (anchor + positive) / 2
#     loss = tf.reduce_sum(tf.square(anchor - positive), 2) \
#            - 4 * sq_tan_alpha * tf.reduce_sum(tf.square(negative - c), 2)
#     return tf.where(tf.less(loss, 0.0), 0.0 * loss, loss)

def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask

def contras_loss(embeddings, margin, labels):
    mask = _get_anchor_negative_triplet_mask(labels)
    d = tf.split(embeddings, 3, 0)




def batch_hard_triplet_loss( embeddings, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    anchor_positive_dist, anchor_negative_dist = _pairwise_angular_distances(embeddings)



    # shape (batch_size, 1)
    hardest_positive_dist = tf.diag_part(anchor_positive_dist)
    hardest_positive_dist = tf.expand_dims(hardest_positive_dist, 1)
    tf.summary.scalar("hardest_positive_dist", tf.reduce_mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)


    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))


    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)
    tf.summary.scalar("hardest_negative_dist", tf.reduce_mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss