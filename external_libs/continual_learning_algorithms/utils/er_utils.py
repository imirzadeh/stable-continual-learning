import numpy as np

def update_reservior(images, labels, episodic_images, episodic_labels, M, N):
    """
    Update the episodic memory with current example using the reservior sampling
    """
    for er_x, er_y in zip(images, labels):
        if M > N:
            episodic_images[N] = er_x
            episodic_labels[N] = er_y
        else:
            j = np.random.randint(0, N)
            if j < M:
                episodic_images[j] = er_x
                episodic_labels[j] = er_y
        N += 1

    return N

def update_fifo_buffer(images, labels, episodic_images, episodic_labels, task_labels, mem_per_class, count_cls, N):
    for er_x, er_y in zip(images, labels):
        cls = np.unique(np.nonzero(er_y))[-1]
        # Write the example at the location pointed by count_cls[cls]
        cls_to_index_map = np.where(np.array(task_labels) == cls)[0][0]
        with_in_task_offset = mem_per_class  * cls_to_index_map
        mem_index = count_cls[cls] + with_in_task_offset + N
        episodic_images[mem_index] = er_x
        episodic_labels[mem_index] = er_y
        count_cls[cls] = (count_cls[cls] + 1) % mem_per_class

    return

def er_mem_update_hindsight(model, sess, x_hat_batch, y_hat_batch, episodic_images, episodic_labels, episodic_filled_counter, 
                            task_labels, logit_mask, phi_hat, avg_img_vectors, args, loop_over_mem=50):
    """
    Update the episodic memory using hindsight
    """
    # Store the current estimate of the parameters in the star_params
    sess.run(model.set_star_vars)
   
    # Train on the episodic memory to get the new estimate of the parameters
    batch_size = 10
    samples_at_a_time = episodic_filled_counter if (episodic_filled_counter <= batch_size) else batch_size
    for jj in range(loop_over_mem):
        mem_indices = np.random.choice(episodic_filled_counter, samples_at_a_time, replace=False)
        train_x = episodic_images[mem_indices]
        train_y = episodic_labels[mem_indices]
        feed_dict = {model.x: train_x, model.y_: train_y, model.keep_prob: 1.0,
                    model.learning_rate: args.learning_rate}
        feed_dict[model.output_mask] = logit_mask
        _, loss = sess.run([model.train, model.reg_loss], feed_dict=feed_dict)
        if jj % 5 == 0:
            print('Hindsight loss:{}'.format(loss))
    

    # Update this synthetic samples by maximizing forgetting loss while maintaining good performance on the current task
    for jj, cls in enumerate(task_labels):
        y_hat_dense = np.repeat(cls, 1)
        y_hat_one_hot = _dense_to_one_hot(y_hat_dense, model.total_classes)
        
        # Initialize the anchor for this task this class
        sess.run(model.anchor_xx.assign(np.expand_dims(avg_img_vectors[cls], axis=0)))
        
        for ii in range(100):
            feed_dict = {model.y_: y_hat_one_hot, model.phi_hat_reference: phi_hat[cls] , model.keep_prob: 1.0}
            feed_dict[model.output_mask] = logit_mask
            fgt_loss, phi_dist, total_loss, _ = sess.run([model.negForgetting_loss, model.phi_distance, model.hindsight_objective, model.update_hindsight_anchor], feed_dict=feed_dict)
            if ii%100 == 0:
                print('Fgt_loss: {}\t Phi_dist: {}\t Total: {}'.format(fgt_loss, phi_dist, total_loss))

        # Store the learned images in the episodic memory
        offset = jj
        class_x_hat = sess.run(model.anchor_xx)
        x_hat_batch[jj] = class_x_hat
        y_hat_batch[jj] = y_hat_one_hot

    # Restore the weights
    sess.run(model.restore_weights)
    return x_hat_batch, y_hat_batch

def update_avg_image_vectors(train_x, train_y, avg_img_vectors, running_alpha=0.5):
    """
    Updates the average image vectors

    avg_img_vectors => TOTAL_CLASSES x H x W x C
    """
    # For each label in the batch, update the corresponding avg_image_vector
    num_examples = train_x.shape[0]
    for ii in range(num_examples):
        yy = train_y[ii]
        cls = np.nonzero(yy)
        avg_img_vectors[cls] -= (1 - running_alpha) * (avg_img_vectors[cls] - train_x[ii])  # running average

    return 


# -------------------------- Internet APIs ----------------------------------------------------------------------------------------
def _dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

def _get_indices_of_class_examples(train_y, cls):
    """
    Returns the indies of examples with given class label
    """
    global_class_indices = np.column_stack(np.nonzero(train_y))
    class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] == cls][:,np.array([True, False])])
    return class_indices