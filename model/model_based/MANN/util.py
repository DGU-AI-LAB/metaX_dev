def visualization(FLAGS, meta_train_losses, meta_test_losses, meta_test_accuracy, step):
    
    plt.title('{}way {}shot Meta Learning Loss'.format(FLAGS.num_classes, FLAGS.num_samples))
    plt.plot(step, meta_train_losses, color='red', marker='o', label="meta train")
    plt.plot(step, meta_test_losses, color='blue',  marker='o', label="meta test")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend(['Meta Train Loss', 'Meta Test Loss'])
    plt.savefig('./figures/{}way_{}shot_losses'.format(FLAGS.num_classes, FLAGS.num_samples), dpi=300)
    plt.close()
    
    plt.title('{}way {}shot Meta Test Accuracy'.format(FLAGS.num_classes, FLAGS.num_samples))
    plt.plot(step, meta_test_accuracy, color='red', marker='o', label="meta test accuracy")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.legend(['Meta Test Accuracy'])
    plt.savefig('./figures/{}way_{}shot_accuracy'.format(FLAGS.num_classes, FLAGS.num_samples), dpi=300)
    plt.close()
    
def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """
    
    # loss for last sample
    ce_loss = softmax_cross_entropy_with_logits(labels=labels[:,-1,:,:], logits=preds[:,-1,:,:], axis=-1, name=None)

    mean_ce_loss = tf.reduce_mean(ce_loss)

    return mean_ce_loss