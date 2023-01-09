import tensorflow as tf

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def label_generator_loss(target, label_gen_output):
    return tf.reduce_mean(tf.abs(target-label_gen_output))


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def inference_generator_loss(disc_generated_output, inference_gen_output, target, _lambda):

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - inference_gen_output))

    total_inference_generator_loss = l1_loss*_lambda+gan_loss

    return total_inference_generator_loss, gan_loss, l1_loss
