import tensorflow as tf

reader = tf.WholeFileReader()
key, value = reader.read(tf.train.string_input_producer(['dog.jpg']))
image0 = tf.image.decode_jpeg(value)

resized_image = tf.image.resize_images(image0, [256, 256], method=tf.image.ResizeMethod.AREA)
cropped_image = tf.image.crop_to_bounding_box(image0, 20, 20, 256, 256)
flipped_image = tf.image.flip_left_right(image0)

img_resize_summary = tf.summary.image('image resized', tf.expand_dims(resized_image, 0))
cropped_image_summary = tf.summary.image('image cropped', tf.expand_dims(cropped_image, 0))
flipped_image_summary = tf.summary.image('image flipped', tf.expand_dims(flipped_image, 0))
histogram_summary = tf.summary.histogram('image hist', image0)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('test/tensorboard', sess.graph)
    summary_all = sess.run(merged)
    summary_writer.add_summary(summary_all, 0)
    summary_writer.close()