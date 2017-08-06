import tensorflow as tf
import os
import numpy as np
import cv2

img_width = 208
img_height = 208


# train_dir='G:\\BaiduDownload\\train\\train\\'

# 返回存放图片的路径以及labels
def get_files(file_dir):
    # 猫狗图片列表及标签列表
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    # 将图片的绝对路径填入相应的列表中，并通过文件名中是否包含‘cat’来为该图片添加标签（0/1）
    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    # 输出猫狗分别的图片个数
    print('There are %d cats\n There are %d dogs' % (len(cats), len(dogs)))

    # 将猫狗图片列表和标签列表堆叠在一起
    img_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    # 将图片列表和标签列表放置在二维数组中
    temp = np.array([img_list, label_list])
    # 打乱图片的顺序，预先打乱顺序提高效率
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 将处理过的图片和标签再次分隔开
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


# 生成batch_size大小的数据批次
def get_batch(image, label, image_width, image_height, batch_size, capacity):
    '''
    :param image: 图片列表
    :param label: 标签列表
    :param image_width: 图片宽度
    :param image_height: 图片高度
    :param batch_size: 批次大小
    :param capacity:队列中最多能够容纳的大小
    :return:
        image_batch:图片批次
        label_batch: 标签批次
    '''

    # 将图片格式转换成tensorflow可识别的tf.string
    image = tf.cast(image, tf.string)
    # 将标签格式转换成tf.int32
    label = tf.cast(label, tf.int32)

    # 生成一个输入队列
    input_queue = tf.train.slice_input_producer([image, label])
    '''
    函数注解
    tf.train.slice_input_producer(tensor_list,num_epochs(optional),
                                    shuffle,seed(optional),capacity,
                                    shared_name(optional),name)
    '''

    '''
    属于（参数）注解
    Epoch
        - One epoch:training all the data once
    Iteration/step
    Batch
        例:
            5000 images 
            Batch size = 10
            How many iterations are needed to train all the data once?
                - 5000 / 10 = 500 iterations 
     
    '''

    label = input_queue[1]
    # 图片需要通过tf.read_file()函数将其从队列中读取出来
    image_contents = tf.read_file(input_queue[0])
    # 以JPG，通道数为３的格式将图片解码
    image = tf.image.decode_jpeg(image_contents, channels=3)

    '''
    data argumentation should go to here
    可以提高模型的精度
    '''

    # 将图片的大小改变成制定的长宽，如果长或宽不够制定的长宽，用黑色填充
    image = tf.image.resize_image_with_crop_or_pad(image, image_width, image_height)
    # 对数据标准化，减去均值，除以方差（归一化）
    image = tf.image.per_image_standardization(image)
    # 生成批次
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch


if __name__=='__main__':
    train_dir = 'G:\\BaiduDownload\\train\\train\\'

    import matplotlib.pyplot as plt

    BATCH_SIZE = 8
    CAPACITY = 512
    image_list, label_list = get_files(train_dir)
    image_batch, label_batch = get_batch(image_list, label_list, image_width=img_width, image_height=img_height,
                                         batch_size=BATCH_SIZE, capacity=CAPACITY)
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i < 1:
                img, label = sess.run([image_batch, label_batch])

                for j in np.arange(BATCH_SIZE):
                    print('label:%d' % label[j])
                    plt.imshow(img[j, :, :, :])
                    plt.show()
                i += 1
        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            coord.request_stop()

        coord.join(threads)
