from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import os
import glob


def dali_dataloader(
        tfrec_filenames,
        tfrec_idx_filenames,
        shard_id=0, num_shards=1,
        batch_size=128, num_threads=os.cpu_count(),
        image_size=224, num_workers=1, training=True):
    pipe = Pipeline(batch_size=batch_size,
                    num_threads=num_threads, device_id=0)
    with pipe:
        inputs = fn.readers.tfrecord(
            path=tfrec_filenames,
            index_path=tfrec_idx_filenames,
            random_shuffle=training,
            shard_id=shard_id,
            num_shards=num_shards,
            initial_fill=10000,
            read_ahead=True,
            pad_last_batch=True,
            prefetch_queue_depth=num_workers,
            name='Reader',
            features={
                'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ""),
                'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64,  -1),
            })
        jpegs = inputs["image/encoded"]
        if training:
            images = fn.decoders.image_random_crop(
                jpegs,
                device="mixed",
                output_type=types.RGB,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100)
            images = fn.resize(images,
                               device='gpu',
                               resize_x=image_size,
                               resize_y=image_size,
                               interp_type=types.INTERP_TRIANGULAR)
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.decoders.image(jpegs,
                                       device='mixed',
                                       output_type=types.RGB)
            images = fn.resize(images,
                               device='gpu',
                               size=int(image_size / 0.875),
                               mode="not_smaller",
                               interp_type=types.INTERP_TRIANGULAR)
            mirror = False

        images = fn.crop_mirror_normalize(
            images.gpu(),
            dtype=types.FLOAT,
            crop=(image_size, image_size),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            mirror=mirror)
        label = inputs["image/class/label"] - 1  # 0-999
        label = fn.element_extract(label, element_map=0)  # Flatten
        label = label.gpu()
        pipe.set_outputs(images, label)

    pipe.build()
    last_batch_policy = LastBatchPolicy.DROP if training else LastBatchPolicy.PARTIAL
    loader = DALIClassificationIterator(
        pipe, reader_name="Reader", auto_reset=True, last_batch_policy=last_batch_policy)
    return loader


class DaliImageNet(dict):
    def __init__(self, root,
                 shard_id=0, num_shards=1,
                 batch_size=128, num_threads=os.cpu_count(),
                 image_size=224, train_batch_size=128,
                 num_workers=1):
        train_pat = os.path.join(root, 'train/*')
        train_idx_pat = os.path.join(root, 'idx_files/train/*')
        train = dali_dataloader(sorted(glob.glob(train_pat)),
                                sorted(glob.glob(train_idx_pat)),
                                shard_id=shard_id,
                                num_shards=num_shards,
                                batch_size=train_batch_size,
                                num_workers=num_workers,
                                training=True)
        test_pat = os.path.join(root, 'validation/*')
        test_idx_pat = os.path.join(root, 'idx_files/validation/*')
        test = dali_dataloader(sorted(glob.glob(test_pat)),
                               sorted(glob.glob(test_idx_pat)),
                               shard_id=shard_id,
                               num_shards=num_shards,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               training=False)
        super().__init__(train=train, test=test)
