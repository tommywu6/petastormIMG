#!/usr/bin/env python3
from glob import glob
import cv2


import os, sys
import numpy as np

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructField, StructType, IntegerType, BinaryType, StringType, TimestampType

from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.codecs import ScalarCodec, CompressedImageCodec, NdarrayCodec
from petastorm.unischema import dict_to_spark_row, Unischema, UnischemaField

ROWGROUP_SIZE_MB = 128 # The same as the default HDFS block size

# The schema defines how the dataset schema looks like
ImageSchema = Unischema('ImageSchema', [
    UnischemaField('path', np.string_, (), ScalarCodec(StringType()), False),
    UnischemaField('image', np.uint8, (1080, 1920, 3), CompressedImageCodec('png'), False)
])

output_url = "file:///home/jovyan/work/petastorm_ingest_test/"
rows_count = 1

def ingest_folder(images_folder, spark):

    # List all images in the folder
    image_files = sorted(glob(os.path.join(images_folder, "*.jpg")))

    # Read all images at once
    image_df = spark.read.format("image").load(image_files)
    # image_df.count()

    print('Schema of image_df')
    print('--------------------------')
    image_df.printSchema()

    with materialize_dataset(spark, output_url, ImageSchema, ROWGROUP_SIZE_MB):

        input_rdd = spark.sparkContext.parallelize(image_files) \
            .map(lambda image_path:
                    {ImageSchema.path.name: image_path,
                     ImageSchema.image.name: cv2.imread(image_path)})

        rows_rdd = input_rdd.map(lambda r: dict_to_spark_row(ImageSchema, r))
        spark.createDataFrame(rows_rdd, ImageSchema.as_spark_schema()) \
            .coalesce(10) \
            .write \
            .mode('overwrite') \
            .option('compression', 'none') \
            .parquet(output_url)

def main():

    # Start the Spark session
    spark = SparkSession.builder.config('spark.driver.memory', '4g').master('local[*]').getOrCreate()    
    sc = spark.sparkContext

    # Ingest images and annotations for a given folder
    ingest_folder("../work/data/JPEGImages", spark)

if __name__ == '__main__':
    main()