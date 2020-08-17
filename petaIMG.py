#!/usr/bin/env python3

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
    UnischemaField('image', np.uint8, (320, 263, 3), CompressedImageCodec('png'), False)
])


output_url = "file:///tmp/petastorm_ingest_test/"
rows_count = 1

def row_generator(x):
    """Converts single row into a dictionary"""
    print("Creating row for: " + x.path)
    return {'path': x.path,
            'image': np.reshape(x.image, (320,263,3))}

def ingest_folder(images_folder, spark):

    image_files = "file:///"+os.path.abspath(images_folder)+"/*.jpg"
    print(image_files)
    # Read all images at once
    image_df = spark.read.format("image").load(image_files)

    print('Schema of image_df')
    print('--------------------------')
    image_df.printSchema()

    with materialize_dataset(spark, output_url, ImageSchema, ROWGROUP_SIZE_MB):

        set_df = image_df.select(image_df.image.origin.alias('path'), image_df.image.data.alias('image'))

        print('Schema of set_df')
        print('--------------------------')
        set_df.printSchema()
        print(ImageSchema.as_spark_schema())
        
        print('Saving to parquet')
        print('--------------------------')

        rows_rdd = set_df.rdd\
            .map(row_generator)\
            .map(lambda x: dict_to_spark_row(ImageSchema, x))

        spark.createDataFrame(rows_rdd, ImageSchema.as_spark_schema()) \
            .coalesce(10) \
            .write \
            .mode('overwrite') \
            .option('compression', 'snappy') \
            .parquet(output_url)


def main():

    # Start the Spark session
    spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[*]').getOrCreate()    
    sc = spark.sparkContext

    # Ingest images and annotations for a given folder
    ingest_folder("../work/flower_photos/sample/", spark)

if __name__ == '__main__':
    main()