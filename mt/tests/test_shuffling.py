import tensorflow as tf


def pro(line):        
    line = tf.strings.strip(line)
    columns = tf.strings.split([line], ' ')
    # parse label
    labels = tf.strings.to_number(columns.values[0], out_type=tf.int32)
    labels = tf.reshape(labels,[-1])  

    return labels


def not_complete():

    files = tf.data.Dataset.list_files("../sim_data_w_purchase/test_*.txt", shuffle=True)
    num_files = tf.data.Dataset.cardinality(files)
    print("num files: ", num_files)

    files = (
        files
        .interleave(tf.data.TextLineDataset, num_parallel_calls=tf.data.AUTOTUNE, cycle_length=num_files, deterministic=False)
    )

    dataset = (
        files
        .map(lambda line: pro(line))
        .shuffle(2)
        .repeat(1)
        .batch(2)
        .prefetch(tf.data.AUTOTUNE) 
    )

    for epoch in range(3):
        print("\n")
        print("-------------------")
        print("\n")
        for i, batch in enumerate(dataset):
            print(i, batch.numpy().squeeze(-1))
