from data_provider.dataset_loader import Dataset_ETT_Hour, Dataset_Custom, Dataset_Pred

data_dict = {
    'ETTh1': Dataset_ETT_Hour,
    'ETTh2': Dataset_ETT_Hour,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    if flag == 'pred':
        Data = Dataset_Pred
    else:
        Data = data_dict[args.data]

    timeenc = 0 if args.embed != 'timeF' else 1

    dataset_obj = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        timeenc=timeenc,
        freq=args.freq,
        flag=flag
    )

    def generator():
        for i in range(len(dataset_obj)):
            yield dataset_obj[i]

    import mindspore.dataset as ds
    dataset = ds.GeneratorDataset(
        source=generator,
        column_names=["seq_x", "seq_y", "seq_x_mark", "seq_y_mark"],
        shuffle=(flag == 'train')
    )
    dataset = dataset.batch(args.batch_size, drop_remainder=(flag == 'train'))

    return dataset
