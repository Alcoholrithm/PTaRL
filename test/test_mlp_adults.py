def test_mlp_adult():
    random_seed = 0
    batch_size = 128
    early_stopping_patience = 16
    
    import torch
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = [0]
    else:
        accelerator = 'cpu'
        devices = 'auto'
        
    n_jobs = 32
    max_epochs = 200
    fast_dev_run = False

    import random
    from sklearn.datasets import fetch_openml
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import QuantileTransformer

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    adult = fetch_openml(data_id = 1590, data_home='./data_cache')

    data = adult.data

    le = LabelEncoder()
    label = pd.Series(le.fit_transform(adult.target))


    category_cols = ['workclass', 'education', 'race', 'sex', "marital-status", "occupation", "relationship", "native-country"]
    continuous_cols = [x for x in data.columns if x not in category_cols]

    for col in category_cols:
        data.loc[:, col] = le.fit_transform(data[col])

    transformer = QuantileTransformer(random_state=random_seed)
    data.loc[:, continuous_cols] = transformer.fit_transform(data[continuous_cols])

    cat_dims = []
    for col in category_cols:
        cat_dims.append(len(set(data[col].values)))

    from sklearn.model_selection import train_test_split

    data, _, label, _ = train_test_split(data, label, train_size = 0.2, random_state=random_seed, stratify=label)
    
    X_train, X_test, y_train, y_test = train_test_split(data, label, train_size = 0.8, random_state=random_seed, stratify=label)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size = 0.75, random_state=random_seed, stratify=y_train)

    import torch


    from torch.utils.data import DataLoader, SequentialSampler

    import sys
    sys.path.append('..')

    from utils.data import PTARLDataset, PTARLDataModule
        
    train_ds = PTARLDataset(X_train, y_train, continuous_cols, category_cols)
    valid_ds = PTARLDataset(X_valid, y_valid, continuous_cols, category_cols)
    test_ds = PTARLDataset(X_test, y_test, continuous_cols, category_cols)
    dm = PTARLDataModule(train_ds, valid_ds, test_ds, batch_size)



    model_hparams = {
        "cont_nums": len(continuous_cols), "cat_dims": cat_dims, "feat_emb_dim": 128, "n_hiddens": 3, "emb_dim": 256, "dropout": 0.1, 

    }


    optim = "AdamW"
    optim_hparams = {
        "lr" : 0.0001,
        "weight_decay" : 0.00005
    }
    
    loss_fn = torch.nn.CrossEntropyLoss
    loss_fn_hparams = {}


    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    callbacks = [
            EarlyStopping(
                monitor= "val_accuracy_score", 
                mode = 'max',
                patience = early_stopping_patience,
                verbose = False
            )
        ]

    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy_score",
        filename='{epoch:02d}-{val_f1:.4f}',
        save_top_k=1,
        mode = 'max'
    )

    callbacks.append(checkpoint_callback)

    trainer = Trainer(
                    devices = devices,
                    accelerator = accelerator,
                    max_epochs = max_epochs,
                    num_sanity_val_steps = 2,
                    callbacks = callbacks,
                    # deterministic=True,
                    fast_dev_run=fast_dev_run,
    )

    from torch.utils.data import DataLoader
    gp_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=False, sampler = SequentialSampler(train_ds), num_workers=n_jobs, drop_last=False)

    from backbones.mlp import MLP
    from ptarl_lightning import PTARLLightning
    ptarl = PTARLLightning(data.shape[1], model_hparams["emb_dim"], 2, gp_dl,
                        MLP, model_hparams, optim, optim_hparams, None, {}, loss_fn,  loss_fn_hparams, False, 0.1, 50,"euclidean",
                        {
                            "task" : 1.0,
                            "projection" : 1.0,
                            "diversifying" : 0.5,
                            "orthogonalization" : 2.5
                        },
                    random_seed)
    ptarl.set_first_phase()
    trainer.fit(ptarl, dm)
    ptarl = PTARLLightning.load_from_checkpoint(checkpoint_callback.best_model_path)


    from sklearn.metrics import accuracy_score
    preds = trainer.predict(ptarl, DataLoader(train_ds, 128))
    y_hat = []
    for pred in preds:
        y_hat.append(pred.argmax(1))
    y_hat = torch.concat(y_hat).numpy()
    print("W/O PTARL Train score:", accuracy_score(y_hat, y_train))

    preds = trainer.predict(ptarl, DataLoader(valid_ds, 128))
    y_hat = []
    for pred in preds:
        y_hat.append(pred.argmax(1))
    y_hat = torch.concat(y_hat).numpy()

    print("W/O PTARL Valid score:", accuracy_score(y_hat, y_valid))

    preds = trainer.predict(ptarl, DataLoader(test_ds, 128))
    y_hat = []
    for pred in preds:
        y_hat.append(pred.argmax(1))
    y_hat = torch.concat(y_hat).numpy()
    print("W/O PTARL Test score:", accuracy_score(y_hat, y_test))

    callbacks = [
            EarlyStopping(
                monitor= "val_accuracy_score", 
                mode = 'max',
                patience = early_stopping_patience,
                verbose = False
            )
        ]

    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy_score",#"val_accuracy_score",
        filename='{epoch:02d}-{val_f1:.4f}',
        save_top_k=1,
        mode = 'max'
    )

    callbacks.append(checkpoint_callback)

    trainer = Trainer(
                    devices = devices,
                    accelerator = accelerator,
                    max_epochs = max_epochs,
                    # min_epochs = 70,
                    num_sanity_val_steps = 2,
                    callbacks = callbacks,
                    # deterministic=True,
                    fast_dev_run=fast_dev_run,
    )
    ptarl.set_second_phase()
    trainer.fit(ptarl, dm)

    ptarl = PTARLLightning.load_from_checkpoint(checkpoint_callback.best_model_path)
    ptarl.set_second_phase()


    preds = trainer.predict(ptarl, DataLoader(train_ds, 128), ckpt_path='best')
    y_hat = []
    for pred in preds:
        y_hat.append(pred.argmax(1))
    y_hat = torch.concat(y_hat).numpy()
    print("PTARL Train score:", accuracy_score(y_hat, y_train))

    preds = trainer.predict(ptarl, DataLoader(valid_ds, 128), ckpt_path='best')
    y_hat = []
    for pred in preds:
        y_hat.append(pred.argmax(1))
    y_hat = torch.concat(y_hat).numpy()
    print("PTARL Valid score:", accuracy_score(y_hat, y_valid))

    preds = trainer.predict(ptarl, DataLoader(test_ds, 128), ckpt_path='best')
    y_hat = []
    for pred in preds:
        y_hat.append(pred.argmax(1))
    y_hat = torch.concat(y_hat).numpy()
    print("PTARL Test score:", accuracy_score(y_hat, y_test))
    
if __name__ == "__main__":
    test_mlp_adult()