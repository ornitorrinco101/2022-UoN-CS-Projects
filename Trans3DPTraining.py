import sys
sys.path.append("./data_loaders.py")
from data_loaders import PvtxDataModule
sys.path.append("./TransfRegModel.py")
import TransfRegModel
#import uproot
#import yaml
#from timeit import default_timer as timer
#import awkward as ak
# import numpy as np

#import torch
#import torch.nn as nn

#from torch.utils.data import random_split, Dataset, DataLoader

import lightning as L
import os
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

def main():
    in_dim = 3
    emb_dim = 256
    nhead_dim = 16
    num_encoder = 8
    ff_dim = 32
    reg_dim = 128
    dropout_e=0.0
    lr_v = 2.5e-3
    #warmup_v = 50
    #max_it = 2000
    
    
    device = "cuda:0"
    
    #TrF = TransfRegModel.Transformer(d_model=emb_dim,nhead=nhead_dim,num_encoder_layers=num_encoder)
    #AttPool = TransfRegModel.AttentionPooling(hidden_dim=emb_dim)
    Model_TrF = TransfRegModel.LightningTrans3DPointReg(dim_index=in_dim,dim_emb=emb_dim,
                                                        dim_nhead=nhead_dim,num_encoder=num_encoder,dim_ff=ff_dim,dropout_enc=dropout_e,
                                                        dim_reg=reg_dim,
                                                        lr=lr_v,)#warmup=warmup_v,max_iters=max_it)

    Model_TrF = Model_TrF.to(device)
    
    rootfile= ["../SimG4S_G4Sol_H3L_MOREdiv0_SimExpPara.root","../SimG4S_G4Sol_H3L_MOREdiv1_SimExpPara.root","../SimG4S_G4Sol_H3L_MOREdiv10_SimExpPara.root","../SimG4S_G4Sol_H3L_MOREdiv11_SimExpPara.root","../SimG4S_G4Sol_H3L_MOREdiv12_SimExpPara.root","../SimG4S_G4Sol_H3L_MOREdiv13_SimExpPara.root","../SimG4S_G4Sol_H3L_MOREdiv14_SimExpPara.root","../SimG4S_G4Sol_H3L_MOREdiv15_SimExpPara.root","../SimG4S_G4Sol_H3L_MOREdiv16_SimExpPara.root","../SimG4S_G4Sol_H3L_MOREdiv17_SimExpPara.root","../SimG4S_G4Sol_H3L_MOREdiv18_SimExpPara.root","../SimG4S_G4Sol_H3L_MOREdiv19_SimExpPara.root"]
    list_inputs= ["FiberD3_Core_log_x.HitPosX","FiberD3_Core_log_x.HitPosY","FiberD3_Core_log_x.HitPosZ","MiniFiberD1_Core_log_x.HitPosX","MiniFiberD1_Core_log_x.HitPosY","MiniFiberD1_Core_log_x.HitPosZ","MiniFiberD2_Core_log_x.HitPosX","MiniFiberD2_Core_log_x.HitPosY","MiniFiberD2_Core_log_x.HitPosZ"]#"FiberD4_Core_log_x.HitPosX","FiberD4_Core_log_x.HitPosY","FiberD4_Core_log_x.HitPosZ","FiberD5_Core_log_x.HitPosX","FiberD5_Core_log_x.HitPosY","FiberD5_Core_log_x.HitPosZ"]
    list_outputs= ['InteractionPoint_Z']
    InOut_dim= [[768,768,768,512,512,512,512,512,512,1], [0,768,1536,2304,2816,3328,3840,4352,4864], [[256,256], [256,256], [256,256], [256,256], [256,512], [256,512], [256,512], [256,512]] ]
    #valDigi= [194.62, 0.05, 3.]
    #valDigi = [194.62,3.]
    valDigi = [196.12,1.5]

    #cuts= [""]
    TogetherIn_dim= [[2304,1536,1536,1],[0,2304,3840]]
    batch_size= 2048

    
    #num_workers= 12
    #entries: [-1, -1]
    nametree= "G4Tree"
    dataset= "3DPointNestedTensor"
    #"Embedding-V0"
    max_mult= 60

    module2 = PvtxDataModule(rootfile,nametree,list_inputs,list_outputs,valDigi,InOut_dim,TogetherIn_dim,batch_size,2,max_mult,dataset)
    module2.prepare_data()
    module2.setup()
    
    train_data =  module2.train_dataloader()
    val_data =  module2.val_dataloader()
    #test_data = module2.test_dataloader()

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join("./", "Reg3DPTask")
    os.makedirs(root_dir, exist_ok=True)

    logger = CSVLogger("./RegTask3DP/logs", name="FirstTrial-agni-Z")

    trainer = L.Trainer(default_root_dir=root_dir,
                        callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")],
                        accelerator="gpu",
                        devices=[0],
                        max_epochs=150,
                        gradient_clip_val=5,
#                        accumulate_grad_batches=4,
#                        precision="16-mixed",
                        logger=logger,
                        )
    #    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    trainer.fit(Model_TrF,train_data,val_data)

    # Test best model on validation and test set
    # val_result = trainer.validate(datamodule=module2, verbose=False)
    #test_result = trainer.test(datamodule=module2) #, verbose=False)
    #result = {"test_loss": test_result[0]["test_loss"]}#, "val_loss": val_result[0]["val_loss"]}

    #print(result)
#model = Model.to(device)


if __name__ == "__main__":
    main()
        
