import uproot
#import yaml
from timeit import default_timer as timer
import awkward as ak
# import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import random_split, Dataset, DataLoader

import lightning as L
import numpy as np
#import pickle


class RootPvtxIndexFull_Multiple_Files_PadIndex(Dataset):
    """
    Works for one ROOT many input files, loads one in 2 seconds (decent speed).
    All inputs from all detectors in one tensor.
    """

    def __init__(self,
                 rootfile,nametree,
                 list_inputs,list_outputs,
                 valDigi,
                 InOut_dim):
        """
        ROOT TTree of primary vtx study, tests, and configured with .yaml file
        (see configs/experiment, or in this case 00-FirstCodeTests/testing_vtx")
        """
        start=timer()

        self.rootfilename=rootfile
        self.tree=nametree
        self.MLIn=list_inputs
        self.MLOut = list_outputs
        self.VtxZDigi=valDigi

        self.Br_inputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLIn,
            library="ak"
        )
        self.Br_outputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLOut,
            library="ak"
        )

        self.InOut_dim = InOut_dim[0]
        self.InOut_dimOffset = InOut_dim[1]
        self.totIn_dim = sum(InOut_dim[0][:-1])
        self.totOut_dim = InOut_dim[0][-1]

        length=len(self.Br_inputs)

        #self.sampleIn = torch.zeros([length, 150])
        # self.sampleOut = torch.zeros([len(self.Br_inputs), self.totOut_dim])



        arr=ak.to_list(self.Br_inputs)
        values_only = [list(event.values()) for event in arr]

        self.lengths = np.array([
            [len(sublist) for sublist in event]  # Lengths for each event
            for event in values_only  # Iterate over events
            ],dtype='i')

        max_le = np.max(self.lengths)
        
        self.sampleOut=torch.tensor(self.Br_outputs[:][self.MLOut[0]]-self.VtxZDigi[0],dtype=torch.float32)/self.VtxZDigi[1]

        end=timer()
        print(f"Data loaded in: {end-start:.2f} seconds, trying PadIndex config for {length} In Samples")

        padded_batch = []

        for ievent in values_only:
            tensors = [torch.tensor(seq + [-1]*(max_le-len(seq))) for seq in ievent]
            padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=-1)
            padded_batch.append(padded)

        self.sampleIn = torch.stack(padded_batch,dim=0)

        end=timer()

        self.nfiles=len(self.rootfilename)
        self.ninputs=len(self.sampleIn[0])
        
        print(f"Flat config solved for {self.nfiles} files of length {self.ninputs} in {end-start:.2f} seconds")

    def __len__(self):
        return len(self.Br_inputs)

    def __getitem__(self, idx:int):
        return [[self.sampleIn[idx], self.lengths[idx]], self.sampleOut[idx]]


class RootPvtxIndexFull_Multiple_Files_NestedTensor(Dataset):
    """
    Works for one ROOT many input files, loads one in 2 seconds (decent speed).
    All inputs from all detectors in one tensor.
    """

    def __init__(self,
                 rootfile,nametree,
                 list_inputs,list_outputs,
                 valDigi,
                 InOut_dim):
        """
        ROOT TTree of primary vtx study, tests, and configured with .yaml file
        (see configs/experiment, or in this case 00-FirstCodeTests/testing_vtx")
        """
        start=timer()

        self.rootfilename=rootfile
        self.tree=nametree
        self.MLIn=list_inputs
        self.MLOut = list_outputs
        self.VtxZDigi=valDigi

        self.Br_inputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLIn,
            library="ak"
        )
        self.Br_outputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLOut,
            library="ak"
        )

        self.InOut_dim = InOut_dim[0]
        self.InOut_dimOffset = InOut_dim[1]
        self.totIn_dim = sum(InOut_dim[0][:-1])
        self.totOut_dim = InOut_dim[0][-1]

        length=len(self.Br_inputs)

        #self.sampleIn = torch.zeros([length, 150])
        # self.sampleOut = torch.zeros([len(self.Br_inputs), self.totOut_dim])



        arr=ak.to_list(self.Br_inputs)
        values_only = [list(event.values()) for event in arr]

        self.lengths = np.array([
            [len(sublist) for sublist in event]  # Lengths for each event
            for event in values_only  # Iterate over events
            ],dtype='i')

        #max_le = np.max(self.lengths)

        if len(self.VtxZDigi) == 2:
            self.sampleOut=torch.tensor(self.Br_outputs[:][self.MLOut[0]]-self.VtxZDigi[0],dtype=torch.float32)/self.VtxZDigi[1]
        else:

            Zbin = self.Br_outputs[:][self.MLOut[0]].to_numpy()
            arr2 = ((Zbin-self.VtxZDigi[0])//self.VtxZDigi[1])*self.VtxZDigi[1]+np.random.normal(0,self.VtxZDigi[1]*0.1,Zbin.shape[0])
            self.sampleOut = torch.tensor(arr2,dtype=torch.float32)


        end=timer()
        print(f"Data loaded in: {end-start:.2f} seconds, trying NestedTensor config for {length} In Samples")

        nested_batch = []
        pos_batch = []
        for ievent in values_only:
            tensors = [torch.tensor(seq, dtype=torch.int32) + 1000*int(i_s) for i_s,seq in enumerate(ievent)]
            pos = [torch.tensor([i_t]*len(tt),dtype=torch.int32) for i_t,tt in enumerate(tensors)]
            nested_batch.append(torch.concat(tensors))
            pos_batch.append(torch.concat(pos))
            
        self.sampleIn = nested_batch
        self.sampleInPos = pos_batch
        
        end=timer()

        self.nfiles=len(self.rootfilename)
        self.ninputs=len(self.sampleIn[0])
        
        print(f"Flat config solved for {self.nfiles} files of length {self.ninputs} in {end-start:.2f} seconds")

    def __len__(self):
        return len(self.Br_inputs)

    def __getitem__(self, idx:int):
        return [[self.sampleIn[idx], self.sampleInPos[idx]], self.sampleOut[idx]]




class RootPvtxIndexFull_Multiple_Files_3DPoint_NestedTensor(Dataset):
    """
    Works for one ROOT many input files, loads one in 2 seconds (decent speed).
    All inputs from all detectors in one tensor.
    """

    def __init__(self,
                 rootfile,nametree,
                 list_inputs,list_outputs,
                 valDigi,
                 InOut_dim):
        """
        ROOT TTree of primary vtx study, tests, and configured with .yaml file
        (see configs/experiment, or in this case 00-FirstCodeTests/testing_vtx")
        """
        start=timer()

        self.rootfilename=rootfile
        self.tree=nametree
        self.MLIn=list_inputs
        self.MLOut = list_outputs
        self.VtxZDigi=valDigi

        self.Br_inputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLIn,
            library="ak"
        )
        self.Br_outputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLOut,
            library="ak"
        )

        self.InOut_dim = InOut_dim[0]
        self.InOut_dimOffset = InOut_dim[1]
        self.totIn_dim = sum(InOut_dim[0][:-1])
        self.totOut_dim = InOut_dim[0][-1]

        length=len(self.Br_inputs)

        #self.sampleIn = torch.zeros([length, 150])
        # self.sampleOut = torch.zeros([len(self.Br_inputs), self.totOut_dim])



        arr=ak.to_list(self.Br_inputs)
        values_only = []
        for event in arr:
            temp_list=list(event.values())
            dim_list = len(temp_list)
            temp_val = []
            for i_e in range(dim_list//3):
                temp_val.append(list(zip(temp_list[i_e*3],temp_list[i_e*3+1],temp_list[i_e*3+2])))

            values_only.append(temp_val)

        self.lengths = np.array([
            [len(sublist) for sublist in event]  # Lengths for each event
            for event in values_only  # Iterate over events
            ],dtype='float')

        #max_le = np.max(self.lengths)

        if len(self.VtxZDigi) == 2:
            self.sampleOut=torch.tensor(self.Br_outputs[:][self.MLOut[0]]-self.VtxZDigi[0],dtype=torch.float32)/self.VtxZDigi[1]
        else:

            Zbin = self.Br_outputs[:][self.MLOut[0]].to_numpy()
            arr2 = ((Zbin-self.VtxZDigi[0])//self.VtxZDigi[1])*self.VtxZDigi[1]+np.random.normal(0,self.VtxZDigi[1]*0.1,Zbin.shape[0])
            self.sampleOut = torch.tensor(arr2,dtype=torch.float32)


        end=timer()
        print(f"Data loaded in: {end-start:.2f} seconds, trying 3D point NestedTensor config for {length} In Samples")

        nested_batch = []
        pos_batch = []
        
        for ievent in values_only:
            tensors = [torch.tensor(seq,dtype=torch.float32) for i_s,seq in enumerate(ievent)]
            pos = [torch.tensor([i_t]*len(tt),dtype=torch.int32) for i_t,tt in enumerate(tensors)]
            nested_batch.append(torch.concat(tensors))
            pos_batch.append(torch.concat(pos))
            
        self.sampleIn = nested_batch
        self.sampleInPos = pos_batch
        end=timer()

        self.nfiles=len(self.rootfilename)
        self.ninputs=len(self.sampleIn[0])

        print(f"Flat config solved for {self.nfiles} files of length {self.ninputs} in {end-start:.2f} seconds")

    def __len__(self):
        return len(self.Br_inputs)

    def __getitem__(self, idx:int):
        return [[self.sampleIn[idx], self.sampleInPos[idx]], self.sampleOut[idx]]


class RootPvtxIndexFull_Multiple_Files_RZPoint_NestedTensor(Dataset):
    """
    Works for one ROOT many input files, loads one in 2 seconds (decent speed).
    All inputs from all detectors in one tensor.
    """

    def __init__(self,
                 rootfile,nametree,
                 list_inputs,list_outputs,
                 valDigi,
                 InOut_dim):
        """
        ROOT TTree of primary vtx study, tests, and configured with .yaml file
        (see configs/experiment, or in this case 00-FirstCodeTests/testing_vtx")
        """
        start=timer()

        self.rootfilename=rootfile
        self.tree=nametree
        self.MLIn=list_inputs
        self.MLOut = list_outputs
        self.VtxZDigi=valDigi

        self.Br_inputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLIn,
            library="ak"
        )
        self.Br_outputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLOut,
            library="ak"
        )

        self.InOut_dim = InOut_dim[0]
        self.InOut_dimOffset = InOut_dim[1]
        self.totIn_dim = sum(InOut_dim[0][:-1])
        self.totOut_dim = InOut_dim[0][-1]

        length=len(self.Br_inputs)

        #self.sampleIn = torch.zeros([length, 150])
        # self.sampleOut = torch.zeros([len(self.Br_inputs), self.totOut_dim])


        arr_out = self.Br_outputs[:][self.MLOut[0]]

        arr=ak.to_list(self.Br_inputs)
        values_only = []
        for event in arr:
            temp_list=list(event.values())
            dim_list = len(temp_list)
            temp_val = []
            for i_e in range(dim_list//3):
                temp_val.append(list(zip(math.sqrt(temp_list[i_e*3]*temp_list[i_e*3]+temp_list[i_e*3+1]*temp_list[i_e*3+1]),temp_list[i_e*3+2])))

            values_only.append(temp_val)

        self.lengths = np.array([
            [len(sublist) for sublist in event]  # Lengths for each event
            for event in values_only  # Iterate over events
            ],dtype='float')

        #max_le = np.max(self.lengths)

        if len(self.VtxZDigi) == 2:
            self.sampleOut=torch.tensor(self.Br_outputs[:][self.MLOut[0]]-self.VtxZDigi[0],dtype=torch.float32)/self.VtxZDigi[1]
        else:
            Zbin = self.Br_outputs[:][self.MLOut[0]].to_numpy()
            off = self.VtxZDigi[2]/3.
            scale = self.VtxZDigi[2]/2.
            arr2 = ((Zbin-self.VtxZDigi[0])//self.VtxZDigi[1])*self.VtxZDigi[1]/scale+np.random.normal(0,self.VtxZDigi[1]*0.1/scale,Zbin.shape[0])+off
            self.sampleOut = torch.tensor(arr2,dtype=torch.float32)


        end=timer()
        print(f"Data loaded in: {end-start:.2f} seconds, trying 3D point NestedTensor config for {length} In Samples")

        nested_batch = []
        pos_batch = []
        
        for ievent in values_only:
            tensors = [torch.tensor(seq,dtype=torch.float32) for i_s,seq in enumerate(ievent)]
            pos = [torch.tensor([i_t]*len(tt),dtype=torch.int32) for i_t,tt in enumerate(tensors)]
            nested_batch.append(torch.concat(tensors))
            pos_batch.append(torch.concat(pos))
            
        self.sampleIn = nested_batch
        self.sampleInPos = pos_batch
        end=timer()

        self.nfiles=len(self.rootfilename)
        self.ninputs=len(self.sampleIn[0])

        print(f"Flat config solved for {self.nfiles} files of length {self.ninputs} in {end-start:.2f} seconds")

    def __len__(self):
        return len(self.Br_inputs)

    def __getitem__(self, idx:int):
        return [[self.sampleIn[idx], self.sampleInPos[idx]], self.sampleOut[idx]]

    

class RootPvtxIndexFull_Multiple_Files(Dataset):
    """
    Works for one ROOT many input files, loads one in 2 seconds (decent speed).
    All inputs from all detectors in one tensor.
    """

    def __init__(self,
                 rootfile,nametree,
                 list_inputs,list_outputs,
                 valDigi,
                 InOut_dim):
        """
        ROOT TTree of primary vtx study, tests, and configured with .yaml file 
        (see configs/experiment, or in this case 00-FirstCodeTests/testing_vtx")
        """
        start=timer()

        self.rootfilename=rootfile
        self.tree=nametree
        self.MLIn=list_inputs
        self.MLOut = list_outputs
        self.VtxZDigi=valDigi
        
        self.Br_inputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLIn,
            library="ak"
        )
        self.Br_outputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLOut,
            library="ak"
        )
        
        self.InOut_dim = InOut_dim[0]
        self.InOut_dimOffset = InOut_dim[1]
        self.totIn_dim = sum(InOut_dim[0][:-1])
        self.totOut_dim = InOut_dim[0][-1]

        length=len(self.Br_inputs)

        self.sampleIn = torch.zeros([length, self.totIn_dim])
        # self.sampleOut = torch.zeros([len(self.Br_inputs), self.totOut_dim])

        

        arr=ak.to_list(self.Br_inputs)
        values_only = [list(event.values()) for event in arr]

        lengths = np.array([
        [len(sublist) for sublist in event]  # Lengths for each event
        for event in values_only  # Iterate over events
        ],dtype='i')
        
        self.sampleOut=torch.tensor(self.Br_outputs[:][self.MLOut[0]]-self.VtxZDigi[0],dtype=torch.float32)/self.VtxZDigi[1]

        end=timer()
        print(f"Data loaded in: {end-start:.2f} seconds, trying ConCat config for {length} In Samples")
        
        for i in range(length):
            self.sampleIn[i][torch.from_numpy(np.concatenate(values_only[i])+np.repeat(self.InOut_dimOffset,lengths[i])).to(dtype=torch.int)]=1

        self.sampleIn=self.sampleIn.unsqueeze(dim=1)
        end=timer()

        self.nfiles=len(self.rootfilename)
        self.ninputs=len(self.sampleIn[0])
        
        print(f"Flat config solved for {self.nfiles} files of length {self.ninputs} in {end-start:.2f} seconds")

    def __len__(self):
        return len(self.Br_inputs)

    def __getitem__(self, idx:int):
        return [self.sampleIn[idx], self.sampleOut[idx]]


class RootPvtxSeparate_UFT_from_MFT_3_Tensors(Dataset):
    """
    Works for several ROOT files.
    A tensor of size (events,3) is outputed, dim=1 indexes detectors UFT3,MFT1, MFT2.
    Tensors for MFT1 and MFT2 are padded with -1 for tensor dimensional consistency.
    """

    def __init__(self,
                 rootfile,nametree,
                 list_inputs,list_outputs,
                 valDigi,
                 InOut_dim,
                 TogetherIn_dim):
        """ROOT TTree of primary vtx study, tests, and configured with .yaml file (see configs/experiment)"""
        start=timer()

        self.rootfilename=rootfile
        self.tree=nametree
        self.MLIn=list_inputs
        self.MLOut = list_outputs
        self.VtxZDigi=valDigi

        self.Br_inputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLIn,
            library="ak"
        )
        self.Br_outputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLOut,
            library="ak"
        )
        
        self.InOut_dim = InOut_dim[0]
        self.InOut_dimOffset = InOut_dim[1]
        self.totIn_dim = sum(InOut_dim[0][:-1])
        self.totOut_dim = InOut_dim[0][-1]
        self.TogetherIn_dim=TogetherIn_dim[0][:-1]
        

        self.sampleIn = torch.zeros([len(self.Br_inputs), self.totIn_dim])

        length=len(self.Br_inputs)

        arr=ak.to_list(self.Br_inputs)
        values_only = [list(event.values()) for event in arr]

        lengths = np.array([
        [len(sublist) for sublist in event]  # Lengths for each event
        for event in values_only  # Iterate over events
        ],dtype='i')
        
        self.sampleOut=torch.tensor(self.Br_outputs[:][self.MLOut[0]]-self.VtxZDigi[0],dtype=torch.float32)/self.VtxZDigi[1]

        end=timer()
        print(f"Data loaded in: {end-start:.2f} seconds, trying ConCat config for {length} In Samples")
        
        for i in range(length):
            self.sampleIn[i][torch.from_numpy(np.concatenate(values_only[i])+np.repeat(self.InOut_dimOffset,lengths[i])).to(dtype=torch.int)]=1
        end=timer()
        
        print(f"Flat config solved in {end-start:.2f} seconds. Separating the PaddingTon-3 form.")

        UFT3,MFT1,MFT2=torch.split(self.sampleIn,self.TogetherIn_dim,1)
            
        target_size=UFT3.size(1)
        
        # MFT1_padded=nn.functional.pad(MFT1, (0, target_size - MFT1.size(1)), value=-1)
        # MFT2_padded=nn.functional.pad(MFT2, (0, target_size - MFT2.size(1)), value=-1)
        MFT1_padded=nn.functional.pad(MFT1, (0, target_size - MFT1.size(1)), value=-1)
        MFT2_padded=nn.functional.pad(MFT2, (0, target_size - MFT2.size(1)), value=-1)
        
        self.sampleIn_separate=torch.stack([UFT3,MFT1_padded,MFT2_padded],dim=1)
        
        end=timer()
        print(f"PaddingTon-3 form completed in {end-start:.2f} seconds.")
        

    def __len__(self):
        return len(self.Br_inputs)

    def __getitem__(self,idx:int):
        return[self.sampleIn_separate[idx],self.sampleOut[idx]]

class RootPvtxSeparate_UFT_from_MFT_9_Tensors(Dataset):
    """
    Works for several ROOT files.
    A tensor of size (events,3) is outputed, dim=1 indexes detectors UFT3,MFT1, MFT2.
    Tensors for MFT1 and MFT2 are padded with -1 for tensor dimensional consistency.
    """

    def __init__(self,
                 rootfile,nametree,
                 list_inputs,list_outputs,
                 valDigi,
                 InOut_dim,
                 max_mult):
        """ROOT TTree of primary vtx study, tests, and configured with .yaml file (see configs/experiment)"""
        start=timer()

        self.rootfilename=rootfile
        self.tree=nametree
        self.MLIn=list_inputs
        self.MLOut = list_outputs
        self.VtxZDigi=valDigi

        self.Br_inputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLIn,
            library="ak"
        )
        self.Br_outputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLOut,
            library="ak"
        )
        
        self.InOut_dim = InOut_dim[0]
        self.InOut_dimOffset = InOut_dim[1]
        self.totIn_dim = sum(InOut_dim[0][:-1])
        self.totOut_dim = InOut_dim[0][-1]
        

        self.sampleIn = torch.full(size=[len(self.Br_inputs), self.totIn_dim],fill_value=-1)

        length=len(self.Br_inputs)

        arr=ak.to_list(self.Br_inputs)
        values_only = [list(event.values()) for event in arr]

        lengths = np.array([
        [len(sublist) for sublist in event]  # Lengths for each event
        for event in values_only  # Iterate over events
        ],dtype='i')
        
        self.sampleOut=torch.tensor(self.Br_outputs[:][self.MLOut[0]]-self.VtxZDigi[0],dtype=torch.float32)/self.VtxZDigi[1]

        end=timer()
        print(f"Data loaded in: {end-start:.2f} seconds, trying ConCat config for {length} In Samples")
        
        for i in range(length):
            self.sampleIn[i][torch.from_numpy(np.concatenate(values_only[i])+np.repeat(self.InOut_dimOffset,lengths[i])).to(dtype=torch.int)]=1
        end=timer()
        
        print(f"Flat config solved in {end-start:.2f} seconds. Separating the PaddingTon-9 form.")

        UFT3x,UFT3u,UFT3v,MFT1x,MFT1u,MFT1v,MFT2x,MFT2u,MFT2v=torch.split(self.sampleIn,self.InOut_dim[:-1],1)
            
        target_size=UFT3x.size(1)
        
        MFT1x_padded=nn.functional.pad(MFT1x, (0, target_size - MFT1x.size(1)), value=-1)
        MFT1u_padded=nn.functional.pad(MFT1u, (0, target_size - MFT1u.size(1)), value=-1)
        MFT1v_padded=nn.functional.pad(MFT1v, (0, target_size - MFT1v.size(1)), value=-1)

        MFT2x_padded=nn.functional.pad(MFT2x, (0, target_size - MFT2x.size(1)), value=-1)
        MFT2u_padded=nn.functional.pad(MFT2u, (0, target_size - MFT2u.size(1)), value=-1)
        MFT2v_padded=nn.functional.pad(MFT2v, (0, target_size - MFT2v.size(1)), value=-1)
        
        self.sampleIn_separate=torch.stack([UFT3x,UFT3u,UFT3v,MFT1x_padded,MFT1u_padded,MFT1v_padded,MFT2x_padded,MFT2u_padded,MFT2v_padded],
        dim=1)+1
        
        end=timer()
        print(f"PaddingTon-9 form completed in {end-start:.2f} seconds.")
        

    def __len__(self):
        return len(self.Br_inputs)

    def __getitem__(self,idx:int):
        return[self.sampleIn_separate[idx],self.sampleOut[idx]]


class RootPvtxSeparate_UFT_from_MFT_9_Tensors_Embedding_MaxMult(Dataset):
    """
    Works for several ROOT files.
    A tensor of size (events,9,max_mult) is outputed, dim=1 indexes detectors UFT3,MFT1, MFT2 on their separate x, u, v axis.
    All tensors are padded with -1 for tensor dimensional consistency to a set value of maximum multiplicity.
    Events with length longer than max_mult are omitted.
    """

    def __init__(self,
                 rootfile,nametree,
                 list_inputs,list_outputs,
                 valDigi,
                 InOut_dim,
                 max_mult):
        """ROOT TTree of primary vtx study, tests, and configured with .yaml file (see configs/experiment)"""
        start=timer()

        self.rootfilename=rootfile
        self.tree=nametree
        self.MLIn=list_inputs
        self.MLOut = list_outputs
        self.VtxZDigi=valDigi

        self.Br_inputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLIn,
            library="ak"
        )
        self.Br_outputs=uproot.concatenate(
            (f"{file}:{self.tree}" for file in self.rootfilename),
            expressions=self.MLOut,
            library="ak"
        )
        
        self.InOut_dim = InOut_dim[0]
        self.InOut_dimOffset = InOut_dim[1]
        self.totIn_dim = sum(InOut_dim[0][:-1])
        self.totOut_dim = InOut_dim[0][-1]
        
        self.max_mult=max_mult

        length=len(self.Br_inputs)

        arr=ak.to_list(self.Br_inputs)
        values_only = [list(event.values()) for event in arr]
        
        self.sampleOut=torch.tensor(self.Br_outputs[:][self.MLOut[0]]-self.VtxZDigi[0],dtype=torch.float32)/self.VtxZDigi[1]

        end=timer()

        print(f"Data loaded in: {end-start:.2f} seconds, limitting multiplicity to {self.max_mult} hits for {length} In Samples")
        
        all_subsublists = [subsublist for sublist in values_only for subsublist in sublist]

        length_all_subsublists = np.sum([len(subsublist)>self.max_mult for subsublist in all_subsublists])
        lost_values=100*length_all_subsublists/len(all_subsublists)


        filtered_subsublists = [subsublist if len(subsublist) <= self.max_mult else [] for subsublist in all_subsublists]

        end=timer()
        
        print(f"Multiplicity solved in {end-start:.2f} seconds. Padding all tensors to length {self.max_mult}.")
        print(f"Lost {lost_values:.3f} % of values.\n")

        padded_values=[nn.functional.pad(torch.tensor(subsublist), (0, self.max_mult - len(subsublist)), value=-1) for subsublist in filtered_subsublists]

        self.sampleIn_separate = torch.stack(padded_values).view(length, 9, self.max_mult)+torch.ones(length, 9, self.max_mult)

        flat=nn.Flatten(start_dim=0,end_dim=2)
        padding_percentage=100*(flat(self.sampleIn_separate)==0).sum()/(length*9*self.max_mult)
        
        end=timer()
        print(f"Padded form completed in {end-start:.2f} seconds.\nPadding percentage: {padding_percentage:.2f} %.\n")
        

    def __len__(self):
        return len(self.Br_inputs)

    def __getitem__(self,idx:int):
        return[self.sampleIn_separate[idx],self.sampleOut[idx]]



def Nested_collate(batch):
    inputs, target = list(zip(*batch))
    inputs1, inputs2 = list(zip(*inputs))
    return [[torch.nested.nested_tensor( inputs1,layout=torch.jagged), torch.nested.nested_tensor(inputs2,layout=torch.jagged)], torch.tensor(target).unsqueeze(1)]


class PvtxDataModule(L.LightningDataModule):
    """
    Loads the DataLoader we want by number:

    1: Original DataLoader
    2: Speed efficient 1 ROOT file concatenated input
    3: Speed efficient 1 ROOT file 3 tensor padded input
    4: Speed efficient many ROOT file concatenated input
    5: Speed efficient many ROOT file 3 tensor padded input (WIP!!)
    """

    def __init__(self,
                 rootfile,
                 nametree,
                 list_inputs,
                 list_outputs,
                 valDigi,
                 InOut_dim,
                 TogetherIn_dim,
                 batch_size,
                 num_workers,
                 max_mult,
                 dataset):
        """
        Args:
            rootfile (string): path of the root file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.rootfile=rootfile
        self.nametree=nametree
        self.list_inputs=list_inputs
        self.list_outputs=list_outputs
        self.valDigi=valDigi
        self.InOut_dim=InOut_dim
        self.TogetherIn_dim=TogetherIn_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_mult=max_mult
        self.dataset=dataset
        self.collect_fn = None

    def prepare_data(self):

        if self.dataset=="ConCat":
            self.ROOTDataset=RootPvtxIndexFull_Multiple_Files(self.rootfile,
                                                              self.nametree,
                                                              self.list_inputs,
                                                              self.list_outputs,
                                                              self.valDigi,
                                                              self.InOut_dim)
        if self.dataset=="PaddingTon-3":
            self.ROOTDataset=RootPvtxSeparate_UFT_from_MFT_3_Tensors(self.rootfile,
                                                                     self.nametree,
                                                                     self.list_inputs,
                                                                     self.list_outputs,
                                                                     self.valDigi,
                                                                     self.InOut_dim,
                                                                     self.TogetherIn_dim)
        if self.dataset=="PaddingTon-9":
            self.ROOTDataset=RootPvtxSeparate_UFT_from_MFT_9_Tensors(self.rootfile,
                                                                     self.nametree,
                                                                     self.list_inputs,
                                                                     self.list_outputs,
                                                                     self.valDigi,
                                                                     self.InOut_dim,
                                                                     self.max_mult)

        if self.dataset=="Embedding-V0":
            self.ROOTDataset=RootPvtxSeparate_UFT_from_MFT_9_Tensors_Embedding_MaxMult(self.rootfile,
                                                                                       self.nametree,
                                                                                       self.list_inputs,
                                                                                       self.list_outputs,
                                                                                       self.valDigi,
                                                                                       self.InOut_dim,
                                                                                       self.max_mult)
                                                                           
        if self.dataset=="PadIndex":
            self.ROOTDataset=RootPvtxIndexFull_Multiple_Files_PadIndex(self.rootfile,
                                                                       self.nametree,
                                                                       self.list_inputs,
                                                                       self.list_outputs,
                                                                       self.valDigi,
                                                                       self.InOut_dim)
                                                                         
        if self.dataset=="NestedTensor":
            self.ROOTDataset=RootPvtxIndexFull_Multiple_Files_NestedTensor(self.rootfile,
                                                                           self.nametree,
                                                                           self.list_inputs,
                                                                           self.list_outputs,
                                                                           self.valDigi,
                                                                           self.InOut_dim)
                                                                         
            self.collect_fn = Nested_collate

        if self.dataset=="3DPointNestedTensor":
            self.ROOTDataset=RootPvtxIndexFull_Multiple_Files_3DPoint_NestedTensor(self.rootfile,
                                                                                   self.nametree,
                                                                                   self.list_inputs,
                                                                                   self.list_outputs,
                                                                                   self.valDigi,
                                                                                   self.InOut_dim)

            self.collect_fn = Nested_collate

        
            
    def setup(self, stage=None):
        nb_events = len(self.ROOTDataset)
        nb_train = nb_events * 8 // 10
        nb_test = nb_events * 1 // 10
        nb_val = nb_events - nb_train - nb_test
        print("setup ", nb_events, " : ", nb_train, " | ", nb_test, " | ", nb_val)
        self.ROOTset_train, self.ROOTset_val, self.ROOTset_test = random_split(self.ROOTDataset, [nb_train, nb_val, nb_test])
        #return[self.ROOTset_train, self.ROOTset_val, self.ROOTset_test]

    def train_dataloader(self):
        return DataLoader(dataset=self.ROOTset_train,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn
                          )

    def val_dataloader(self):
        return DataLoader(self.ROOTset_val, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          collate_fn=self.collect_fn,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.ROOTset_test, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          collate_fn=self.collect_fn,
                          shuffle=False)
