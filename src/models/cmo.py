
from src.models.rnn import RNNModule

class UnetModule(RNNModule):

    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)


    def RNN_test_step(self, batch, step_name: str):
        logs = {} 
        ims = batch["images"][:, :self.model.configs.pre_seq_length]
        lables = batch["images"][:, self.model.configs.pre_seq_length:]
        img_gen = self.forward(ims)
        loss = self.loss(img_gen, lables)
        logs[f"{step_name}/loss"] = loss

        return img_gen, logs
    
    def RNN_train_step(self, batch,  eta=1.0, num_updates=0,**kwargs):
        logs = {}
        ims = batch["images"][:, :self.model.configs.pre_seq_length]
        lables = batch["images"][:, self.model.configs.pre_seq_length:]
        img_gen = self.forward(ims)
        loss = self.loss(img_gen, lables)
        logs["train/loss"] = loss 
        return loss, logs
