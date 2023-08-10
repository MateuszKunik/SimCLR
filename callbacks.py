from pytorch_lightning.callbacks import  Callback, GradientAccumulationScheduler

class PrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("====================\nStarting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.\n====================")


