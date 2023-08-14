import config
from pytorch_lightning.callbacks import Callback, EarlyStopping, GradientAccumulationScheduler, ModelCheckpoint


class PrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("====================\nStarting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.\n====================")

print_callback = PrintingCallback()

earlystopper = EarlyStopping(monitor="valid_loss")

accumulator = GradientAccumulationScheduler(
    scheduling={0: config.gradient_accumulation_steps}
)

# checkpoint_callback = ModelCheckpoint(
#     filename="lol", #filename,
#     dirpath="path", #save_model_path,
#     every_n_val_epochs=2,
#     save_last=True,
#     save_top_k=2,
#     monitor='Contrastive loss_epoch',
#     mode='min'
# )