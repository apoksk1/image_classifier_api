import tensorflow as tf
import uvicorn
from fastapi import FastAPI

from data import Data
from train_save import train_save

app = FastAPI()


@app.get("/")
def train_model(model_path, data_folder_path, epoch: int):
    data = Data()
    img_data = data.get_data(data_folder_path)

    batch_size = 32
    img_height = 180
    img_width = 180

    train_ds = data.get_train_ds(img_data, img_height, img_width, batch_size)
    val_ds = data.get_val_ds(img_data, img_height, img_width, batch_size)
    class_names = data.get_class_names(train_ds)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    train_save(class_names, train_ds, val_ds, epoch, img_width, batch_size, img_height, model_path)

    return {"message": "Model training completed."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
