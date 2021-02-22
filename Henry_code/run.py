from tensorflow.keras.optimizers import RMSprop, Adam, SGD

from Utils.load_binarized_mnist import x_train, x_test, train_ds, test_ds, image_dim
X_train = x_train
X_test = x_test

from Henry_code.encoder_decoder import encoder, decoder
from Henry_code.iaf_vae import IAF_VAE

if __name__ == "__main__":
    enc = encoder()
    dec = decoder()
    iaf = IAF_VAE(enc, dec)
    iaf.compile(optimizer=Adam(learning_rate=0.001))
    iaf([x_test[0:10, :, :, :]])
    # iaf.summary()
    #iaf.fit(X_train, epochs=5, batch_size=128, validation_data=(X_test, X_test))