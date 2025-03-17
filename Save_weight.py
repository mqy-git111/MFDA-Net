import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import load_model
from DA_attention import CAM, PAM
import combo_loss



model = load_model('A_V_att_deconcat.hdf5', custom_objects={'CAM':CAM, 'PAM':PAM, 'Combo_loss': combo_loss.Combo_loss,
                                                            'Dice': combo_loss.Dice})
model.summary()
model.save_weights('A_V_att_deconcat_weights.h5')