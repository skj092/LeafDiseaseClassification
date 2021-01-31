import pandas as pd 
import dataset
import engine
from torch.utils.data import DataLoader
import torch 
import model 

if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv('input/sample_submission.csv')
    test_df['label'] = -1
    test_df['kfold'] = -1
    test_ds = dataset.LeafData(test_df, dir="input/test_images/")
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1)
    model_inf = model.get_model()
    model_inf.load_state_dict(torch.load('models/model.pt'))
    model_inf.to(device)
    prediction = engine.Predict(test_ds, test_dl, model_inf, device)
    test_df['label'] = prediction
    test_df = test_df.drop(['kfold'], axis=1)
    test_df.to_csv('submission.csv', index=False)
