import numpy as np

def compute_map_at_3(eval_predictions):
    preds, labels = eval_predictions
    acc = (np.argmax(preds, axis=1) == labels).astype(np.float32).mean().item()
    
    preds = np.asarray(preds)
    p = np.argsort(-preds, 1)
    map3 = (p[:, 0] == labels) + 1/2 * (p[:, 1] == labels) + 1/3 * (p[:, 2] == labels) 
    return {"MAP@3": np.mean(map3), "ACC": acc}